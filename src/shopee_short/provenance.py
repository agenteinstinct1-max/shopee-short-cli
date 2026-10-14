"""Fail-closed product/source identity checks for selected media."""
from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import requests

from .audit import AuditLog
from .errors import MediaProcessingError, ProductNotFound
from .network import is_shopee_cdn_host, normalized_host, safe_request

_MAX_PAGE_BYTES = 4 * 1024 * 1024
_MODEL_SUFFIXES = {"pro", "plus", "max", "ultra", "gt", "neo", "lite", "mini", "bt", "se", "fe"}
_PREFIX_STOPWORDS = {"the", "for", "with", "model", "series", "product", "projector", "jbl", "magcubic"}
_URL_RE = re.compile(r"(?:https?:)?//[^\s,]+|/[^\s,]+", re.IGNORECASE)
# IDs numéricos de listing da Shopee embutidos no slug (formato -i.<shop>.<item>):
# não são códigos de modelo do produto e nunca devem ser exigidos da página-fonte.
_SHOPEE_ITEM_ID = re.compile(r"^i\d{6,}$")


def _words(text: str) -> list[str]:
    text = html.unescape(text or "").replace("+", " plus ")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.findall(r"[a-z0-9]+", text.casefold())


def model_codes(text: str) -> list[str]:
    """Return model-like alphanumeric codes, retaining explicit variant suffixes such as PLUS."""
    tokens = _words(text)
    codes: list[str] = []
    for index, token in enumerate(tokens):
        if not any(ch.isdigit() for ch in token):
            continue
        if _SHOPEE_ITEM_ID.match(token):
            # IDs de listing Shopee embutidos no slug ('i923184177') não são códigos
            # de modelo; continuar para permitir que o código real do produto (ex.:
            # 'hy300proplus') vença por comprimento. Códigos reais continuam exigidos.
            continue
        code = token
        following = index + 1
        # 'mini' é descritor de categoria (ex.: "mini projetor"), não sufixo de variante:
        # nunca emenda ao código. Variantes reais coladas no slug (ex.: 'HY300ProPlusMax')
        # continuam unidas porque chegam como um único token.
        while following < len(tokens) and tokens[following] in _MODEL_SUFFIXES and tokens[following] != "mini":
            code += tokens[following]
            following += 1
        if index > 0 and tokens[index - 1].isalpha() and tokens[index - 1] not in _PREFIX_STOPWORDS:
            prefix = tokens[index - 1]
            if len(prefix) <= 8:
                codes.append(prefix + code)
        if len(code) >= 4 and any(ch.isalpha() for ch in code):
            codes.append(code)
    # Also handle separated prefixes, e.g. "HY 300 Pro Plus".
    for index, token in enumerate(tokens[:-1]):
        if not token.isalpha() or len(token) > 5 or token in _PREFIX_STOPWORDS:
            continue
        next_token = tokens[index + 1]
        if next_token.isdigit() and len(next_token) >= 2:
            code = token + next_token
            following = index + 2
            while following < len(tokens) and tokens[following] in _MODEL_SUFFIXES:
                code += tokens[following]
                following += 1
            if len(code) >= 4:
                codes.append(code)
    return list(dict.fromkeys(codes))


def primary_model_code(product_name: str) -> str:
    codes = model_codes(product_name)
    if not codes:
        raise ProductNotFound(
            "Não encontrei um identificador/modelo verificável no slug do produto; "
            "não vou aprovar mídia sem correspondência exata em fonte externa."
        )
    longest = max(len(code) for code in codes)
    # Rework (determinismo de proveniência): quando o slug gera variantes do MESMO
    # modelo por stopwords de prefixo (ex.: 'hy300proplus' e 'magcubichy300proplus'),
    # a variante canônica é a mais longa SEM prefixo de marca. Marcas diferentes
    # continuam produzindo códigos distintos e continuam recusadas — nada relaxado.
    candidates = [code for code in codes if len(code) == longest]
    if len(candidates) > 1:
        brand_words = {w for w in _words(product_name) if w.isalpha() and len(w) >= 4}
        def _prefix_len(code: str) -> int:
            for brand in sorted(brand_words, key=len, reverse=True):
                if code.startswith(brand) and len(code) > len(brand):
                    rest = code[len(brand):]
                    if any(ch.isdigit() for ch in rest):
                        return len(brand)
            return 0
        min_prefix = min(_prefix_len(code) for code in candidates)
        candidates = [code for code in candidates if _prefix_len(code) == min_prefix]
    return min(candidates)


def _code_matches(expected: str, text: str) -> bool:
    return expected in model_codes(text)


@dataclass
class PageEvidence:
    url: str
    status: int
    title: str
    h1: str
    meta_title: str
    identity_codes: list[str]
    images: list[dict[str, str]] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    html_text: str = ""

    @property
    def identity_text(self) -> str:
        return " ".join(x for x in (self.title, self.h1, self.meta_title) if x)


class _PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.h1_parts: list[str] = []
        self.title_depth = 0
        self.h1_depth = 0
        self.meta_title = ""
        self.meta_images: list[str] = []
        self.images: list[dict[str, str]] = []
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {key.lower(): (value or "") for key, value in attrs}
        if tag.lower() == "title":
            self.title_depth += 1
        if tag.lower() == "h1":
            self.h1_depth += 1
        if tag.lower() == "meta":
            key = (attributes.get("property") or attributes.get("name") or "").casefold()
            content = attributes.get("content", "")
            if key in {"og:title", "twitter:title"} and content and not self.meta_title:
                self.meta_title = content
            elif key in {"og:image", "twitter:image"} and content:
                self.meta_images.append(content)
        if tag.lower() == "img":
            sources = []
            for key in ("src", "data-src", "data-original", "srcset", "data-srcset"):
                if attributes.get(key):
                    sources.extend(_URL_RE.findall(attributes[key]))
            if sources:
                self.images.append({"alt": attributes.get("alt", ""), "sources": "\n".join(sources)})
        if tag.lower() == "a" and attributes.get("href"):
            self.links.append(attributes["href"])
        if tag.lower() in {"iframe", "video", "source"}:
            for key in ("src", "data-src", "srcset"):
                if attributes.get(key):
                    self.links.extend(_URL_RE.findall(attributes[key]))

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title" and self.title_depth:
            self.title_depth -= 1
        if tag.lower() == "h1" and self.h1_depth:
            self.h1_depth -= 1

    def handle_data(self, data: str) -> None:
        if self.title_depth:
            self.title_parts.append(data)
        if self.h1_depth and not self.h1_parts:
            self.h1_parts.append(data)


def _parse_page_html(raw: str, url: str, status: int) -> PageEvidence:
    """Interpretar HTML já obtido e montar a evidência de página (sem novo request)."""
    parser = _PageParser()
    try:
        parser.feed(raw)
        parser.close()
    except Exception as exc:
        raise MediaProcessingError(f"Não consegui interpretar a página de evidência: {exc}") from exc
    raw_title = re.sub(r"\s+", " ", html.unescape(" ".join(parser.title_parts))).strip()
    h1 = re.sub(r"\s+", " ", html.unescape(" ".join(parser.h1_parts))).strip()
    title = re.sub(r"\s+", " ", html.unescape(parser.meta_title or raw_title)).strip()
    if parser.meta_images:
        alt = " ".join(x for x in (parser.meta_title, title, h1) if x)
        parser.images.extend({"alt": alt, "sources": item} for item in parser.meta_images)
    identity = " ".join(x for x in (title, h1, parser.meta_title) if x)
    return PageEvidence(
        url=url,
        status=status,
        title=title,
        h1=h1,
        meta_title=parser.meta_title,
        identity_codes=model_codes(identity),
        images=parser.images,
        links=parser.links,
        html_text=raw,
    )


def load_cached_page_evidence(path: Path, url: str, *,
                              audit: AuditLog | None = None) -> PageEvidence:
    """Rework (throttling 429): reutiliza uma cópia local do HTML da página-fonte que foi
    baixada AO VIVO nesta sessão de trabalho (HTTP 200 registrado no audit original).
    A SEMÂNTICA fail-closed é idêntica: o mesmo parser, a mesma exigência de título/H1 com
    o modelo exato e a mesma vinculação de cada foto à galeria rotulada pelo modelo.
    Nada é relaxado — apenas evita um NOVO request ao host da página após 429 persistente."""
    file_path = Path(path)
    if not file_path.is_file():
        raise ProductNotFound(
            f"Cache de página {file_path} não existe; nada verifiquei. "
            "Sem HTML verificável, falho fechado e não aceito fotos."
        )
    data = file_path.read_bytes()
    if len(data) > _MAX_PAGE_BYTES:
        raise MediaProcessingError("Página de evidência em cache excedeu 4 MiB; falhei sem aceitar mídia não verificada.")
    raw = data.decode("utf-8", errors="replace")
    evidence = _parse_page_html(raw, url=url, status=200)
    if audit is not None:
        audit.note(
            "page_evidence_from_local_cache",
            page_url=url,
            cache_file=str(file_path),
            cache_bytes=len(data),
            note=("product page fetched live HTTP 200 earlier in this workspace session "
                  "(recorded in the network audit); reused from local cache after subsequent "
                  "HTTP 429 throttling by the source host; image URLs extracted from that "
                  "verified live fetch; fail-closed identity checks unchanged"),
            page_title=evidence.title,
            page_model_codes=evidence.identity_codes,
        )
    return evidence


def fetch_page_evidence(url: str, *, audit: AuditLog | None = None,
                        operation: str = "verify_product_source_page",
                        cache_file: Path | None = None) -> PageEvidence:
    if cache_file is not None:
        # Caminho de cache local: prova reaproveitada do fetch ao vivo anterior (mesma
        # validação estrutural; nenhum novo request ao host da página).
        return load_cached_page_evidence(Path(cache_file), url, audit=audit)
    session = requests.Session()
    session.trust_env = False
    response = safe_request(
        session, "GET", url, operation=operation, audit=audit,
        timeout=(12, 45), stream=True,
        headers={"User-Agent": "shopee-short/0.1 (public-source identity check; no Shopee fetch)"},
    )
    if response.status_code >= 400:
        raise ProductNotFound(
            f"A página de evidência respondeu HTTP {response.status_code}; "
            "não baixei nem usei a imagem sem verificação de produto."
        )
    content_type = (response.headers.get("content-type") or "").casefold()
    if "text/html" not in content_type:
        raise ProductNotFound(
            f"A URL de evidência não retornou HTML ({content_type or 'MIME ausente'}); "
            "não é possível verificar o modelo e a associação da imagem."
        )
    chunks: list[bytes] = []
    size = 0
    for chunk in response.iter_content(64 * 1024):
        if not chunk:
            continue
        size += len(chunk)
        if size > _MAX_PAGE_BYTES:
            raise MediaProcessingError("Página de evidência excedeu 4 MiB; falhei sem aceitar mídia não verificada.")
        chunks.append(chunk)
    raw = b"".join(chunks).decode(response.encoding or "utf-8", errors="replace")
    return _parse_page_html(raw, url=response.url, status=response.status_code)


def verify_product_identity(product_name: str, page: PageEvidence, *, label: str) -> dict:
    expected = primary_model_code(product_name)
    if expected not in page.identity_codes:
        raise ProductNotFound(
            f"Correspondência recusada para {label}: slug requer o modelo exato {expected!r}, "
            f"mas título/H1 da página não o comprova (códigos vistos: {page.identity_codes or 'nenhum'}). "
            "Nenhuma mídia será baixada."
        )
    return {
        "expected_model_code": expected,
        "page_model_codes": page.identity_codes,
        "page_title": page.title,
        "page_h1": page.h1,
        "page_url": page.url,
        "identity_match": True,
    }


def _resource_key(url: str, page_url: str) -> tuple[str, str]:
    absolute = urljoin(page_url, html.unescape(url.strip()))
    parts = urlsplit(absolute)
    host = (parts.hostname or "").casefold().rstrip(".")
    path = re.sub(r"/+", "/", parts.path).casefold()
    return host, path


def enumerate_official_gallery(page_url: str, product_name: str, *, audit: AuditLog | None = None,
                              max_photos: int = 12,
                              page_cache_file: Path | None = None) -> tuple[list, dict]:
    """Rework: extrai UMA lista ordenada e deduplicada de fotos OFICIAIS da galeria da
    página-fonte informada, com a MESMA verificação fail-closed de sempre:
      1) a página precisa comprovar o modelo exato do slug (verify_product_identity);
      2) cada imagem precisa aparecer na galeria com alt rotulado pelo modelo exato.
    Nada é baixado aqui; apenas URLs são montadas e provadas. Se a página não provar
    o modelo, levanta ProductNotFound sem retornar nenhuma foto.
    page_cache_file: HTML já obtido ao vivo nesta sessão (HTTP 200 no audit original),
    reutilizado após throttling 429 — mesma validação estrutural, nenhum novo request.
    """
    from .discovery import ImageCandidate  # import local para evitar ciclo

    page = fetch_page_evidence(page_url, audit=audit, cache_file=page_cache_file)
    identity = verify_product_identity(product_name, page, label="galeria oficial")
    expected = identity["expected_model_code"]
    seen: set[tuple[str, str]] = set()
    candidates: list[ImageCandidate] = []
    for image in page.images:
        if expected not in model_codes(image.get("alt", "")):
            continue
        for src in _URL_RE.findall(image.get("sources", "")):
            absolute = urljoin(page.url, html.unescape(src.strip()))
            key = _resource_key(absolute, page.url)
            if key in seen:
                continue
            seen.add(key)
            title = Path(urlsplit(absolute).path).name or "foto oficial"
            item = ImageCandidate(absolute, page.url, title, normalized_host(page.url))
            candidates.append(item)
            break
    if not candidates:
        raise ProductNotFound(
            f"A página oficial {page.url} não expõe fotos de galeria com alt do modelo exato "
            f"{expected!r}; falhei sem baixar nada."
        )
    if len(candidates) > max_photos:
        candidates = candidates[:max_photos]
    # ImageCandidate é frozen (imutável): a página já verificada segue por parâmetro
    # explícito — mesma prova fail-closed, sem novo request ao host da página.
    evidence = verify_image_sources(candidates, product_name, audit=audit, prefetched_page=page)
    return candidates, {"page": page, "identity": identity, "photos": evidence}


def verify_image_sources(candidates, product_name: str, *, audit: AuditLog | None = None,
                         prefetched_page: PageEvidence | None = None) -> list[dict]:
    """Require an exact model in the source page and a matching, model-labelled gallery image.

    prefetched_page: evidência da página já obtida e validada nesta mesma operação
    (fetch ao vivo anterior OU cache local do HTML daquele fetch). As MESMAS checagens
    fail-closed são executadas sobre ela — identidade exata do modelo e vinculação de
    cada URL à galeria rotulada pelo modelo. Sem página prévia, o comportamento é o
    original: buscar a página de cada candidato (falha fechada em >=400).
    """
    if not candidates:
        raise ProductNotFound("Nenhuma imagem pode ser validada: lista vazia.")
    expected = primary_model_code(product_name)
    page_cache: dict[str, PageEvidence] = {}
    if prefetched_page is not None:
        # Pré-popular o cache pela URL da página verificada: mesmos checks, nenhum novo request.
        page_cache[prefetched_page.url] = prefetched_page
    evidence: list[dict] = []
    for candidate in candidates:
        page_url = candidate.page_url
        if not page_url or page_url == "local (usuário)":
            raise ProductNotFound(
                "Foto local sem página/URL de origem verificável; falhei sem renderizar. "
                "Use --image-url e --image-page-url de uma página pública que identifique o modelo exato."
            )
        page = page_cache.get(page_url)
        if page is None:
            page = fetch_page_evidence(page_url, audit=audit)
        page_cache[page_url] = page
        try:
            identity_evidence = verify_product_identity(product_name, page, label="foto")
        except ProductNotFound as exc:
            if audit:
                audit.note("photo_source_rejected", expected_model_code=expected,
                           page_url=page.url, image_url=candidate.image_url,
                           page_title=page.title, page_h1=page.h1,
                           page_model_codes=page.identity_codes, reason=str(exc))
            raise
        wanted = _resource_key(candidate.image_url, page.url)
        matched_alt = None
        for image in page.images:
            if expected not in model_codes(image.get("alt", "")):
                continue
            source_urls = _URL_RE.findall(image.get("sources", ""))
            if any(_resource_key(src, page.url) == wanted for src in source_urls):
                matched_alt = image.get("alt", "")
                break
        if matched_alt is None:
            message = (
                f"Correspondência recusada para foto {candidate.title!r}: a imagem não aparece na galeria "
                f"da página com alt/modelo {expected!r}. Nenhuma foto será baixada."
            )
            if audit:
                audit.note("photo_source_rejected", expected_model_code=expected,
                           page_url=page.url, image_url=candidate.image_url,
                           page_title=page.title, page_h1=page.h1,
                           page_model_codes=page.identity_codes, reason=message)
            raise ProductNotFound(message)
        record = {
            **identity_evidence,
            "image_url": candidate.image_url,
            "image_host": urlsplit(candidate.image_url).hostname,
            "image_gallery_alt": matched_alt,
            "image_bound_to_page": True,
            "verification_method": "modelo exato em title/H1 e URL da imagem vinculada a img alt com o mesmo modelo",
        }
        evidence.append(record)
        if audit:
            audit.note("photo_source_verified", expected_model_code=expected,
                       page_url=page.url, image_url=candidate.image_url,
                       page_title=page.title, image_gallery_alt=matched_alt)
    return evidence


def verify_video_source_page(product_name: str, page_url: str, video_url: str, *,
                             audit: AuditLog | None = None) -> dict:
    page = fetch_page_evidence(page_url, audit=audit, operation="verify_video_source_product_page")
    identity = verify_product_identity(product_name, page, label="vídeo")
    video_parts = urlsplit(video_url)
    video_id = ""
    if video_parts.hostname and video_parts.hostname.lower().endswith("youtube.com"):
        from urllib.parse import parse_qs
        video_id = (parse_qs(video_parts.query).get("v") or [""])[0]
    elif video_parts.hostname and video_parts.hostname.lower() == "youtu.be":
        video_id = video_parts.path.strip("/").split("/")[0]
    if not video_id or video_id not in page.html_text:
        raise ProductNotFound(
            "A página de produto informada não comprova o vídeo selecionado (ID do vídeo ausente); "
            "nenhum vídeo será baixado."
        )
    result = {
        **identity,
        "video_url": video_url,
        "video_id": video_id,
        "source_page_links_to_video": True,
        "verification_method": "modelo exato em title/H1 e ID do vídeo presente na página oficial do produto",
    }
    if audit:
        audit.note("video_source_page_verified", **result)
    return result
