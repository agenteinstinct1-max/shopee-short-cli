"""Acesso HTTP com allowlist negativa, redirects controlados e fail-fast anti-bot."""
from __future__ import annotations

import re
from urllib.parse import urljoin, urlsplit

import requests

from .audit import AuditLog
from .errors import InvalidProductLink, SourceBlocked, ProviderError

_REDIRECTS = {301, 302, 303, 307, 308}
_SHOPEE_RE = re.compile(r"(?:^|\.)shopee\.", re.IGNORECASE)
_CAPTCHA_MARKERS = (
    # Match active interstitials, not ordinary sites that load optional CAPTCHA scripts/forms.
    "verify you are human", "prove you are human", "robot check", "unusual traffic",
    "challenge-platform", "cf-chl-", "checking your browser", "access denied",
    "captcha challenge", "complete the captcha", "complete captcha", "captcha verification",
    "please solve the captcha", "bot verification required", "sign in to confirm you’re not a bot",
)


def normalized_host(url: str) -> str:
    return (urlsplit(url).hostname or "").rstrip(".").lower()


def is_shopee_host(host: str) -> bool:
    return bool(_SHOPEE_RE.search(host.lower()))


def is_shopee_cdn_host(host: str) -> bool:
    host = host.lower().rstrip(".")
    return is_shopee_host(host) or host == "susercontent.com" or host.endswith(".susercontent.com")


def is_shp_ee_host(host: str) -> bool:
    return host.lower().rstrip(".") == "shp.ee"


def assert_safe_request_url(url: str, *, operation: str, allow_shp_ee: bool = False) -> str:
    parts = urlsplit(url)
    host = (parts.hostname or "").lower().rstrip(".")
    if parts.scheme.lower() != "https" or not host:
        raise SourceBlocked(f"URL insegura/inválida para {operation}; somente HTTPS é aceito.")
    if is_shopee_cdn_host(host):
        raise SourceBlocked(f"Bloqueado por política: a CLI não acessa {host}.")
    if is_shp_ee_host(host) and not allow_shp_ee:
        raise SourceBlocked("shp.ee só pode ser acessado pelo resolvedor de redirect do link curto.")
    return host


def _looks_like_challenge(response: requests.Response) -> bool:
    ctype = (response.headers.get("content-type") or "").lower()
    if "text/html" not in ctype and "application/json" not in ctype and "text/plain" not in ctype:
        return False
    try:
        body = response.content[:256_000].decode("utf-8", errors="ignore").lower()
    except Exception:
        return False
    return any(marker in body for marker in _CAPTCHA_MARKERS)


def safe_request(session: requests.Session, method: str, url: str, *, operation: str,
                 audit: AuditLog | None = None, timeout: tuple[int, int] = (12, 35),
                 allow_shp_ee: bool = False, stop_at_shopee_redirect: bool = False,
                 max_redirects: int = 5, **kwargs) -> requests.Response:
    """Executa request sem redirects automáticos; nunca segue para Shopee/CDN Shopee."""
    current = url
    for hop in range(max_redirects + 1):
        host = assert_safe_request_url(current, operation=operation, allow_shp_ee=allow_shp_ee)
        # A exceção shp.ee é exclusiva para resolver o redirect do próprio link.
        if allow_shp_ee and not is_shp_ee_host(host):
            raise InvalidProductLink("O redirect de shp.ee apontou para host não autorizado pelo resolvedor.")
        response = session.request(method, current, allow_redirects=False, timeout=timeout, **kwargs)
        if audit:
            audit.request(operation, current, method=method.upper(), status=response.status_code)
        if response.status_code in _REDIRECTS:
            location = response.headers.get("location")
            if not location:
                raise SourceBlocked(f"Redirect sem Location em {host}; interrompi sem seguir.")
            target = urljoin(current, location)
            target_host = normalized_host(target)
            if is_shopee_cdn_host(target_host):
                if stop_at_shopee_redirect:
                    # Retorna a resposta 3xx ao resolvedor; nenhum request foi feito ao destino.
                    response.headers["x-shopee-short-target"] = target
                    return response
                raise SourceBlocked(f"Redirect para {target_host} bloqueado; nenhum request foi feito ao destino.")
            if allow_shp_ee and not is_shp_ee_host(target_host):
                raise InvalidProductLink("shp.ee redirecionou para destino que não é um link de produto Shopee.")
            current = target
            continue
        if response.status_code in (403, 429):
            raise SourceBlocked(f"Fonte bloqueada (HTTP {response.status_code}) em {host}; encerrando sem retry/CAPTCHA/proxy.")
        if _looks_like_challenge(response):
            raise SourceBlocked(f"Página de desafio/CAPTCHA detectada em {host}; encerrando sem tentar resolver.")
        return response
    raise SourceBlocked("Limite de redirects excedido; interrompi sem continuar a cadeia.")


def safe_get(url: str, *, operation: str, audit: AuditLog | None = None,
             max_bytes: int | None = None, expected_content_prefix: str | None = None) -> tuple[bytes, requests.Response]:
    session = requests.Session()
    session.trust_env = False
    response = safe_request(session, "GET", url, operation=operation, audit=audit, stream=True)
    if response.status_code >= 400:
        raise SourceBlocked(f"Fonte respondeu HTTP {response.status_code} ({normalized_host(url)}).")
    content_type = (response.headers.get("content-type") or "").lower()
    if expected_content_prefix and not content_type.startswith(expected_content_prefix):
        # Alguns CDNs não informam MIME; aceitar bytes que sejam decodificados depois.
        if content_type.startswith("text/html"):
            raise SourceBlocked(f"O servidor de {normalized_host(url)} devolveu HTML em vez do arquivo esperado.")
    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_content(1024 * 128):
        if not chunk:
            continue
        total += len(chunk)
        if max_bytes and total > max_bytes:
            raise ProviderError(f"Arquivo excedeu o limite de {max_bytes} bytes.")
        chunks.append(chunk)
    data = b"".join(chunks)
    # Checagem adicional de HTML/challenge em respostas com MIME incorreto.
    sample = data[:256_000].decode("utf-8", errors="ignore").lower()
    if any(marker in sample for marker in _CAPTCHA_MARKERS):
        raise SourceBlocked(f"Desafio/CAPTCHA detectado no conteúdo de {normalized_host(url)}.")
    return data, response


def resolve_shp_ee(short_url: str, audit: AuditLog | None = None) -> str:
    """Resolve apenas redirects hospedados em shp.ee; nunca requisita o destino Shopee."""
    host = normalized_host(short_url)
    if not is_shp_ee_host(host):
        raise InvalidProductLink("O resolvedor só aceita links curtos no host shp.ee.")
    session = requests.Session()
    session.trust_env = False
    response = safe_request(
        session, "GET", short_url, operation="resolve_shp_ee", audit=audit,
        allow_shp_ee=True, stop_at_shopee_redirect=True, stream=True,
        headers={"User-Agent": "shopee-short/0.1 (short-link resolver; no Shopee fetch)"},
    )
    target = response.headers.get("x-shopee-short-target")
    if target:
        return target
    location = response.headers.get("location")
    if location:
        # Pode ser uma URL final recebida após cadeia de redirects internos shp.ee.
        target = urljoin(short_url, location)
        target_host = normalized_host(target)
        if is_shopee_host(target_host):
            return target
    raise InvalidProductLink("shp.ee não retornou um redirect direto para um produto Shopee; nada além de shp.ee foi acessado.")
