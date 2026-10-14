"""Entry point da CLI."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

from . import __version__
from .assets import SelectedImage, download_confirmed_images
from .audit import AuditLog
from .discovery import (
    ImageCandidate, VideoCandidate, download_video, is_likely_review,
    metadata_for_video, search_image_candidates, search_video_candidates,
)
from .errors import (
    InvalidProductLink, MediaProcessingError, ProductNotFound,
    ProviderError, ShopeeShortError, SourceBlocked, UserDeclined,
)
from .llm import DEFAULT_BASE_URL, DEFAULT_FALLBACK_MODEL, DEFAULT_MODEL, generate_script
from .media import inspect_source_audio, media_duration, probe_media, render_video, verify_final_video
from .network import is_shopee_cdn_host, normalized_host
from .product import Product, parse_product_link
from .provenance import (enumerate_official_gallery, model_codes, primary_model_code,
                                      verify_image_sources, verify_video_source_page)
from .tts import synthesize_ptbr


def _safe_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return (stem[:70] or "product")


def _confirm(prompt: str, *, flag: bool = False, noninteractive_hint: str) -> bool:
    if flag:
        return True
    if not sys.stdin.isatty():
        raise UserDeclined(f"Confirmação necessária: {prompt} Reexecute com {noninteractive_hint}.")
    answer = input(prompt + " [s/N] ").strip().lower()
    return answer in {"s", "sim", "y", "yes"}


def _duration_text(seconds: int | None) -> str:
    if seconds is None:
        return "duração desconhecida"
    return f"{seconds}s"


def _choose_video(product: Product, args: argparse.Namespace, audit: AuditLog,
                  output_dir: Path) -> tuple[Path | None, dict | None]:
    if args.slideshow_only:
        return None, {"mode": "slideshow", "video": None,
                      "reason": "slideshow-only solicitado; a galeria Shopee não foi consultada",
                      "shopee_gallery_checked": False}

    expected_model = primary_model_code(product.name)

    if args.source_file:
        path = Path(args.source_file).expanduser().resolve()
        if not path.is_file():
            raise ProductNotFound(f"Arquivo de vídeo de origem não encontrado: {path}")
        candidate = {
            "url": args.source_url or "local file",
            "title": args.source_title or path.name,
            "channel": args.source_channel or "fonte local confirmada pelo usuário",
            "platform": "arquivo local" if not args.source_url else "alternativa externa (arquivo fornecido)",
        }
        title_codes = model_codes(candidate["title"])
        if expected_model not in title_codes:
            raise ProductNotFound(
                f"Correspondência recusada: o título/rótulo local não comprova o modelo exato "
                f"{expected_model!r}. Nenhum vídeo será renderizado."
            )
        identity_evidence = {
            "expected_model_code": expected_model,
            "source_title_model_codes": title_codes,
            "identity_match": True,
            "verification_method": "código de modelo exato no rótulo/título local; usuário confirma a demo",
        }
        if not _confirm(
            f"Confirma que {candidate['title']!r} mostra o MESMO produto e é demo, não review/UGC?",
            flag=args.confirm_source,
            noninteractive_hint="--confirm-source",
        ):
            raise UserDeclined("Fonte de vídeo recusada pelo usuário.")
        return path, {**candidate, "user_confirmed_product_and_demo": True,
                      "source_type": args.source_type, "product_identity_evidence": identity_evidence}

    if args.video_url:
        candidate = metadata_for_video(args.video_url, audit=audit)
        if is_likely_review(candidate):
            raise ProductNotFound(
                f"Fonte recusada por parecer review/UGC: {candidate.title!r}. Escolha uma demo de produto, não review."
            )
        title_codes = model_codes(candidate.title)
        if expected_model not in title_codes:
            reason = (
                f"Correspondência recusada: título da fonte {candidate.title!r} não contém o modelo exato "
                f"{expected_model!r} (modelos encontrados: {title_codes or 'nenhum'}). "
                "Escolha uma fonte do mesmo modelo; nenhum vídeo foi baixado."
            )
            audit.note("video_source_rejected_model_mismatch", url=candidate.url, title=candidate.title,
                       expected_model_code=expected_model, found_model_codes=title_codes, reason=reason)
            raise ProductNotFound(reason)
        identity_evidence = {
            "expected_model_code": expected_model,
            "source_title_model_codes": title_codes,
            "identity_match": True,
            "verification_method": "código de modelo exato no título público da página YouTube",
        }
        if args.source_page_url:
            identity_evidence["product_page"] = verify_video_source_page(
                product.name, args.source_page_url, candidate.url, audit=audit,
            )
        print(f"Candidato de vídeo: {candidate.title}\nCanal: {candidate.channel}\nURL: {candidate.url}\n")
        if not _confirm(
            "Confirma que este vídeo mostra o produto/modelo exato e é demo (não review/UGC)?",
            flag=args.confirm_source,
            noninteractive_hint="--confirm-source",
        ):
            raise UserDeclined("Fonte de vídeo recusada; a CLI não vai baixá-la.")
        local = download_video(candidate, output_dir / "source", audit=audit)
        return local, {
            "url": candidate.url, "title": candidate.title, "channel": candidate.channel,
            "duration_seconds": candidate.duration, "platform": candidate.platform,
            "user_confirmed_product_and_demo": True, "source_type": args.source_type,
            "downloaded_file": str(local), "product_identity_evidence": identity_evidence,
        }

    print("Buscando vídeos alternativos no YouTube (sem acessar Shopee)...")
    candidates = search_video_candidates(product.name, audit=audit)
    matching = [candidate for candidate in candidates if expected_model in model_codes(candidate.title)]
    for candidate in candidates:
        if candidate not in matching:
            audit.note("video_candidate_rejected_model_mismatch", title=candidate.title,
                       expected_model_code=expected_model, found_model_codes=model_codes(candidate.title))
    if not matching:
        print("Nenhuma demo com o modelo exato foi encontrada; vou tentar o slideshow de fotos.")
        return None, {"mode": "slideshow", "video": None,
                      "reason": "nenhum vídeo elegível com código de modelo exato; galeria Shopee não consultada",
                      "shopee_gallery_checked": False}
    print("Candidatos de vídeo que contêm o modelo exato e não parecem review/UGC:")
    for i, c in enumerate(matching, start=1):
        print(f"  {i}. {c.title} — {c.channel} — {_duration_text(c.duration)}\n     {c.url}")
    if not sys.stdin.isatty():
        raise UserDeclined(
            "A busca encontrou candidatos do modelo exato; confirme um em terminal "
            "ou passe --video-url URL --confirm-source."
        )
    selection = input("Escolha o número do produto certo (0 = nenhum, usar fotos): ").strip()
    if not selection.isdigit():
        raise UserDeclined("Seleção inválida; nenhuma fonte foi baixada.")
    index = int(selection)
    if index == 0:
        return None, {"mode": "slideshow", "video": None,
                      "reason": "operador não selecionou vídeo; fallback para fotos",
                      "shopee_gallery_checked": False}
    if not 1 <= index <= len(matching):
        raise UserDeclined("Número fora da lista; nenhuma fonte foi baixada.")
    candidate = matching[index - 1]
    if not _confirm(
        f"Confirma que {candidate.title!r} é o MESMO modelo e uma demo de produto (não review/UGC)?",
        noninteractive_hint="--video-url URL --confirm-source",
    ):
        return None, {"mode": "slideshow", "video": None,
                      "reason": "candidato não confirmado pelo operador; fallback para fotos",
                      "shopee_gallery_checked": False}
    local = download_video(candidate, output_dir / "source", audit=audit)
    return local, {
        "url": candidate.url, "title": candidate.title, "channel": candidate.channel,
        "duration_seconds": candidate.duration, "platform": candidate.platform,
        "user_confirmed_product_and_demo": True, "source_type": args.source_type,
        "downloaded_file": str(local),
        "product_identity_evidence": {
            "expected_model_code": expected_model,
            "source_title_model_codes": model_codes(candidate.title),
            "identity_match": True,
            "verification_method": "código de modelo exato no título público da página YouTube",
        },
    }


def _select_images(product: Product, args: argparse.Namespace, audit: AuditLog,
                   output_dir: Path) -> tuple[list[Path], list[dict]]:
    if args.official_gallery_url:
        # Rework: fonte única oficial — a galeria inteira da página do fabricante é
        # enumerada e provada (modelo exato + alt por foto), com o MESMO fail-closed.
        gallery = args.official_gallery_url
        host = normalized_host(gallery)
        if is_shopee_cdn_host(host):
            raise SourceBlocked(f"Página oficial bloqueada por política; não vou acessar {host}.")
        from .script_lines import script_lines as _script_lines_fn
        # Rework: UMA cena por linha do roteiro => exatamente N fotos oficiais.
        # Se a galeria tiver menos fotos que linhas, o fail-closed de sincronia abaixo
        # recusa renderizar com dessincronia (nada é duplicado nem inventado).
        wanted_scenes = len(_script_lines_fn(Path(args.script_file).read_text(encoding="utf-8"))) \
            if args.script_file else 8
        candidates, gallery_proof = enumerate_official_gallery(
            gallery, product.name, audit=audit, max_photos=wanted_scenes,
            page_cache_file=Path(args.page_cache_file).expanduser() if args.page_cache_file else None,
        )
        photo_evidence = gallery_proof["photos"]
        print(f"Galeria oficial verificada: {gallery_proof['page'].title!r} | "
              f"modelo {gallery_proof['identity']['expected_model_code']} | "
              f"{len(candidates)} fotos oficiais")
        for proof in photo_evidence:
            print(f"Fonte de foto verificada: {proof['page_title']} | modelo {proof['expected_model_code']}")
        if not _confirm(
            f"Confirma visualmente que as {len(candidates)} fotos da galeria oficial mostram "
            f"{product.name!r}?",
            flag=args.confirm_photos,
            noninteractive_hint="--confirm-photos",
        ):
            raise UserDeclined("Fotos recusadas pelo usuário.")
        selected = download_confirmed_images(candidates, output_dir / "source_photos", audit=audit)
        return [x.path for x in selected], [
            {"title": x.title, "image_url": x.source_url, "page_url": x.page_url,
             "aliexpress": x.aliexpress, "user_confirmed": True,
             "official_gallery": gallery,
             "product_match_evidence": proof}
            for x, proof in zip(selected, photo_evidence)
        ]

    if args.image_file:
        reason = (
            "Fotos locais sem URL de origem e página que prove o modelo não são aceitas; falhei sem renderizar. "
            "Use --image-url e --image-page-url, depois de confirmar visualmente as fotos."
        )
        audit.note("photo_source_rejected", reason=reason, local_file_count=len(args.image_file),
                   evidence_required="source page exact model plus gallery-bound image URL")
        raise ProductNotFound(reason)

    photo_evidence: list[dict]
    if args.image_url:
        if len(args.image_page_url) != len(args.image_url):
            raise UserDeclined("É obrigatório informar exatamente um --image-page-url verificável para cada --image-url.")
        candidates = []
        for index, url in enumerate(args.image_url):
            host = normalized_host(url)
            if is_shopee_cdn_host(host):
                raise SourceBlocked(f"Foto bloqueada por política; não vou acessar {host}.")
            page_url = args.image_page_url[index]
            page_host = normalized_host(page_url)
            if is_shopee_cdn_host(page_host):
                raise SourceBlocked(f"Página-fonte bloqueada por política; não vou acessar {page_host}.")
            candidates.append(ImageCandidate(url, page_url, Path(urlsplit(url).path).name or "foto fornecida",
                                            page_host or host,
                                            "aliexpress." in page_host or "alicdn.com" in page_host or "alicdn.com" in host))
        # Fonte e associação da imagem são verificadas antes da confirmação e antes de qualquer download.
        photo_evidence = verify_image_sources(candidates, product.name, audit=audit)
        for proof in photo_evidence:
            print(f"Fonte de foto verificada: {proof['page_title']} | modelo {proof['expected_model_code']}")
        if any(c.aliexpress for c in candidates) and not args.confirm_aliexpress and not sys.stdin.isatty():
            raise UserDeclined("Foto de origem AliExpress requer --confirm-aliexpress antes do download.")
        if not _confirm(
            f"Confirma visualmente que as {len(candidates)} fotos verificadas mostram {product.name!r}?",
            flag=args.confirm_photos,
            noninteractive_hint="--confirm-photos",
        ):
            raise UserDeclined("Fotos recusadas pelo usuário.")
        if any(c.aliexpress for c in candidates) and not args.confirm_aliexpress:
            if not sys.stdin.isatty() or not _confirm(
                "Confirma explicitamente a origem AliExpress correta antes do download?",
                noninteractive_hint="--confirm-aliexpress",
            ):
                raise UserDeclined("Origem AliExpress não confirmada; nada foi baixado.")
    else:
        print("Buscando fotos alternativas do produto (CDNs/domínios Shopee são filtrados)...")
        candidates = search_image_candidates(product.name, audit=audit)
        if not candidates:
            raise ProductNotFound(
                "Nenhuma foto pública correspondente foi encontrada fora da Shopee. "
                "Não vou inventar imagens; forneça --image-url e --image-page-url após confirmar a origem."
            )
        print("Candidatos de imagem (a CLI verificará modelo/página antes do download):")
        for i, c in enumerate(candidates, start=1):
            source_tag = " [ALIEXPRESS — requer confirmação explícita]" if c.aliexpress else ""
            print(f"  {i}. {c.title}{source_tag}\n     página: {c.page_url}\n     imagem: {c.image_url}")
        if args.select_images:
            raw = args.select_images
            try:
                indexes = [int(x.strip()) for x in raw.split(",") if x.strip()]
            except ValueError as exc:
                raise UserDeclined("--select-images deve ser uma lista como 1,2,4.") from exc
        elif sys.stdin.isatty():
            raw = input("Números das fotos corretas separados por vírgula (0 = cancelar): ").strip()
            if raw == "0":
                raise UserDeclined("Seleção de fotos cancelada.")
            try:
                indexes = [int(x.strip()) for x in raw.split(",") if x.strip()]
            except ValueError as exc:
                raise UserDeclined("Seleção de fotos inválida.") from exc
        else:
            raise UserDeclined(
                "Confirmação de fotos necessária: use terminal interativo ou --select-images 1,2 --confirm-photos."
            )
        if not indexes or any(i < 1 or i > len(candidates) for i in indexes):
            raise UserDeclined("Seleção de fotos inválida ou vazia.")
        chosen = [candidates[i - 1] for i in indexes]
        photo_evidence = verify_image_sources(chosen, product.name, audit=audit)
        if any(c.aliexpress for c in chosen) and not args.confirm_aliexpress and not sys.stdin.isatty():
            raise UserDeclined(
                "Você selecionou um candidato AliExpress; confirme explicitamente com --confirm-aliexpress "
                "(ou responda 's' no terminal) antes do download."
            )
        print("Fotos selecionadas e verificadas:")
        for c, proof in zip(chosen, photo_evidence):
            tag = " [AliExpress]" if c.aliexpress else ""
            print(f"  - {c.title}{tag} — {proof['page_url']} | modelo {proof['expected_model_code']}")
        if not _confirm(
            f"Confirma visualmente que as fotos selecionadas mostram o mesmo produto/modelo {product.name!r}?",
            flag=args.confirm_photos,
            noninteractive_hint="--confirm-photos",
        ):
            raise UserDeclined("Fotos recusadas pelo usuário.")
        if any(c.aliexpress for c in chosen) and not args.confirm_aliexpress:
            if not sys.stdin.isatty() or not _confirm(
                "Confirma explicitamente o candidato AliExpress correto, antes do download?",
                noninteractive_hint="--confirm-aliexpress",
            ):
                raise UserDeclined("Candidato AliExpress não confirmado; nada foi baixado.")
        candidates = chosen

    selected = download_confirmed_images(candidates, output_dir / "source_photos", audit=audit)
    return [x.path for x in selected], [
        {"title": x.title, "image_url": x.source_url, "page_url": x.page_url,
         "aliexpress": x.aliexpress, "user_confirmed": True,
         "product_match_evidence": proof}
        for x, proof in zip(selected, photo_evidence)
    ]


def _load_script(args: argparse.Namespace, product: Product, audit: AuditLog) -> tuple[str, dict]:
    if args.script_file:
        path = Path(args.script_file).expanduser().resolve()
        if not path.is_file():
            raise ProviderError(f"Arquivo de roteiro não encontrado: {path}")
        # Rework: a ESTRUTURA DE LINHAS do arquivo é preservada — cada linha é uma cena.
        raw_lines = [re.sub(r"\s+", " ", ln).strip()
                     for ln in path.read_text(encoding="utf-8").splitlines()]
        text = "\n".join(ln for ln in raw_lines if ln)
        if not text.strip():
            raise ProviderError("Arquivo de roteiro vazio.")
        audit.note("script_provider", provider="script-file", path=str(path), llm_bypassed=True)
        return text, {"provider": "script-file", "path": str(path), "llm_bypassed": True}
    text, metadata = generate_script(product.name, base_url=args.llm_base_url,
                                     model=args.model, fallback_model=args.fallback_model, audit=audit)
    return text, metadata


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_inventory(output_dir: Path, exclude: set[Path] | None = None) -> list[dict]:
    exclude = exclude or set()
    inventory = []
    for path in sorted(p for p in output_dir.rglob("*") if p.is_file()):
        if path in exclude or path.name in {"report.json", "audit.jsonl"}:
            continue
        inventory.append({"path": str(path.relative_to(output_dir)), "size_bytes": path.stat().st_size,
                          "sha256": _sha256(path)})
    return inventory


def _clean_previous_run(output_dir: Path, source_file: str | None = None) -> None:
    """Remove only artifacts owned by this CLI, preventing stale evidence in reruns."""
    protected_source = Path(source_file).expanduser().resolve() if source_file else None
    if protected_source and protected_source.is_relative_to(output_dir):
        raise InvalidProductLink("Mantenha --source-file fora de --output-dir, que é limpo de artefatos da execução anterior.")
    for name in ("final.mp4", "narration.wav", "captions.ass", "captions.srt", "script.txt",
                 "report.json", "audit.jsonl", "llm_failure.json", "frames", "source", "source_photos", ".work"):
        path = output_dir / name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shopee-short",
        description="Link Shopee -> demo externo confirmado ou slideshow -> locução pt-BR -> legenda ASS queimada -> MP4 9:16.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("product_url", help="URL individual de produto shopee.*; o slug é lido localmente")
    parser.add_argument("-o", "--output-dir", default=None, help="Pasta da saída")
    parser.add_argument("--video-url", help="URL de vídeo de fonte alternativa; metadados são mostrados antes do download")
    parser.add_argument("--source-page-url", help="Página oficial do produto que comprova modelo exato e inclui o vídeo")
    parser.add_argument("--source-file", help="Arquivo de vídeo local já obtido pelo usuário")
    parser.add_argument("--source-title", help="Rótulo da fonte quando --source-file é usado")
    parser.add_argument("--source-channel", help="Canal/vendedor da fonte quando --source-file é usado")
    parser.add_argument("--source-type", choices=["alternative", "seller-gallery"], default="alternative",
                        help="Fonte alternativa aceita fala porque será mutada; gallery seller bloqueia fala detectada")
    parser.add_argument("--confirm-source", action="store_true",
                        help="Confirma explicitamente modelo exato e vídeo demo (necessário sem TTY)")
    parser.add_argument("--slideshow-only", action="store_true", help="Pula a busca/download de vídeo e usa fotos")
    parser.add_argument("--image-file", action="append", default=[],
                        help="Foto local sem proveniência é recusada; use --image-url com --image-page-url")
    parser.add_argument("--image-url", action="append", default=[], help="URL HTTPS de foto alternativa confirmada; repita")
    parser.add_argument("--image-page-url", action="append", default=[], help="Página fonte para cada --image-url, na mesma ordem")
    parser.add_argument("--select-images", help="Índices de candidatos DDG, ex.: 1,2,4")
    parser.add_argument("--official-gallery-url",
                       help="Rework: página oficial do fabricante cuja GALERIA inteira é verificada "
                            "(modelo exato + alt por foto) e usada como cenas do slideshow (default: None)")
    parser.add_argument("--page-cache-file",
                        help="Rework (throttling): HTML da página oficial baixado AO VIVO nesta sessão "
                             "(HTTP 200 registrado no audit anterior); reutilizado para a verificação "
                             "fail-closed de proveniência sem novo request ao host da página, quando o "
                             "site está devolvendo 429. A validação estrutural é idêntica; nada é relaxado.")
    parser.add_argument("--confirm-photos", action="store_true", help="Confirma que as fotos são do mesmo produto")
    parser.add_argument("--confirm-aliexpress", action="store_true",
                        help="Confirma explicitamente candidato AliExpress antes de download")
    parser.add_argument("--script-file", help="Roteiro de teste/manual; sem este arquivo Pollinations v1 é usado")
    parser.add_argument("--llm-base-url", default=DEFAULT_BASE_URL, help="Base OpenAI-compatible, sufixo /v1")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Modelo LLM primário Pollinations v1")
    parser.add_argument("--fallback-model", default=DEFAULT_FALLBACK_MODEL,
                        help="Modelo fallback Pollinations v1; mesma rota e mesmo método POST")
    parser.add_argument("--asr-model", default="base", help="Whisper local usado para verificar fala da origem")
    parser.add_argument("--min-duration", type=float,
                        default=float(os.environ.get("SHOPEE_SHORT_MIN_DURATION", "25")),
                        help="Duração mínima do vídeo em segundos (env SHOPEE_SHORT_MIN_DURATION)")
    parser.add_argument("--max-duration", type=float,
                        default=float(os.environ.get("SHOPEE_SHORT_MAX_DURATION", "34")),
                        help="Teto de duração em segundos (default 34: alvo 25-30 + folga para pausas/"
                             "padding do WAV; env SHOPEE_SHORT_MAX_DURATION). Acima disso, com "
                             "--no-mild-speedup, o excedente natural é aceito até a tolerância (~35s) e "
                             "registrado no relatório.")
    parser.add_argument("--no-mild-speedup", action="store_true",
                        help="Proíbe aceleração leve do TTS; se necessário o vídeo passa do teto (>=25s natural)")
    parser.add_argument("--pronunciation-file", default=os.environ.get("SHOPEE_SHORT_PRONUNCIATION"),
                        help="JSON {grafia: respelling} aplicado só ao texto do TTS (legenda intacta); "
                             "complementa/sobrescreve o dicionário padrão")
    parser.add_argument("--no-subtitle-ocr", action="store_true",
                        help="Pula OCR tesseract dos frames; verificação via .srt + inspeção visual")
    parser.add_argument("--version", action="version", version=f"shopee-short {__version__}")
    return parser


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    if args.source_file and args.video_url:
        raise InvalidProductLink("Use --source-file ou --video-url, não ambos.")
    if args.source_page_url and not args.video_url:
        raise InvalidProductLink("--source-page-url exige --video-url para verificar o vínculo do ID do vídeo.")
    if args.slideshow_only and (args.source_file or args.video_url):
        raise InvalidProductLink("--slideshow-only não pode ser combinado com uma fonte de vídeo.")
    if args.official_gallery_url and (args.image_url or args.image_file or args.select_images):
        raise InvalidProductLink(
            "--official-gallery-url é a fonte única oficial; não combine com --image-url/--image-file/--select-images."
        )
    # O primeiro passo também audita shp.ee quando houver redirect curto; o destino Shopee nunca é seguido.
    audit = AuditLog()
    product = parse_product_link(args.product_url, audit=audit)
    if output_dir is None:
        output_dir = (Path.cwd() / "output" / _safe_stem(product.slug)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _clean_previous_run(output_dir, args.source_file)
    # Um log antigo não pode contaminar a prova desta execução; eventos do resolver são preservados.
    audit.bind(output_dir / "audit.jsonl")
    audit.note("run_started", version=__version__, input_host=normalized_host(args.product_url),
               input_url=product.input_url, resolved_url=product.resolved_url,
               slug=product.slug, product_name_from_slug=product.name,
               policy_no_shopee_fetch=True)
    audit.note("network_policy", blocked=["shopee.*", "*.susercontent.com"],
               allowed_exception="shp.ee apenas para resolver redirect, sem seguir destino Shopee")

    print(f"Produto (slug local): {product.name}")
    print("Política: nenhuma requisição à página/CDN Shopee; só o slug do link é lido.")
    try:
        script, script_metadata = _load_script(args, product, audit)
    except ProviderError as exc:
        attempts = [event for event in audit.events if event.get("event") == "pollinations_model_attempt"]
        if attempts:
            failure_path = output_dir / "llm_failure.json"
            failure_doc = {
                "project": "shopee-short",
                "status": "failed_before_render",
                "stage": "Pollinations script generation",
                "input_url": product.input_url,
                "product_name_from_slug": product.name,
                "endpoint": args.llm_base_url.rstrip("/") + "/chat/completions",
                "method": "POST",
                "primary_model": args.model,
                "fallback_model": args.fallback_model,
                "authorization_header_sent": bool(os.environ.get("POLLINATIONS_API_KEY", "").strip()),
                "attempts": attempts,
                "network_audit": audit.summary(),
                "error": str(exc),
                "render_created": False,
            }
            failure_path.write_text(json.dumps(failure_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            audit.note("run_failed", stage="Pollinations script generation", failure_report=str(failure_path))
            print(f"Relatório de falha LLM: {failure_path}", file=sys.stderr)
        raise
    (output_dir / "script.txt").write_text(script + "\n", encoding="utf-8")

    source_video, source_meta = _choose_video(product, args, audit, output_dir)
    image_meta: list[dict] = []
    images: list[Path] | None = None
    if source_video is None:
        images, image_meta = _select_images(product, args, audit, output_dir)
        if not images:
            raise ProductNotFound("Não foi possível montar vídeo nem slideshow; nenhum conteúdo será inventado.")

    work_parent = output_dir / ".work"
    work_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="run_", dir=work_parent) as tmp:
        work = Path(tmp)
        source_audio_report = None
        if source_video is not None:
            source_probe = probe_media(source_video)
            source_audio_report = inspect_source_audio(
                source_video, work, model_name=args.asr_model, source_type=args.source_type,
            )
            source_meta = source_meta or {}
            source_meta["ffprobe_source"] = {
                "duration_seconds": round(media_duration(source_probe), 3),
                "video_streams": sum(s.get("codec_type") == "video" for s in source_probe.get("streams", [])),
                "audio_streams": source_audio_report.get("audio_streams", 0),
                "video_codec": next((s.get("codec_name") for s in source_probe.get("streams", []) if s.get("codec_type") == "video"), None),
                "audio_codec": source_audio_report.get("codec"),
            }
            source_meta["audio_verification"] = source_audio_report
        else:
            source_audio_report = {"audio_streams": 0, "inspection": "slideshow de fotos: sem arquivo de vídeo de origem"}

        voice_file = output_dir / "narration.wav"
        ass_file = work / "captions.ass"
        srt_file = output_dir / "captions.srt"
        tts_report = synthesize_ptbr(script, output_wav=voice_file, output_ass=ass_file,
                                     output_srt=srt_file,
                                     cache_dir=Path.home() / ".cache" / "shopee-short" / "voices",
                                     audit=audit,
                                     min_duration_seconds=args.min_duration,
                                     max_duration_seconds=args.max_duration,
                                     allow_mild_speedup=not args.no_mild_speedup,
                                     pronunciation_file=args.pronunciation_file)
        shutil.copy2(ass_file, output_dir / "captions.ass")
        final_path = output_dir / "final.mp4"
        scene_spans = tts_report.get("scenes") or None
        if scene_spans and images and len(scene_spans) != len(images):
            # Fail-closed de sincronia: uma cena por foto oficial, sem exceções silenciosas.
            raise MediaProcessingError(
                f"Sincronia cena↔foto falhou: {len(scene_spans)} cenas de narração vs "
                f"{len(images)} fotos oficiais. Nada será renderizado com dessincronia."
            )
        render_report = render_video(source_video, images, voice_file, ass_file, work, final_path,
                                     scene_spans=scene_spans)
        final_probe = verify_final_video(final_path, voice_file, output_dir / "frames",
                                         tts_report["cues"], ocr=not args.no_subtitle_ocr)

    # Confirmação de integridade da política, com contagem derivada do log auditável.
    network_summary = audit.summary()
    if network_summary["shopee_domain_request_count"] != 0:
        raise SourceBlocked("Auditoria detectou request a domínio Shopee/CDN; execução invalidada.")
    inventory = _artifact_inventory(output_dir)
    report = {
        "project": "shopee-short",
        "version": __version__,
        "input": {
            "url": product.input_url,
            "resolved_url": product.resolved_url,
            "slug": product.slug,
            "product_name_from_slug": product.name,
            "shop_id": product.shop_id,
            "item_id": product.item_id,
            "slug_extraction": "local; sem fetch do produto",
        },
        "source": source_meta or {"mode": "slideshow", "video": None},
        "photo_sources": image_meta,
        "script": {"text": script, **script_metadata},
        "tts": tts_report,
        "render": render_report,
        "final_probe": final_probe,
        "network_audit": network_summary,
        "policy": {
            "shopee_or_cdn_requests": 0,
            "source_audio_in_final": False,
            "final_audio_mapping": "somente stream da narração Piper",
            "browser_automation": False,
            "credentials_or_login": False,
            "captcha_bypass": False,
            "proxy_or_farm": False,
        },
        "artifacts": inventory,
    }
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit.note("run_completed", output=str(final_path), duration_seconds=final_probe["duration_seconds"],
               width=final_probe["video"]["width"], height=final_probe["video"]["height"],
               audio_streams=final_probe["audio"]["stream_count"], shopee_domain_request_count=0)
    print(f"\nOK: {final_path}")
    print(f"Duração: {final_probe['duration_seconds']} s | {final_probe['video']['width']}x{final_probe['video']['height']} "
          f"| H.264/AAC | áudio streams: {final_probe['audio']['stream_count']}")
    print(f"Legenda OCR em 10/50/90%: {', '.join(str(x['percentage'])+'%' for x in final_probe['frames_ocr'])}")
    print(f"Relatório: {output_dir / 'report.json'}")
    print(f"Auditoria: {output_dir / 'audit.jsonl'} (requests shopee.*: 0)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except ShopeeShortError as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return exc.exit_code
    except KeyboardInterrupt:
        print("ERRO: execução interrompida pelo usuário; nenhum CAPTCHA/login/proxy será tentado.", file=sys.stderr)
        return 130
    except Exception as exc:
        # Stack traces não vazam por padrão; detalhes inesperados permanecem limpos e acionáveis.
        print(f"ERRO inesperado: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
