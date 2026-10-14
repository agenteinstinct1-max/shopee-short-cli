"""Busca de fontes alternativas; nenhum domínio Shopee é consultado."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from .audit import AuditLog
from .errors import ProductNotFound, ProviderError, SourceBlocked
from .network import assert_safe_request_url, is_shopee_cdn_host

_URL_HOST_RE = re.compile(r"https?://([A-Za-z0-9.-]+)", re.IGNORECASE)
_BLOCK_MARKERS = (
    "captcha", "verify you are human", "not a robot", "confirm you're not a bot",
    "http error 403", "http error 429", "403 forbidden", "429 too many requests", "too many requests", "access denied",
    "sign in to confirm", "bot detection", "challenge required",
)
_REVIEW_MARKERS = (
    "review", "unboxing", "first look", "honest test", "reaction", "comparison",
    "worth buying", "should you buy", "vale a pena", "análise", "analise", "review e",
)
_PROXY_ENV_NAMES = {"http_proxy", "https_proxy", "all_proxy", "ftp_proxy", "ddgs_proxy"}


def _environment_without_proxies() -> dict[str, str]:
    """O processo filho nunca herda configuração de proxy/farm do ambiente."""
    return {k: v for k, v in os.environ.items() if k.lower() not in _PROXY_ENV_NAMES}


@dataclass(frozen=True)
class VideoCandidate:
    url: str
    title: str
    channel: str
    duration: int | None
    description: str
    platform: str = "YouTube"

    @property
    def source_kind(self) -> str:
        return "demo externo (YouTube)"


@dataclass(frozen=True)
class ImageCandidate:
    image_url: str
    page_url: str
    title: str
    source: str
    aliexpress: bool = False


def _run_yt_dlp(args: list[str], *, operation: str, audit: AuditLog | None,
                timeout: int = 120) -> tuple[str, str]:
    cmd = [sys.executable, "-m", "yt_dlp", "--ignore-config", "--no-cookies", "--no-cache-dir", "--verbose", *args]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False,
                              env=_environment_without_proxies())
    except FileNotFoundError as exc:
        raise ProviderError("yt-dlp não está instalado; instale as dependências do projeto.") from exc
    except subprocess.TimeoutExpired as exc:
        raise SourceBlocked("yt-dlp excedeu o tempo limite; encerrei sem retry ou automação de browser.") from exc

    # O stderr contém o log de rede; não salvamos URLs assinadas com query string.
    stderr = proc.stderr or ""
    hosts = {h.lower().rstrip(".") for h in _URL_HOST_RE.findall(stderr)}
    if audit:
        audit.tool_hosts(operation, sorted(hosts), note="hosts extraídos do stderr do yt-dlp; URLs/token omitidos")
    combined_lower = (stderr + "\n" + (proc.stdout or "")).lower()
    if proc.returncode != 0 and any(marker in combined_lower for marker in _BLOCK_MARKERS):
        raise SourceBlocked("YouTube/yt-dlp retornou desafio, HTTP 403/429 ou bloqueio; encerrando sem retry, cookie, proxy ou CAPTCHA.")
    if proc.returncode != 0:
        short_error = " ".join(stderr.strip().split())[-700:]
        raise ProductNotFound(f"Fonte de vídeo indisponível ({short_error or 'yt-dlp sem detalhes'}).")
    return proc.stdout or "", stderr


def _video_url_from_json(data: dict) -> str:
    return data.get("webpage_url") or data.get("original_url") or (
        f"https://www.youtube.com/watch?v={data['id']}" if data.get("id") else ""
    )


def metadata_for_video(url: str, *, audit: AuditLog | None = None) -> VideoCandidate:
    assert_safe_request_url(url, operation="video_metadata")
    stdout, _ = _run_yt_dlp(["--no-playlist", "--dump-single-json", url],
                            operation="yt_dlp_video_metadata", audit=audit, timeout=120)
    try:
        data = json.loads(stdout)
    except Exception as exc:
        raise ProviderError("yt-dlp não retornou metadados JSON para o vídeo.") from exc
    return VideoCandidate(
        url=_video_url_from_json(data) or url,
        title=str(data.get("title") or "(sem título)"),
        channel=str(data.get("channel") or data.get("uploader") or "(canal não identificado)"),
        duration=int(data["duration"]) if data.get("duration") else None,
        description=str(data.get("description") or "")[:5000],
    )


def search_video_candidates(product_name: str, *, audit: AuditLog | None = None,
                            limit: int = 8) -> list[VideoCandidate]:
    query = f"{product_name} official product demo showcase"
    stdout, _ = _run_yt_dlp(["--flat-playlist", "--dump-single-json", f"ytsearch{limit}:{query}"],
                            operation="yt_dlp_youtube_search", audit=audit, timeout=120)
    try:
        data = json.loads(stdout)
    except Exception as exc:
        raise ProviderError("A busca no YouTube não retornou JSON válido.") from exc
    entries = data.get("entries") or []
    result = []
    for entry in entries:
        title = str(entry.get("title") or "")
        url = _video_url_from_json(entry)
        if not url:
            continue
        # Review/UGC não entra como vídeo de produto. Não baixamos esse tipo de resultado.
        if any(marker in title.lower() for marker in _REVIEW_MARKERS):
            continue
        result.append(VideoCandidate(
            url=url,
            title=title or "(sem título)",
            channel=str(entry.get("channel") or entry.get("uploader") or "(canal não identificado)"),
            duration=int(entry["duration"]) if entry.get("duration") else None,
            description=str(entry.get("description") or "")[:1200],
        ))
    return result


def download_video(candidate: VideoCandidate, dest_dir: Path, *, audit: AuditLog | None = None) -> Path:
    assert_safe_request_url(candidate.url, operation="video_download")
    dest_dir.mkdir(parents=True, exist_ok=True)
    template = str(dest_dir / "source.%(ext)s")
    stdout, stderr = _run_yt_dlp([
        "--no-playlist", "--force-overwrites", "-f", "bv*[height<=1080]+ba/b[height<=1080]/best",
        "--merge-output-format", "mp4", "-o", template, candidate.url,
    ], operation="yt_dlp_video_download", audit=audit, timeout=600)
    if audit:
        audit.note("yt_dlp_download_completed", title=candidate.title, source_url=candidate.url)
    # Os formatos auxiliares HLS/DASH são removidos pelo yt-dlp após a mesclagem.
    found = sorted(p for p in dest_dir.glob("source.*") if p.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"})
    if not found:
        raise ProductNotFound("yt-dlp concluiu sem deixar um arquivo de vídeo reproduzível.")
    return found[0]


def search_image_candidates(product_name: str, *, audit: AuditLog | None = None,
                            limit: int = 12) -> list[ImageCandidate]:
    try:
        from ddgs import DDGS
    except ImportError as exc:
        raise ProviderError("Instale o pacote ddgs para buscar imagens de fontes alternativas.") from exc

    query = f"{product_name} product photos"
    if audit:
        audit.request("duckduckgo_images_search", "duckduckgo.com", method="SEARCH",
                      note="busca pública de imagens; candidatos filtrados antes de download")
    saved_proxy_env = {k: v for k, v in os.environ.items() if k.lower() in _PROXY_ENV_NAMES}
    try:
        for key in list(os.environ):
            if key.lower() in _PROXY_ENV_NAMES:
                os.environ.pop(key, None)
        raw = list(DDGS(proxy=None, timeout=10).images(query, max_results=limit))
    except Exception as exc:
        text = str(exc).lower()
        if any(mark in text for mark in ("captcha", "403", "429", "rate limit", "blocked", "robot")):
            raise SourceBlocked("DuckDuckGo bloqueou a busca de imagens; encerrando sem retry/CAPTCHA/proxy.") from exc
        if "no results" in text:
            return []
        raise ProviderError(f"Falha na busca de imagens públicas: {exc}") from exc
    finally:
        for key, value in saved_proxy_env.items():
            os.environ[key] = value

    candidates: list[ImageCandidate] = []
    for entry in raw:
        image_url = str(entry.get("image") or "")
        page_url = str(entry.get("url") or "")
        title = str(entry.get("title") or "(sem título)")
        if not image_url.startswith("https://") or not page_url.startswith("https://"):
            continue
        image_host = (urlsplit(image_url).hostname or "").lower()
        page_host = (urlsplit(page_url).hostname or "").lower()
        # Shopee e o CDN associado são excluídos mesmo quando retornados pelo buscador.
        if is_shopee_cdn_host(image_host) or is_shopee_cdn_host(page_host):
            continue
        aliexpress = "aliexpress." in page_host or page_host.endswith("alicdn.com") or ".alicdn.com" in image_host
        source = page_host or image_host
        candidates.append(ImageCandidate(image_url, page_url, title, source, aliexpress))
    return candidates


def is_likely_review(candidate: VideoCandidate) -> bool:
    text = (candidate.title + " " + candidate.description).lower()
    return any(marker in text for marker in _REVIEW_MARKERS)
