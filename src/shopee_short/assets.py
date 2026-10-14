"""Baixa e valida imagens alternativas confirmadas, bloqueando hosts Shopee/CDN Shopee."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

from .audit import AuditLog
from .discovery import ImageCandidate
from .errors import MediaProcessingError, ProviderError
from .network import safe_get


@dataclass
class SelectedImage:
    path: Path
    source_url: str
    page_url: str
    title: str
    aliexpress: bool = False


def download_confirmed_images(candidates: list[ImageCandidate], destination: Path,
                              *, audit: AuditLog | None = None) -> list[SelectedImage]:
    if not candidates:
        raise MediaProcessingError("Nenhuma imagem candidata foi confirmada.")
    if len(candidates) > 12:
        raise MediaProcessingError("No máximo 12 imagens podem ser baixadas para o slideshow.")
    destination.mkdir(parents=True, exist_ok=True)
    selected: list[SelectedImage] = []
    for index, candidate in enumerate(candidates, start=1):
        try:
            data, response = safe_get(candidate.image_url, operation="download_confirmed_product_photo",
                                      audit=audit, max_bytes=25 * 1024 * 1024,
                                      expected_content_prefix="image/")
            with Image.open(BytesIO(data)) as im:
                im.verify()
            with Image.open(BytesIO(data)) as im:
                path = destination / f"photo_{index:02d}.jpg"
                # Transparência de PNG é composta em branco para evitar halos pretos artificiais.
                rgba = im.convert("RGBA")
                matte = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                matte.alpha_composite(rgba)
                matte.convert("RGB").save(path, format="JPEG", quality=94, optimize=True)
        except Exception as exc:
            if isinstance(exc, (MediaProcessingError, ProviderError)):
                raise
            raise MediaProcessingError(f"Não consegui validar a foto selecionada ({candidate.source}): {exc}") from exc
        selected.append(SelectedImage(path, candidate.image_url, candidate.page_url,
                                      candidate.title, candidate.aliexpress))
    return selected


def copy_local_images(paths: list[Path], destination: Path) -> list[SelectedImage]:
    if not 1 <= len(paths) <= 12:
        raise MediaProcessingError("Informe de 1 a 12 fotos locais.")
    destination.mkdir(parents=True, exist_ok=True)
    selected: list[SelectedImage] = []
    for index, original in enumerate(paths, start=1):
        if not original.is_file():
            raise MediaProcessingError(f"Foto local não encontrada: {original}")
        try:
            with Image.open(original) as im:
                im.verify()
            with Image.open(original) as im:
                saved = destination / f"photo_{index:02d}.jpg"
                rgba = im.convert("RGBA")
                matte = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                matte.alpha_composite(rgba)
                matte.convert("RGB").save(saved, format="JPEG", quality=94, optimize=True)
        except Exception as exc:
            raise MediaProcessingError(f"Foto local inválida: {original}: {exc}") from exc
        selected.append(SelectedImage(saved, str(original.resolve()), "local (usuário)", original.name, False))
    return selected
