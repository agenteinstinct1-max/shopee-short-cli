"""Validação e extração local do slug. Nenhum fetch da página Shopee é feito."""
from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass
from urllib.parse import unquote, urlsplit

from .audit import AuditLog
from .errors import InvalidProductLink
from .network import is_shopee_host, is_shp_ee_host, normalized_host, resolve_shp_ee

_ITEM_SUFFIX = re.compile(r"-i\.\d+\.\d+$", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class Product:
    input_url: str
    resolved_url: str
    slug: str
    name: str
    shop_id: str | None = None
    item_id: str | None = None


def _parse_product_path(url: str) -> Product:
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if parts.scheme.lower() not in {"https", "http"} or not is_shopee_host(host):
        raise InvalidProductLink("O input precisa ser um link de produto em shopee.* (ou um shp.ee resolvível).")
    raw_path = unquote(parts.path).strip("/")
    if not raw_path:
        raise InvalidProductLink("O link Shopee não contém slug de produto no próprio path.")
    segment = raw_path.split("/")[-1]
    if segment.lower() in {"product", "search", "cart", "checkout", "list"}:
        raise InvalidProductLink("O path parece categoria/busca, não um link individual de produto.")
    match = re.search(r"-i\.(\d+)\.(\d+)$", segment, re.IGNORECASE)
    shop_id, item_id = (match.group(1), match.group(2)) if match else (None, None)
    slug = _ITEM_SUFFIX.sub("", segment)
    # O slug é a única fonte do nome: não carrega título de página nem chama Shopee.
    name = html.unescape(slug.replace("-", " ").replace("_", " "))
    name = _WHITESPACE.sub(" ", name).strip(" .")
    if len(name) < 3 or not any(ch.isalpha() for ch in name):
        raise InvalidProductLink("Não foi possível extrair um nome de produto utilizável do slug.")
    return Product(
        input_url=url,
        resolved_url=url,
        slug=slug,
        name=name,
        shop_id=shop_id,
        item_id=item_id,
    )


def parse_product_link(url: str, audit: AuditLog | None = None) -> Product:
    if not isinstance(url, str) or not url.strip():
        raise InvalidProductLink("Informe um link de produto Shopee.")
    url = url.strip()
    host = normalized_host(url)
    if is_shp_ee_host(host):
        resolved = resolve_shp_ee(url, audit)
        product = _parse_product_path(resolved)
        return Product(
            input_url=url,
            resolved_url=resolved,
            slug=product.slug,
            name=product.name,
            shop_id=product.shop_id,
            item_id=product.item_id,
        )
    return _parse_product_path(url)


def normalized_product_query(name: str) -> str:
    """Texto de busca derivado do slug; remove apenas sinais/palavras promocionais comuns."""
    text = unicodedata.normalize("NFKC", name)
    text = re.sub(r"[^\w\s+.-]", " ", text, flags=re.UNICODE)
    text = _WHITESPACE.sub(" ", text).strip()
    return text
