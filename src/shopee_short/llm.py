"""Geração de roteiro via endpoint OpenAI-compatible Pollinations v1."""
from __future__ import annotations

import hashlib
import os
import re

import requests

from .audit import AuditLog
from .errors import ProviderError, SourceBlocked
from .network import safe_request

DEFAULT_BASE_URL = "https://gen.pollinations.ai/v1"
DEFAULT_MODEL = "community/YoannDev90/muse-glimmer-30b:free"
DEFAULT_FALLBACK_MODEL = "openai/gpt-5.4-nano"


def generate_script(product_name: str, *, base_url: str = DEFAULT_BASE_URL,
                    model: str = DEFAULT_MODEL, fallback_model: str = DEFAULT_FALLBACK_MODEL,
                    audit: AuditLog | None = None) -> tuple[str, dict]:
    """POST script generation; fall back only to the configured model, never to a new endpoint/key."""
    endpoint = base_url.rstrip("/") + "/chat/completions"
    key = os.environ.get("POLLINATIONS_API_KEY", "").strip()
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    models = list(dict.fromkeys(name.strip() for name in (model, fallback_model) if name and name.strip()))
    if not models:
        raise ProviderError("Nenhum modelo Pollinations foi configurado.")

    session = requests.Session()
    session.trust_env = False
    attempts: list[dict] = []
    last_error = ""
    for model_name in models:
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Você escreve locuções curtas e naturais em português brasileiro para vídeos de produto. "
                        "Use somente fatos que apareçam no nome informado; não invente marcações, números, "
                        "desempenho, garantia, preço, avaliações ou promessas. Evite alegações médicas. "
                        "Retorne somente o texto falado, em 3 frases curtas, cerca de 25 a 40 palavras, "
                        "sem títulos, emojis, aspas ou markdown."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Nome extraído localmente do slug do produto: {product_name}",
                },
            ],
            "temperature": 0.65,
            "max_tokens": 150,
            "stream": False,
        }
        try:
            response = safe_request(
                session, "POST", endpoint, operation="pollinations_chat_completions",
                audit=audit, json=payload, headers=headers,
                timeout=(12, 55),
            )
        except SourceBlocked as exc:
            last_error = str(exc)
            attempt = {"model": model_name, "status": None, "body": None, "transport_error": last_error}
            attempts.append(attempt)
            if audit:
                audit.note("pollinations_model_attempt", endpoint=endpoint, method="POST", **attempt)
            # 403/429, CAPTCHA, and network policy blocks remain fail-fast; never retry around them.
            raise ProviderError(f"Pollinations bloqueou a solicitação; sem retry/CAPTCHA/proxy. {last_error}") from exc
        except requests.RequestException as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            attempt = {"model": model_name, "status": None, "body": None, "transport_error": last_error}
            attempts.append(attempt)
            if audit:
                audit.note("pollinations_model_attempt", endpoint=endpoint, method="POST", **attempt)
            raise ProviderError(f"Falha de rede ao chamar Pollinations v1; sem retry de transporte. {last_error}") from exc

        body_text = response.text
        attempt = {
            "model": model_name,
            "status": response.status_code,
            "body": body_text,
            "body_sha256": hashlib.sha256(response.content).hexdigest(),
        }
        attempts.append(attempt)
        if audit:
            # A resposta textual exata é preservada inclusive em erros 4xx para evidência/reprodução.
            audit.note("pollinations_model_attempt", endpoint=endpoint, method="POST", **attempt)

        if response.status_code >= 400:
            last_error = f"HTTP {response.status_code}: {body_text}"
            continue

        try:
            body = response.json()
            text = body["choices"][0]["message"]["content"]
        except Exception as exc:
            last_error = f"Resposta inválida (HTTP {response.status_code}): {body_text}"
            if audit:
                audit.note("pollinations_model_parse_failed", model=model_name, status=response.status_code,
                           body=body_text, error=f"{type(exc).__name__}: {exc}")
            continue
        if isinstance(text, list):
            text = " ".join(str(x.get("text", "")) for x in text if isinstance(x, dict))
        text = re.sub(r"^```(?:\w+)?\s*|\s*```$", "", str(text).strip(), flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip()
        if not text or len(text) > 1200:
            last_error = f"Roteiro vazio/grande demais (HTTP {response.status_code}): {body_text}"
            if audit:
                audit.note("pollinations_model_parse_failed", model=model_name, status=response.status_code,
                           body=body_text, error="LLM retornou texto vazio ou acima de 1200 caracteres")
            continue
        if audit:
            audit.note("script_generated", provider="Pollinations v1", model=model_name,
                       fallback_used=(model_name != models[0]), api_key_env_present=bool(key))
        return text, {
            "provider": "Pollinations v1",
            "endpoint": endpoint,
            "model": model_name,
            "primary_model": models[0],
            "fallback_model": models[1] if len(models) > 1 else None,
            "fallback_used": model_name != models[0],
            "models_attempted": [entry["model"] for entry in attempts],
            "api_key_env_present": bool(key),
        }

    if audit:
        audit.note("pollinations_generation_failed", endpoint=endpoint, method="POST",
                   models_attempted=[entry["model"] for entry in attempts], attempts=attempts)
    raise ProviderError(
        "Pollinations v1 não gerou roteiro com os modelos configurados. "
        f"Modelos tentados: {', '.join(entry['model'] for entry in attempts)}. "
        "Status e corpo completos das respostas estão em audit.jsonl/llm_failure.json; "
        "não usei GET, endpoint /openai, chave nova nem fallback silencioso. "
        f"Último resultado: {last_error}"
    )
