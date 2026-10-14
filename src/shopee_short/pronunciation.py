"""Pronúncia só para o TTS: a legenda mantém a grafia original.

`prepare_for_tts(text, extra)` devolve o texto que vai ao Piper. Ordem:
1. dicionário (padrão embutido + arquivo JSON opcional do usuário; o do usuário vence);
2. decimais ("5.4"/"1,6" -> "cinco vírgula quatro"), pois o Piper some com o ponto;
3. códigos alfanuméricos (HY300, 720P) -> letras pelo nome + número por extenso;
4. inteiros soltos ficam em dígitos (o Piper os lê bem; por extenso alonga o vídeo).
O dicionário é {"grafia": "respelling"}; chaves casam como palavra inteira, sem
diferenciar maiúsculas. Arquivo extra: --pronunciation-file ou env SHOPEE_SHORT_PRONUNCIATION.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

DEFAULT_DICTIONARY: dict[str, str] = {
    "Magcubic": "Mag quiúbic",
    "Wi-Fi": "uai fai",
    "WiFi": "uai fai",
    "Bluetooth": "blu tuf",
    "keystone": "quistoun",
    "ANSI": "ânsi",
    "Android": "Andróide",
    "Pro": "Pró",
    "Plus": "Plás",
    "HDMI": "agá dê ême i",
    "USB": "u ésse bê",
    "LED": "léd",
    "Full HD": "full agá dê",
}

LETTERS = {
    "a": "á", "b": "bê", "c": "cê", "d": "dê", "e": "é", "f": "éfe", "g": "gê",
    "h": "agá", "i": "í", "j": "jota", "k": "cá", "l": "éle", "m": "ême", "n": "ene",
    "o": "ó", "p": "pê", "q": "quê", "r": "érre", "s": "ésse", "t": "tê", "u": "u",
    "v": "vê", "w": "dáblio", "x": "xis", "y": "ípsilon", "z": "zê",
}


def load_dictionary(path: str | Path | None = None) -> dict[str, str]:
    merged = dict(DEFAULT_DICTIONARY)
    if path:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in data.items()):
            raise ValueError("O arquivo de pronúncia deve ser um objeto JSON {grafia: respelling}.")
        merged.update(data)
    return merged


def _num(n: str) -> str:
    try:
        from num2words import num2words
        return num2words(int(n), lang="pt_BR")
    except Exception:
        return n  # o espeak do Piper lê dígitos sozinho


def _digits(n: str) -> str:
    return " ".join(_num(d) for d in n)


def _spell(letters: str) -> str:
    return " ".join(LETTERS.get(c.lower(), c) for c in letters)


def prepare_for_tts(text: str, dictionary: dict[str, str] | None = None) -> str:
    d = DEFAULT_DICTIONARY if dictionary is None else dictionary
    out = text
    for key in sorted(d, key=len, reverse=True):
        out = re.sub(rf"(?<![\w-]){re.escape(key)}(?![\w-])", lambda _m, v=d[key]: v, out, flags=re.I)
    out = re.sub(r"(\d+)[.,](\d+)", lambda m: f"{_num(m.group(1))} vírgula {_digits(m.group(2))}", out)
    # códigos: 1-3 letras + dígitos (HY300) ou dígitos + 1-2 letras maiúsculas (720P)
    out = re.sub(r"\b([A-Za-z]{1,3})(\d+)\b", lambda m: f"{_spell(m.group(1))} {_num(m.group(2))}", out)
    out = re.sub(r"\b(\d+)([A-Z]{1,2})\b", lambda m: f"{_num(m.group(1))} {_spell(m.group(2))}", out)
    # inteiros soltos ficam como dígitos: o espeak do Piper os lê bem e por extenso alonga o áudio
    return re.sub(r"\s+", " ", out).strip()
