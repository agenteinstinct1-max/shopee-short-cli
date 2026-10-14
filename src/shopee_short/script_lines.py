"""Preserva a estrutura de LINHAS do roteiro para o mapeamento cena↔frase.

Regra do rework: cada linha não vazia do arquivo de roteiro é UMA unidade de
cena (uma frase de narração por cena). A segmentação por pontuação só acontece
DENTRO de uma linha — as frases nunca são consolidadas entre linhas diferentes.
"""
from __future__ import annotations

import re


def script_lines(raw_text: str) -> list[str]:
    """Retorna as linhas não vazias do roteiro, com espaços colapsados internamente."""
    lines = []
    for line in raw_text.splitlines():
        cleaned = re.sub(r"\s+", " ", line).strip()
        if cleaned:
            lines.append(cleaned)
    return lines


def split_line_sentences(line: str) -> list[str]:
    """Divide uma linha em frases por pontuação final; garante terminator."""
    pieces = [p.strip() for p in re.split(r"(?<=[.!?])\s+", line) if p.strip()]
    return [p if p[-1] in ".!?…" else p + "." for p in pieces]
