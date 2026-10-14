"""TTS neural local gratuito via Piper (pt-BR), sem chave ou conta."""
from __future__ import annotations

import re
import textwrap
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
from piper import PiperVoice

from .audit import AuditLog
from .errors import MediaProcessingError, ProviderError
from .network import safe_get
from .pronunciation import load_dictionary, prepare_for_tts
from .script_lines import script_lines, split_line_sentences

VOICE_ID = "pt_BR-faber-medium"
VOICE_BASE = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
    "pt/pt_BR/faber/medium/pt_BR-faber-medium.onnx"
)


@dataclass
class Cue:
    start: float
    end: float
    text: str


def _sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        raise MediaProcessingError("O roteiro está vazio; não há narração para sintetizar.")
    pieces = re.split(r"(?<=[.!?])\s+", text)
    result = [p.strip() for p in pieces if p.strip()]
    return [p if p[-1] in ".!?…" else p + "." for p in result]


def _wrap_caption(text: str, width: int = 34) -> str:
    lines = textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False)
    if not lines:
        return text
    return "\\N".join(lines)


def _ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    total_cs = round(seconds * 100)
    cs = total_cs % 100
    total_s = total_cs // 100
    ss = total_s % 60
    total_m = total_s // 60
    mm = total_m % 60
    hh = total_m // 60
    return f"{hh}:{mm:02d}:{ss:02d}.{cs:02d}"


def _srt_time(seconds: float) -> str:
    ms = max(0, round(seconds * 1000))
    hh, rem = divmod(ms, 3_600_000)
    mm, rem = divmod(rem, 60_000)
    ss, milli = divmod(rem, 1000)
    return f"{hh:02d}:{mm:02d}:{ss:02d},{milli:03d}"


def ensure_voice(cache_dir: Path, audit: AuditLog | None = None) -> tuple[Path, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    model = cache_dir / f"{VOICE_ID}.onnx"
    config = cache_dir / f"{VOICE_ID}.onnx.json"
    session = requests.Session()
    session.trust_env = False
    for path, url in ((model, VOICE_BASE), (config, VOICE_BASE + ".json")):
        if path.exists() and path.stat().st_size > 100:
            continue
        try:
            data, _ = safe_get(url, operation="download_piper_voice", audit=audit,
                               max_bytes=100 * 1024 * 1024)
        except Exception as exc:
            raise ProviderError(f"Não foi possível obter o modelo Piper pt-BR: {exc}") from exc
        if len(data) < 100:
            raise ProviderError("Download do modelo Piper veio vazio/truncado.")
        tmp = path.with_suffix(path.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(path)
    return model, config


def synthesize_ptbr(text: str, *, output_wav: Path, output_ass: Path, output_srt: Path,
                    cache_dir: Path, audit: AuditLog | None = None,
                    min_duration_seconds: float | None = None,
                    max_duration_seconds: float | None = None,
                    base_gap_seconds: float = 0.16,
                    allow_mild_speedup: bool = True,
                    pronunciation_file: str | Path | None = None) -> dict:
    pron_dict = load_dictionary(pronunciation_file)
    model_path, config_path = ensure_voice(cache_dir, audit)
    try:
        voice = PiperVoice.load(model_path, config_path)
    except Exception as exc:
        raise ProviderError(f"Falha ao carregar voz Piper {VOICE_ID}: {exc}") from exc

    def _synth(sentence: str, length_scale: float = 1.0):
        # length_scale > 1 alonga a fala (mais devagar); < 1 acelera levemente.
        # O piso é 0.85 (~1.18x de velocidade), nunca além disso (teto ~1.2x).
        from piper.config import SynthesisConfig
        cfg = SynthesisConfig()
        cfg.length_scale = max(0.85, length_scale)
        # Só o texto falado muda; legendas/SRT seguem com a grafia original.
        return list(voice.synthesize(prepare_for_tts(sentence, pron_dict), cfg))

    # ---- Cena = LINHA do roteiro (rework) -------------------------------------
    # Cada linha não vazia do arquivo de roteiro forma UMA cena com sua própria
    # foto oficial. A divisão por pontuação só acontece DENTRO da linha; frases
    # NUNCA são consolidadas entre linhas diferentes. Isso mantém "uma frase de
    # narração por cena" e preserva o ritmo leitura-natural decidido pelo autor.
    lines = script_lines(text)
    if not lines:
        raise MediaProcessingError("O roteiro está vazio; não há narração para sintetizar.")
    scene_sentences: list[list[str]] = [split_line_sentences(line) for line in lines]
    flat: list[str] = [s for group in scene_sentences for s in group]
    scene_groups: list[list[int]] = []
    cursor_i = 0
    for group in scene_sentences:
        scene_groups.append(list(range(cursor_i, cursor_i + len(group))))
        cursor_i += len(group)

    def _measure(sentences: list[str], length_scale: float) -> tuple[list[float], int]:
        durs: list[float] = []
        sr_local = 0
        for sentence in sentences:
            chunks = _synth(sentence, length_scale)
            if not chunks:
                continue
            durs.append(sum(len(c.audio_int16_array) for c in chunks) / chunks[0].sample_rate)
            sr_local = int(chunks[0].sample_rate)
        return durs, sr_local

    target_min = float(min_duration_seconds or 0.0)
    target_max = float(max_duration_seconds) if max_duration_seconds else None
    if target_max is not None and target_max < target_min:
        target_max = target_min
    n_scenes = len(scene_groups)
    gap_seconds = base_gap_seconds
    GAP_FLOOR = 0.16          # pausa mínima natural entre cenas/frases
    gap_floor = GAP_FLOOR

    baseline_durs, _sr0 = _measure(flat, 1.0)
    if not baseline_durs:
        raise MediaProcessingError("Piper não produziu áudio utilizável.")
    speech_total = sum(baseline_durs)
    pacing_mode = "natural-default"
    length_scale = 1.0

    # (a) mais longo que o mínimo natural? alongamos pausas e (se preciso) um leve SLOWDOWN.
    if target_min > 0:
        need = target_min - (speech_total + max(n_scenes - 1, 0) * base_gap_seconds)
        if need > 0:
            per_gap = need / max(n_scenes - 1, 1)
            if per_gap <= 0.9:
                gap_seconds = round(per_gap, 3)
                pacing_mode = "extended-pauses-only"
            else:
                gap_seconds = 0.9
                remaining = need - max(n_scenes - 1, 0) * (0.9 - base_gap_seconds)
                length_scale = min(1.25, max(1.0, 1.0 + remaining / max(speech_total, 0.001)))
                pacing_mode = "extended-pauses-plus-slight-slowdown"

    # (b) mais curto que o máximo? ORÇAMENTO DE VELOCIDADE MILD (permitido até 0.85 ≈ 1.18x).
    # Ordem de preferência do rework: primeiro fechar o orçamento encurtando pausas até o
    # piso natural (0.16 s); se ainda assim passar do teto, aplicamos UM único valor de
    # length_scale (>= 0.85) calibrado por medição real — nunca aceleramos além de ~1.2x,
    # e reportamos tts_speedup_applied honestamente. As cenas NUNCA são fundidas.
    speedup_budget_used = False
    over_max_accepted = False
    MARGIN = 0.35  # folga para que o total medido fique levemente ABAIXO do teto
    if target_max is not None:
        budget = target_max - MARGIN
        projected = speech_total + max(n_scenes - 1, 0) * gap_seconds
        if projected > budget:
            gap_seconds = max(GAP_FLOOR, (budget - speech_total) / max(n_scenes - 1, 1))
            projected = speech_total + max(n_scenes - 1, 0) * gap_seconds
        if projected > budget and not allow_mild_speedup:
            # Preferência 1 do rework: manter todas as linhas em ritmo totalmente natural
            # (length_scale >= 1.0, pausas no piso) e aceitar o excedente acima do teto —
            # desde que dentro da janela de tolerância (~35 s). Cenas nunca são fundidas.
            pacing_mode = "natural-lines-over-max-accepted"
            over_max_accepted = True
        elif projected > budget and allow_mild_speedup:
            # Busca binária: MAIOR length_scale (menor aceleração) em [0.85, 1.0]
            # cujo total projetado caiba no orçamento. Nunca aceleramos além de ~1.18x.
            lo, hi = 0.85, 1.0
            found = None
            for _ in range(7):
                mid = (lo + hi) / 2
                durs_mid, _ = _measure(flat, mid)
                proj_mid = sum(durs_mid) + max(n_scenes - 1, 0) * gap_seconds
                if proj_mid <= budget:
                    found = mid
                    lo = mid
                else:
                    hi = mid
            if found is not None:
                length_scale = found
                speedup_budget_used = True
                pacing_mode = "natural-lines-mild-speedup-to-fit-max"
            else:
                # nem no piso 0.85 cabe no orçamento: usamos o piso (máx ~1.18x permitido)
                length_scale = 0.85
                speedup_budget_used = True
                pacing_mode = "natural-lines-mild-speedup-floor"

    measured: list[tuple[str, np.ndarray]] = []
    sr_global: int | None = None
    for sentence in flat:
        chunks = _synth(sentence, length_scale)
        if not chunks:
            continue
        samples = np.concatenate([chunk.audio_int16_array for chunk in chunks])
        chunk_sr = int(chunks[0].sample_rate)
        if sr_global is None:
            sr_global = chunk_sr
        elif chunk_sr != sr_global:
            raise MediaProcessingError("A voz Piper retornou sample rates inconsistentes.")
        measured.append((sentence, samples))
    if sr_global is None or not measured:
        raise MediaProcessingError("Piper não produziu áudio utilizável.")
    sr = sr_global
    durations = [len(x[1]) / sr for x in measured]
    total_speech = sum(durations)
    groups = [g for g in scene_groups if all(i < len(measured) for i in g)]
    scenes: list[tuple[float, float, list[int]]] = []
    cursor = 0.0
    for g in groups:
        scene_start = cursor
        for k, i in enumerate(g):
            cursor += durations[i]
            if k < len(g) - 1:
                cursor += gap_floor
        scenes.append((scene_start, cursor, g))
        cursor += gap_seconds
    narration_total = cursor - gap_seconds if scenes else 0.0
    lead_pad = tail_pad = 0.0
    if target_min > 0 and narration_total < target_min - 0.05:
        pad = min(1.5, (target_min - narration_total) / 2)
        lead_pad = tail_pad = pad
    cues: list[Cue] = []
    for si, (scene_start, scene_end, g) in enumerate(scenes):
        shift = scene_start
        rendered: list[tuple[float, float, str]] = []
        for k, i in enumerate(g):
            cue_start = shift
            cue_end = cue_start + durations[i]
            rendered.append((cue_start, cue_end, _wrap_caption(measured[i][0])))
            shift = cue_end + (gap_floor if k < len(g) - 1 else gap_seconds)
        boundary_end = scene_end + (gap_seconds if si < len(scenes) - 1 else 0.0)
        for idx, (cs, ce, t) in enumerate(rendered):
            hold = boundary_end if idx == len(rendered) - 1 else ce + gap_floor
            cues.append(Cue(cs + lead_pad, hold + lead_pad, t))
    pcm_parts: list[np.ndarray] = []
    if lead_pad > 0:
        pcm_parts.append(np.zeros(round(sr * lead_pad), dtype=np.int16))
    for si, (_, _, g) in enumerate(scenes):
        for k, i in enumerate(g):
            pcm_parts.append(measured[i][1].astype(np.int16, copy=False))
            if k < len(g) - 1:
                pcm_parts.append(np.zeros(round(sr * gap_floor), dtype=np.int16))
            elif si < len(scenes) - 1:
                pcm_parts.append(np.zeros(round(sr * gap_seconds), dtype=np.int16))
    if tail_pad > 0:
        pcm_parts.append(np.zeros(round(sr * tail_pad), dtype=np.int16))
    audio = np.concatenate(pcm_parts)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_wav), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sr)
        wav.writeframes(audio.tobytes())

    output_srt.parent.mkdir(parents=True, exist_ok=True)
    srt = []
    for i, cue in enumerate(cues, start=1):
        readable_text = cue.text.replace("\\N", "\n")
        srt.extend([str(i), f"{_srt_time(cue.start)} --> {_srt_time(cue.end)}", readable_text, ""])
    output_srt.write_text("\n".join(srt), encoding="utf-8")

    # ASS força texto branco com contorno preto, centralizado na zona segura de TikTok.
    ass_events = []
    for cue in cues:
        safe = cue.text.replace("{", "\\{").replace("}", "\\}").replace("\n", " ")
        ass_events.append(
            f"Dialogue: 0,{_ass_time(cue.start)},{_ass_time(cue.end)},TikTok,,0,0,0,,{safe}"
        )
    ass = "\n".join([
        "[Script Info]",
        "ScriptType: v4.00+",
        "PlayResX: 1080",
        "PlayResY: 1920",
        "ScaledBorderAndShadow: yes",
        "WrapStyle: 2",
        "YCbCr Matrix: TV.709",
        "",
        "[V4+ Styles]",
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding",
        "Style: TikTok,DejaVu Sans,58,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,5,1,2,82,82,290,1",
        "",
        "[Events]",
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text",
        *ass_events,
        "",
    ])
    output_ass.write_text(ass, encoding="utf-8")
    duration = len(audio) / sr
    return {
        "engine": "Piper TTS",
        "voice": VOICE_ID,
        "language": "pt-BR",
        "sample_rate": sr,
        "duration_seconds": round(duration, 3),
        "sentence_count": len(cues),
        "scene_count": len(scenes),
        "scenes": [{"start": round(a + lead_pad, 3), "end": round(b + lead_pad, 3),
                    "sentence_indices": g} for a, b, g in scenes],
        "pacing": {
            "mode": pacing_mode,
            "length_scale": round(length_scale, 4),
            "speed_factor": round(1.0 / length_scale, 4),
            "tts_speedup_applied": speedup_budget_used,
            "speedup_allowed_max": 1.2,
            "line_scene_rule": "cada linha do roteiro e uma cena; nenhuma fusao entre linhas",
            "inter_sentence_gap_seconds": gap_seconds,
            "lead_padding_seconds": round(lead_pad, 3),
            "tail_padding_seconds": round(tail_pad, 3),
            "speech_only_seconds": round(total_speech, 3),
            "min_duration_target_seconds": target_min or None,
            "max_duration_target_seconds": target_max,
            "over_max_accepted": over_max_accepted,
            "over_max_tolerance_seconds": 35.0 if over_max_accepted else None,
        },
        "caption_style": "ASS queimado: branco, negrito, contorno preto, margem inferior 290px",
        "cues": [{"start": round(c.start, 3), "end": round(c.end, 3),
                  "text": c.text.replace("\\N", " ")} for c in cues],
    }
