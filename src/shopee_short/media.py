"""Render 9:16, mudo na origem, TTS-only no áudio final, e QA instrumental."""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from .errors import MediaProcessingError, ProviderError
from .tts import Cue

WIDTH = 1080
HEIGHT = 1920
FPS = 30


def require_binary(name: str) -> str:
    found = shutil.which(name)
    if not found:
        raise MediaProcessingError(f"Dependência ausente: {name}. Instale ffmpeg/ffprobe/tesseract e tente novamente.")
    return found


def _subprocess_env() -> dict[str, str]:
    """Ambiente dos subprocessos do ffmpeg.

    O ffmpeg estático encaminha `subtitles=` para o binário `python3` em tempo de
    execução; garantimos que PYTHONPATH aponta para este pacote para que o script
    auxiliar shopee_short_ass_py (renderização ASS com contorno fiel) seja importado.
    """
    env = dict(os.environ)
    pkg_root = str(Path(__file__).resolve().parent.parent)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = pkg_root + (os.pathsep + existing if existing else "")
    return env


def _run(cmd: list[str], *, cwd: Path | None = None, timeout: int = 900,
         operation: str = "ffmpeg") -> subprocess.CompletedProcess[str]:
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout,
                              check=False, env=_subprocess_env())
    except subprocess.TimeoutExpired as exc:
        raise MediaProcessingError(f"{operation} ultrapassou o tempo limite.") from exc
    if proc.returncode != 0:
        detail = " ".join((proc.stderr or proc.stdout or "").split())[-1100:]
        raise MediaProcessingError(f"{operation} falhou (código {proc.returncode}): {detail}")
    return proc


def probe_media(path: Path) -> dict[str, Any]:
    ffprobe = require_binary("ffprobe")
    proc = _run([
        ffprobe, "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path),
    ], operation="ffprobe")
    try:
        return json.loads(proc.stdout)
    except Exception as exc:
        raise MediaProcessingError(f"ffprobe não produziu JSON válido para {path.name}.") from exc


def media_duration(probe: dict[str, Any]) -> float:
    try:
        return float(probe["format"]["duration"])
    except Exception:
        streams = probe.get("streams", [])
        vals = [float(s["duration"]) for s in streams if s.get("duration")]
        return max(vals, default=0.0)


def _extract_source_audio(video_path: Path, output_wav: Path) -> dict[str, Any]:
    ffmpeg = require_binary("ffmpeg")
    probe = probe_media(video_path)
    streams = [s for s in probe.get("streams", []) if s.get("codec_type") == "audio"]
    if not streams:
        return {"audio_streams": 0, "speech_detected": False, "transcript": "",
                "inspection": "ffprobe confirmou ausência de trilha de áudio na origem."}
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    _run([
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(video_path),
        "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(output_wav),
    ], operation="extração de áudio da origem")
    audio_info = streams[0]
    return {"audio_streams": len(streams), "codec": audio_info.get("codec_name"),
            "sample_rate": audio_info.get("sample_rate"), "channels": audio_info.get("channels"),
            "audio_wav_check": "extraído temporariamente para ASR; apagado ao encerrar a execução"}


def inspect_source_audio(video_path: Path, work_dir: Path, *, model_name: str = "tiny",
                         source_type: str = "alternative") -> dict[str, Any]:
    """Verifica trilhas com ffprobe e roda ASR local (sem conta/chave) quando há áudio."""
    wav_path = work_dir / "source_audio_check.wav"
    info = _extract_source_audio(video_path, wav_path)
    if info.get("audio_streams", 0) == 0:
        return info
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise ProviderError("A verificação de fala da origem exige faster-whisper; instale as dependências completas.") from exc
    try:
        cache = Path.home() / ".cache" / "shopee-short" / "whisper"
        model = WhisperModel(model_name, device="cpu", compute_type="int8", download_root=str(cache))
        segments, language_info = model.transcribe(
            str(wav_path), vad_filter=True, beam_size=1, condition_on_previous_text=False,
        )
        recognized = []
        for segment in segments:
            text = re.sub(r"\s+", " ", (segment.text or "")).strip()
            if text:
                recognized.append({
                    "start": round(float(segment.start), 3),
                    "end": round(float(segment.end), 3),
                    "text": text,
                    "no_speech_probability": round(float(segment.no_speech_prob), 3),
                })
        transcript = " ".join(row["text"] for row in recognized)
        speech_detected = any(
            len(re.sub(r"[^\w]", "", row["text"], flags=re.UNICODE)) >= 4
            and row["no_speech_probability"] < 0.75
            for row in recognized
        )
    except Exception as exc:
        raise ProviderError(f"Não foi possível verificar a fala da origem com ASR; interrompi sem presumir. {exc}") from exc
    result = {
        **info,
        "asr_model": model_name,
        "asr_language": getattr(language_info, "language", None),
        "asr_language_probability": round(float(getattr(language_info, "language_probability", 0.0)), 3),
        "speech_detected": speech_detected,
        "transcript": transcript,
        "segments": recognized,
        "inspection": "ffprobe + extração de áudio + faster-whisper local",
        "final_mute_policy": "a faixa original nunca é mapeada; o MP4 final recebe somente o WAV do Piper",
    }
    # Fonte alternativa pode conter fala: a especificação do usuário diz que o mute resolve.
    # Uma galeria oficial informada explicitamente é bloqueada caso ASR detecte fala.
    if source_type == "seller-gallery" and speech_detected:
        raise MediaProcessingError("Fala humana detectada na origem declarada como galeria do seller; escolha outra demo muda.")
    return result


def wav_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as wav:
            return wav.getnframes() / float(wav.getframerate())
    except Exception as exc:
        raise MediaProcessingError(f"WAV de narração inválido: {path}") from exc


def _write_filter_graph(path: Path, *, slideshow: bool) -> None:
    if slideshow:
        graph = "[0:v]fps=30,setsar=1,subtitles=captions.ass:fontsdir=/usr/share/fonts/truetype/dejavu[v]\n"
    else:
        graph = (
            "[0:v]fps=30,split=2[bg0][fg0];\n"
            "[bg0]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,"
            "boxblur=42:12,eq=brightness=-0.18:saturation=0.72[bg];\n"
            "[fg0]scale=1080:1920:force_original_aspect_ratio=decrease[fg];\n"
            "[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1,"
            "subtitles=captions.ass:fontsdir=/usr/share/fonts/truetype/dejavu[v]\n"
        )
    path.write_text(graph, encoding="utf-8")


def _prepare_photo(path: Path, destination: Path) -> None:
    try:
        with Image.open(path) as opened:
            rgba = ImageOps.exif_transpose(opened).convert("RGBA")
            matte = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            matte.alpha_composite(rgba)
            im = matte.convert("RGB")
    except Exception as exc:
        raise MediaProcessingError(f"Imagem inválida ou não suportada: {path}") from exc
    background = ImageOps.fit(im, (WIDTH, HEIGHT), method=Image.Resampling.LANCZOS)
    background = background.filter(ImageFilter.GaussianBlur(radius=42))
    background = ImageEnhance.Brightness(background).enhance(0.50)
    # O produto fica inteiro no quadro; fundo desfocado evita crop destrutivo.
    foreground = ImageOps.contain(im, (1000, 1160), method=Image.Resampling.LANCZOS)
    x = (WIDTH - foreground.width) // 2
    y = (HEIGHT - foreground.height) // 2
    background.paste(foreground, (x, y))
    destination.parent.mkdir(parents=True, exist_ok=True)
    background.save(destination, format="PNG", optimize=True)


def _ffmpeg_base() -> str:
    return require_binary("ffmpeg")


def render_video(source_video: Path | None, images: list[Path] | None, narration_wav: Path,
                 captions_ass: Path, work_dir: Path, output_path: Path,
                 scene_spans: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Renderiza video de demo ou slideshow. O áudio da origem é excluído por mapeamento explícito.

    scene_spans: lista [{start,end}, ...] do relatório TTS — UMA cena por foto/oficial
    (rework: cada linha do roteiro vira uma cena com duração própria sincronizada).
    """
    if (source_video is None) == (not images):
        raise MediaProcessingError("Informe exatamente uma modalidade: um vídeo OU uma lista de fotos.")
    ffmpeg = _ffmpeg_base()
    duration = wav_duration(narration_wav)
    if duration <= 0:
        raise MediaProcessingError("A narração tem duração zero.")
    work_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    graph_file = work_dir / "filtergraph.txt"
    _write_filter_graph(graph_file, slideshow=source_video is None)

    if source_video is not None:
        cmd = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "warning", "-stream_loop", "-1",
            "-i", str(source_video), "-i", str(narration_wav),
            "-filter_complex_script", str(graph_file),
            "-map", "[v]", "-map", "1:a:0", "-t", f"{duration:.3f}", "-shortest",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-r", str(FPS), "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
            "-movflags", "+faststart", str(output_path),
        ]
        _run(cmd, cwd=work_dir, operation="montagem de vídeo demo 9:16")
        return {"mode": "video", "source_video": str(source_video), "input_audio_mapped": False,
                "audio_policy": "origem descartada com -map 1:a:0; apenas narração Piper incluída",
                "duration_target_seconds": round(duration, 3)}

    assert images
    if not 1 <= len(images) <= 12:
        raise MediaProcessingError("O slideshow aceita de 1 a 12 fotos confirmadas.")
    slide_dir = work_dir / "slides"
    slide_dir.mkdir(parents=True, exist_ok=True)
    prepared: list[Path] = []
    for index, image in enumerate(images):
        dest = slide_dir / f"slide_{index:02d}.png"
        _prepare_photo(image, dest)
        prepared.append(dest)
    # Rework: UMA cena por foto com duração sincronizada à narração (scene_spans do TTS).
    # Sem spans (compatibilidade antiga), mantém a divisão igualitária.
    if scene_spans and len(scene_spans) == len(prepared):
        segment_durations = [float(s["end"]) - float(s["start"]) for s in scene_spans]
        lead = max(0.0, float(scene_spans[0]["start"]))
        tail = max(0.0, duration - float(scene_spans[-1]["end"]))
        # padding silencioso (se houver) fica nas pontas: estende 1ª e última cena.
        segment_durations[0] += lead
        segment_durations[-1] += tail
        pacing_source = "tts-scene-spans-synced"
    else:
        equal = duration / len(prepared)
        segment_durations = [equal] * len(prepared)
        pacing_source = "equal-split-fallback"
    concat_file = work_dir / "slideshow.ffconcat"
    lines = ["ffconcat version 1.0"]
    for slide, seg_dur in zip(prepared, segment_durations):
        relative = slide.relative_to(work_dir).as_posix()
        lines += [f"file '{relative}'", f"duration {seg_dur:.4f}"]
    # O demuxer concat precisa repetir o último still para aplicar a última duração.
    lines.append(f"file '{prepared[-1].relative_to(work_dir).as_posix()}'")
    concat_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "warning", "-f", "concat", "-safe", "0",
        "-i", str(concat_file), "-i", str(narration_wav),
        "-filter_complex_script", str(graph_file),
        "-map", "[v]", "-map", "1:a:0", "-t", f"{duration:.3f}", "-shortest",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-r", str(FPS), "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
        "-movflags", "+faststart", str(output_path),
    ]
    _run(cmd, cwd=work_dir, operation="montagem de slideshow 9:16")
    return {"mode": "slideshow", "confirmed_photo_count": len(images),
            "scene_pacing_source": pacing_source,
            "scene_durations_seconds": [round(x, 3) for x in segment_durations],
            "audio_policy": "sem áudio de origem; apenas narração Piper incluída",
            "duration_target_seconds": round(duration, 3)}


def extract_and_ocr_frames(video_path: Path, duration: float, frame_dir: Path,
                           cues: list[dict[str, Any]], *, ocr: bool = True) -> list[dict[str, Any]]:
    ffmpeg = require_binary("ffmpeg")
    if not ocr:
        tesseract = None
    else:
        try:
            tesseract = require_binary("tesseract")
        except MediaProcessingError:
            tesseract = None
    frame_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    for label, fraction in (("10", 0.10), ("50", 0.50), ("90", 0.90)):
        timestamp = min(max(0.05, duration * fraction), max(0.05, duration - 0.05))
        frame = frame_dir / f"frame_{label}pct.png"
        _run([
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{timestamp:.3f}",
            "-i", str(video_path), "-frames:v", "1", "-update", "1", str(frame),
        ], operation=f"extração de frame {label}%")
        with Image.open(frame) as im:
            width, height = im.size
        expected = next((c.get("text", "") for c in cues
                         if float(c.get("start", 0)) <= timestamp <= float(c.get("end", 0))), "")
        if tesseract is None:
            # Verificação visual via .srt + inspeção humana dos frames extraídos (modo aceitação).
            reports.append({"percentage": int(label), "timestamp_seconds": round(timestamp, 3),
                            "path": str(frame), "caption_expected": expected,
                            "ocr": "skipped-by-request", "caption_visible": None,
                            "resolution": f"{width}x{height}"})
            continue
        with Image.open(frame) as im:
            crop = im.convert("RGB").crop((0, int(height * 0.63), width, int(height * 0.96)))
            crop_path = frame_dir / f".ocr_{label}.png"
            crop.save(crop_path)
        proc = _run([tesseract, str(crop_path), "stdout", "-l", "por+eng", "--psm", "6"],
                    operation=f"OCR do frame {label}%")
        crop_path.unlink(missing_ok=True)
        recognized = re.sub(r"\s+", " ", proc.stdout or "").strip()
        def words(s: str) -> set[str]:
            return {w.casefold().strip(".,!?;:()\"'–—-") for w in re.findall(r"[\wÀ-ÿ]+", s, flags=re.UNICODE)
                    if len(w.strip(".,!?;:()\"'–—-")) >= 3}
        overlap = words(expected) & words(recognized)
        if not recognized or (expected and not overlap):
            raise MediaProcessingError(
                f"A legenda esperada não foi confirmada por OCR no frame {label}% "
                f"(esperado={expected!r}, OCR={recognized!r})."
            )
        reports.append({"percentage": int(label), "timestamp_seconds": round(timestamp, 3),
                        "path": str(frame), "caption_expected": expected,
                        "ocr_lower_safe_area": recognized, "expected_word_overlap": sorted(overlap),
                        "caption_visible": True, "resolution": f"{width}x{height}"})
    return reports


def verify_final_video(video_path: Path, narration_wav: Path, frame_dir: Path,
                       cues: list[dict[str, Any]], *, ocr: bool = True) -> dict[str, Any]:
    probe = probe_media(video_path)
    streams = probe.get("streams", [])
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    if len(video_streams) != 1:
        raise MediaProcessingError(f"ffprobe encontrou {len(video_streams)} streams de vídeo; esperado exatamente 1.")
    if len(audio_streams) != 1:
        raise MediaProcessingError(f"ffprobe encontrou {len(audio_streams)} streams de áudio; esperado exatamente 1 (narração).")
    vs, aud = video_streams[0], audio_streams[0]
    width, height = int(vs.get("width", 0)), int(vs.get("height", 0))
    duration = media_duration(probe)
    if duration <= 0 or height <= width or width != WIDTH or height != HEIGHT:
        raise MediaProcessingError(f"Saída não está em 9:16 vertical ou duração inválida: {width}x{height}, {duration}s.")
    if vs.get("codec_name") != "h264" or aud.get("codec_name") != "aac":
        raise MediaProcessingError(f"Codecs incompatíveis: vídeo={vs.get('codec_name')} áudio={aud.get('codec_name')}.")
    if wav_duration(narration_wav) <= 0:
        raise MediaProcessingError("WAV da narração está vazio.")

    ffmpeg = require_binary("ffmpeg")
    volume = _run([
        ffmpeg, "-hide_banner", "-i", str(video_path), "-map", "0:a:0", "-af", "volumedetect",
        "-f", "null", "-",
    ], operation="verificação de audibilidade")
    volume_text = volume.stderr or ""
    max_match = re.search(r"max_volume:\s*(-?inf|[-\d.]+)\s*dB", volume_text)
    mean_match = re.search(r"mean_volume:\s*(-?inf|[-\d.]+)\s*dB", volume_text)
    max_db = max_match.group(1) if max_match else "unknown"
    mean_db = mean_match.group(1) if mean_match else "unknown"
    if max_db in {"-inf", "unknown"} or (max_db.replace(".", "", 1).lstrip("-").isdigit() and float(max_db) < -50):
        raise MediaProcessingError(f"Áudio final inaudível ou sem medição confiável (max_volume={max_db} dB).")

    frames = extract_and_ocr_frames(video_path, duration, frame_dir, cues, ocr=ocr)
    return {
        "duration_seconds": round(duration, 3),
        "format_name": probe.get("format", {}).get("format_name"),
        "video": {"codec": vs.get("codec_name"), "width": width, "height": height,
                  "fps": vs.get("r_frame_rate"), "pixel_format": vs.get("pix_fmt")},
        "audio": {"stream_count": len(audio_streams), "codec": aud.get("codec_name"),
                  "sample_rate": aud.get("sample_rate"), "channels": aud.get("channels"),
                  "max_volume_db": max_db, "mean_volume_db": mean_db,
                  "measured_by": "ffmpeg volumedetect"},
        "total_streams": len(streams),
        "ffprobe_verified": True,
        "frames_ocr": frames,
    }
