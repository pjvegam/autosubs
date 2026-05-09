"""Recupera transcripciones perdidas en SRT usando video cacheado.

Detecta gaps entre entries del SRT, extrae el audio correspondiente del video local,
y re-transcribe con gpt-4o-transcribe (estable, evita 503 de Gemini).
"""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from openai import AsyncOpenAI

from livesubs.config import LOGS_DIR, YT_ID_RE, Config
from livesubs.costs import CostTracker
from livesubs.subtitles import _format_timestamp
from livesubs.transcribe import transcribe_audio
from livesubs.translate import translate_text

logger = logging.getLogger(__name__)

_TS_RE = re.compile(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})")
_MIN_GAP = 3.0  # segundos
_CHUNK_SECONDS = 8  # subdividir gaps largos en chunks de N segundos


@dataclass
class SrtEntry:
    start: float
    end: float
    text: str


def _parse_ts(s: str) -> float:
    m = _TS_RE.match(s)
    if not m:
        return 0.0
    return int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]) + int(m[4]) / 1000


def parse_srt(path: Path) -> list[SrtEntry]:
    """Parsea SRT a lista de entries."""
    text = path.read_text(encoding="utf-8")
    entries: list[SrtEntry] = []
    blocks = re.split(r"\n\n+", text.strip())
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        ts_line = lines[1]
        if " --> " not in ts_line:
            continue
        s, e = ts_line.split(" --> ")
        start = _parse_ts(s.strip())
        end = _parse_ts(e.strip())
        body = "\n".join(lines[2:]).strip()
        if body:
            entries.append(SrtEntry(start, end, body))
    return entries


def find_gaps(entries: list[SrtEntry], min_gap: float = _MIN_GAP) -> list[tuple[float, float]]:
    """Retorna lista de (gap_start, gap_end) >= min_gap segundos."""
    gaps = []
    for i in range(len(entries) - 1):
        gap = entries[i + 1].start - entries[i].end
        if gap >= min_gap:
            gaps.append((entries[i].end, entries[i + 1].start))
    return gaps


async def _extract_audio(
    video_path: Path, start: float, duration: float, output: Path,
) -> bool:
    """Extrae segmento de audio mono 16kHz del video con ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-i", str(video_path),
        "-t", f"{duration:.3f}",
        "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(output),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    return await proc.wait() == 0


async def _fill_gap(
    asr_client: AsyncOpenAI,
    translate_client: AsyncOpenAI,
    config: Config,
    video_path: Path,
    gap_start: float,
    gap_end: float,
    tmp_dir: Path,
    tracker: CostTracker,
) -> list[SrtEntry]:
    """Recupera un gap. Subdivide en chunks de ~8s si es largo."""
    duration = gap_end - gap_start
    gap_id = int(gap_start)

    # Subdividir el gap en sub-chunks
    sub_chunks: list[tuple[Path, float, float]] = []  # (path, start, end)
    if duration <= _CHUNK_SECONDS + 2:
        # Gap corto: un solo chunk
        chunk_path = tmp_dir / f"gap_{gap_id}.wav"
        if await _extract_audio(video_path, gap_start, duration, chunk_path):
            sub_chunks.append((chunk_path, gap_start, gap_end))
    else:
        # Gap largo: dividir en chunks de _CHUNK_SECONDS
        offset = 0.0
        i = 0
        while offset < duration:
            chunk_dur = min(_CHUNK_SECONDS, duration - offset)
            if chunk_dur < 1.0:  # ignorar últimos <1s
                break
            chunk_path = tmp_dir / f"gap_{gap_id}_{i}.wav"
            if await _extract_audio(
                video_path, gap_start + offset, chunk_dur, chunk_path,
            ):
                sub_chunks.append((
                    chunk_path,
                    gap_start + offset,
                    gap_start + offset + chunk_dur,
                ))
            offset += _CHUNK_SECONDS
            i += 1

    new_entries: list[SrtEntry] = []
    for chunk_path, ch_start, ch_end in sub_chunks:
        try:
            text = await transcribe_audio(
                asr_client, chunk_path, config.source_lang, "gpt-4o-transcribe",
            )
        except Exception as e:
            logger.warning("Transcripcion fallo en %.0fs: %s", ch_start, e)
            text = None

        if not text:
            continue

        if tracker:
            tracker.add_whisper(ch_end - ch_start)

        try:
            translated = await translate_text(
                translate_client, text, config.source_lang, config.target_lang,
                config.gpt_model, tracker=tracker,
            )
        except Exception as e:
            logger.warning("Traduccion fallo en %.0fs: %s", ch_start, e)
            translated = None

        if translated:
            new_entries.append(SrtEntry(ch_start, ch_end, translated))

    return new_entries


def write_srt(path: Path, entries: list[SrtEntry]) -> None:
    """Escribe SRT ordenado y renumerado."""
    sorted_entries = sorted(entries, key=lambda e: e.start)
    lines: list[str] = []
    for i, entry in enumerate(sorted_entries, 1):
        lines.append(
            f"{i}\n{_format_timestamp(entry.start)} --> "
            f"{_format_timestamp(entry.end)}\n{entry.text}\n",
        )
    path.write_text("\n".join(lines), encoding="utf-8")


async def fill_gaps(config: Config) -> None:
    """Repara SRT cacheado: encuentra gaps y los re-transcribe."""
    if not (m := YT_ID_RE.search(config.url)):
        logger.error("No se pudo extraer video_id de la URL")
        return
    video_id = m.group(1)

    srt_path = LOGS_DIR / f"{video_id}.srt"
    video_path = LOGS_DIR / f"{video_id}.mkv"

    if not srt_path.exists():
        logger.error("SRT no encontrado: %s", srt_path)
        return
    if not video_path.exists():
        logger.error("Video local no encontrado: %s", video_path)
        return

    entries = parse_srt(srt_path)
    gaps = find_gaps(entries)
    if not gaps:
        logger.info("No hay gaps que rellenar en %s", srt_path.name)
        return

    total_gap = sum(e - s for s, e in gaps)
    logger.info(
        "Encontrados %d gaps en %s (%.0f min %.0fs perdidos)",
        len(gaps), srt_path.name, total_gap // 60, total_gap % 60,
    )

    # Backup del SRT original
    backup = srt_path.with_suffix(".srt.bak")
    shutil.copy2(srt_path, backup)
    logger.info("Backup creado: %s", backup.name)

    # Clientes (importar el factory de pipeline)
    from livesubs.pipeline import _create_translate_client
    asr_client = AsyncOpenAI(api_key=config.api_key)
    translate_client = _create_translate_client(config.gpt_model, config.api_key)
    tracker = CostTracker(
        whisper_model="gpt-4o-transcribe",
        translation_model=config.gpt_model,
        source_lang=config.source_lang,
        target_lang=config.target_lang,
    )

    tmp_dir = Path(tempfile.mkdtemp(prefix="livesubs_fillgaps_"))
    try:
        new_entries: list[SrtEntry] = []
        for i, (gs, ge) in enumerate(gaps, 1):
            mm_s, ss_s = divmod(int(gs), 60)
            logger.info(
                "Gap %d/%d en %02d:%02d (+%.1fs)...",
                i, len(gaps), mm_s, ss_s, ge - gs,
            )
            recovered = await _fill_gap(
                asr_client, translate_client, config,
                video_path, gs, ge, tmp_dir, tracker,
            )
            new_entries.extend(recovered)
            logger.info("  → recuperadas %d entries", len(recovered))

            # Guardar progreso cada 10 gaps (resilient a Ctrl+C)
            if i % 10 == 0:
                merged = entries + new_entries
                write_srt(srt_path, merged)
                logger.info("Checkpoint: %d entries totales", len(merged))

        # Final write
        merged = entries + new_entries
        write_srt(srt_path, merged)

        logger.info(
            "Completado: %d gaps procesados, %d entries nuevas (total %d)",
            len(gaps), len(new_entries), len(merged),
        )
        logger.info("\n%s", tracker.summary())
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
