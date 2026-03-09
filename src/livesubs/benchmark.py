"""Benchmark: compara modelos de transcripcion sobre el mismo audio."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from openai import AsyncOpenAI

from livesubs.config import Config
from livesubs.costs import CostTracker, _escape_md_table
from livesubs.stream import download_audio, set_cookies_browser
from livesubs.transcribe import transcribe_audio, transcribe_verbose
from livesubs.translate import translate_text
from livesubs.vad import SAMPLE_RATE, AudioChunk, VadSegmenter

logger = logging.getLogger(__name__)

LOGS_DIR = Path("livesubs_logs")
_YT_ID_RE = re.compile(r"(?:v=|youtu\.be/|/shorts/)([a-zA-Z0-9_-]{11})")
_BENCHMARK_DURATION = 1200  # 20 minutos


@dataclass
class BenchmarkEntry:
    start: float
    end: float
    text: str
    translation: str


@dataclass
class BenchmarkResult:
    model: str
    entries: list[BenchmarkEntry] = field(default_factory=list)
    tracker: CostTracker = field(default_factory=CostTracker)
    transcribe_seconds: float = 0.0
    translate_seconds: float = 0.0


# -- Checkpoint: serializar/deserializar resultados parciales --

def _checkpoint_path(video_id: str, start_time: int) -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    return LOGS_DIR / f"benchmark_cache_{video_id}_{start_time}.json"


def _result_to_dict(r: BenchmarkResult) -> dict:
    return {
        "model": r.model,
        "transcribe_seconds": r.transcribe_seconds,
        "translate_seconds": r.translate_seconds,
        "entries": [
            {"start": e.start, "end": e.end, "text": e.text, "translation": e.translation}
            for e in r.entries
        ],
        "tracker": {
            "whisper_seconds": r.tracker.whisper_seconds,
            "whisper_model": r.tracker.whisper_model,
            "translation_prompt_tokens": r.tracker.translation_prompt_tokens,
            "translation_completion_tokens": r.tracker.translation_completion_tokens,
            "translation_model": r.tracker.translation_model,
            "source_lang": r.tracker.source_lang,
            "target_lang": r.tracker.target_lang,
        },
    }


def _dict_to_result(d: dict) -> BenchmarkResult:
    t = d["tracker"]
    tracker = CostTracker(
        whisper_seconds=t["whisper_seconds"],
        whisper_model=t["whisper_model"],
        translation_prompt_tokens=t["translation_prompt_tokens"],
        translation_completion_tokens=t["translation_completion_tokens"],
        translation_model=t["translation_model"],
        source_lang=t["source_lang"],
        target_lang=t["target_lang"],
    )
    entries = [
        BenchmarkEntry(start=e["start"], end=e["end"], text=e["text"], translation=e["translation"])
        for e in d["entries"]
    ]
    return BenchmarkResult(
        model=d["model"],
        entries=entries,
        tracker=tracker,
        transcribe_seconds=d["transcribe_seconds"],
        translate_seconds=d["translate_seconds"],
    )


def _save_checkpoint(
    cp_path: Path,
    results: list[BenchmarkResult],
    segment_file: str,
) -> None:
    data = {
        "segment_file": segment_file,
        "results": [_result_to_dict(r) for r in results],
    }
    cp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Checkpoint guardado: %s (%d paths)", cp_path.name, len(results))


def _load_checkpoint(cp_path: Path) -> tuple[str, list[BenchmarkResult]] | None:
    if not cp_path.exists():
        return None
    try:
        data = json.loads(cp_path.read_text(encoding="utf-8"))
        segment_file = data["segment_file"]
        if not Path(segment_file).exists():
            logger.info("Checkpoint cargado pero segment.mp3 no existe, se re-extraera")
            segment_file = ""
        results = [_dict_to_result(d) for d in data["results"]]
        # Migrar nombre viejo -> nuevo
        for r in results:
            if r.model == "faster-whisper-local":
                r.model = "faster-whisper-turbo"
        logger.info(
            "Checkpoint cargado: %s (%d paths)",
            cp_path.name, len(results),
        )
        return segment_file, results
    except (json.JSONDecodeError, KeyError):
        logger.warning("Checkpoint corrupto, ignorando: %s", cp_path.name)
        return None


def _result_needs_translation(r: BenchmarkResult) -> bool:
    return any(not e.translation for e in r.entries)


# -- Benchmark principal --

async def run_benchmark(config: Config) -> None:
    """Orquesta el benchmark: descarga, segmenta, compara paths."""
    import faulthandler
    faulthandler.enable()

    set_cookies_browser(config.cookies_browser)
    client = AsyncOpenAI(api_key=config.api_key)

    video_id = ""
    if m := _YT_ID_RE.search(config.url):
        video_id = m.group(1)

    cp_path = _checkpoint_path(video_id, config.start_time)
    cached = _load_checkpoint(cp_path)

    # Si hay cache valido, intentar resumir
    if cached:
        segment_file, results = cached
        models_done = {r.model for r in results}

        # Si todo esta completo (transcrito + traducido), solo regenerar markdown
        all_translated = all(not _result_needs_translation(r) for r in results)
        has_real_entries = any(r.entries for r in results)
        if all_translated and has_real_entries and len(models_done) >= 2:
            logger.info("Benchmark completo en cache, regenerando reporte...")
            now = datetime.now()
            md = _generate_comparison(results, config, video_id, now)
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            output_path = LOGS_DIR / f"benchmark_{video_id}_{now:%Y-%m-%d_%H-%M-%S}.md"
            output_path.write_text(md, encoding="utf-8")
            logger.info("Benchmark guardado en: %s", output_path)
            return

        logger.info("Resumiendo benchmark, paths cacheados: %s", models_done)
    else:
        segment_file = ""
        results = []
        models_done = set()

    # Guardar segment en logs_dir (persistente, no tmp)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    persistent_segment = str(LOGS_DIR / f"segment_{video_id}_{config.start_time}.mp3")

    tmp_dir = Path(tempfile.mkdtemp(prefix="livesubs_bench_"))
    try:
        # 1-2. Descargar y extraer segmento (si no hay cache)
        if not segment_file:
            audio_file = str(tmp_dir / "audio.mp3")
            await download_audio(config.url, audio_file)

            await _extract_segment(
                audio_file, persistent_segment, config.start_time, _BENCHMARK_DURATION,
            )
            Path(audio_file).unlink(missing_ok=True)
            segment_file = persistent_segment

        # 3. Transcribir paths (secuencial para evitar abort de ctranslate2)
        has_local = _faster_whisper_available()

        # faster-whisper turbo (large-v3-turbo)
        if has_local and "faster-whisper-turbo" not in models_done:
            logger.info(
                "Transcribiendo faster-whisper turbo (%ds desde %ds)",
                _BENCHMARK_DURATION, config.start_time,
            )
            result_turbo = await _transcribe_local_subprocess(
                segment_file, config, model_size="turbo", model_name="faster-whisper-turbo",
            )
            results.append(result_turbo)
            _save_checkpoint(cp_path, results, segment_file)

        # faster-whisper large-v3
        if has_local and "faster-whisper-large-v3" not in models_done:
            logger.info(
                "Transcribiendo faster-whisper large-v3 (%ds desde %ds)",
                _BENCHMARK_DURATION, config.start_time,
            )
            result_v3 = await _transcribe_local_subprocess(
                segment_file, config, model_size="large-v3", model_name="faster-whisper-large-v3",
            )
            results.append(result_v3)
            _save_checkpoint(cp_path, results, segment_file)

        if not has_local:
            logger.warning(
                "faster-whisper no instalado (instalar con: uv sync --extra local-whisper)",
            )

        # Path A: gpt-4o-transcribe API
        if "gpt-4o-transcribe" not in models_done:
            logger.info(
                "Transcribiendo gpt-4o-transcribe API (%ds desde %ds)",
                _BENCHMARK_DURATION, config.start_time,
            )
            result_a = await _transcribe_path_a(
                client, segment_file, tmp_dir / "path_a", config,
            )
            results.append(result_a)
            _save_checkpoint(cp_path, results, segment_file)

        # Fallback Path B (solo si no hay faster-whisper)
        if not has_local and "whisper-1" not in models_done:
            result_b = await _transcribe_path_b(client, segment_file, config)
            results.append(result_b)
            _save_checkpoint(cp_path, results, segment_file)

        logger.info(
            "Transcripcion completa: %s",
            ", ".join(f"{r.model}={len(r.entries)} entries" for r in results),
        )

        # 4. Traducir (solo los que faltan)
        for result in results:
            if _result_needs_translation(result):
                logger.info(
                    "Traduciendo %s (%d entries)...", result.model, len(result.entries),
                )
                await _translate_entries(client, result, config)
                _save_checkpoint(cp_path, results, segment_file)
            else:
                logger.info("%s ya traducido, saltando", result.model)

        # 5. Generar markdown comparativo
        now = datetime.now()
        md = _generate_comparison(results, config, video_id, now)
        output_path = LOGS_DIR / f"benchmark_{video_id}_{now:%Y-%m-%d_%H-%M-%S}.md"
        output_path.write_text(md, encoding="utf-8")
        logger.info("Benchmark guardado en: %s", output_path)

        # Resumen de costos
        total_cost = 0.0
        cost_parts: list[str] = []
        for result in results:
            cost = result.tracker._whisper_cost() + sum(result.tracker._translation_cost())
            total_cost += cost
            cost_parts.append(f"{result.model}: ${cost:.4f}")
        logger.info(
            "Costo total benchmark: $%.4f (%s)",
            total_cost, ", ".join(cost_parts),
        )

        # Guardar segment en checkpoint para re-ejecuciones futuras
        _save_checkpoint(cp_path, results, segment_file)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# -- Extraccion y segmentacion --

async def _extract_segment(
    input_path: str, output_path: str, start_seconds: int, duration: int = 1200,
) -> None:
    """Extrae un segmento de audio con ffmpeg -ss -t."""
    cmd = [
        "ffmpeg", "-y", "-ss", str(start_seconds),
        "-i", input_path, "-t", str(duration),
        "-c:a", "libmp3lame", "-q:a", "5", output_path,
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    rc = await proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg segment extraction fallo (exit {rc})")

    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    mm_start, ss_start = divmod(start_seconds, 60)
    mm_end, ss_end = divmod(start_seconds + duration, 60)
    logger.info(
        "Segmento extraido: %02d:%02d-%02d:%02d (%.1f MB)",
        mm_start, ss_start, mm_end, ss_end, size_mb,
    )


async def _segment_audio_with_vad(
    audio_path: str, output_dir: Path, config: Config,
) -> list[AudioChunk]:
    """Convierte MP3 a PCM y segmenta con VAD. Retorna lista de chunks."""
    output_dir.mkdir(parents=True, exist_ok=True)

    segmenter = VadSegmenter(
        output_dir,
        min_duration=config.vad_min_duration,
        target_duration=float(config.segment_duration),
        max_duration=config.vad_max_duration,
        aggressiveness=config.vad_aggressiveness,
    )

    # ffmpeg -> PCM 16kHz mono
    cmd = [
        "ffmpeg", "-loglevel", "warning",
        "-i", audio_path,
        "-ar", str(SAMPLE_RATE), "-ac", "1",
        "-f", "s16le", "-acodec", "pcm_s16le",
        "pipe:1",
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    if proc.stdout is None:
        await proc.wait()
        raise RuntimeError("ffmpeg no abrio stdout")

    read_size = SAMPLE_RATE * 2  # ~1s
    chunks: list[AudioChunk] = []
    while True:
        data = await proc.stdout.read(read_size)
        if not data:
            break
        chunks.extend(segmenter.feed(data))

    if remaining := segmenter.flush_remaining():
        chunks.append(remaining)

    await proc.wait()
    logger.info("VAD segmento: %d chunks generados", len(chunks))
    return chunks


# -- Paths de transcripcion --

async def _transcribe_path_a(
    client: AsyncOpenAI, audio_path: str, work_dir: Path, config: Config,
) -> BenchmarkResult:
    """Path A: transcribe con gpt-4o-transcribe + VAD ~5s (sin traducir)."""
    tracker = CostTracker(
        whisper_model="gpt-4o-transcribe",
        translation_model=config.gpt_model,
        source_lang=config.source_lang,
        target_lang=config.target_lang,
    )
    result = BenchmarkResult(model="gpt-4o-transcribe", tracker=tracker)

    chunks = await _segment_audio_with_vad(audio_path, work_dir, config)

    semaphore = asyncio.Semaphore(3)
    completed = 0

    async def _transcribe_chunk(chunk: AudioChunk) -> str | None:
        nonlocal completed
        async with semaphore:
            try:
                text = await transcribe_audio(
                    client, chunk.path, config.source_lang, "gpt-4o-transcribe",
                )
                tracker.add_whisper(chunk.duration)
                completed += 1
                if completed % 20 == 0:
                    logger.info("Path A progreso: %d/%d chunks", completed, len(chunks))
                return text
            finally:
                chunk.path.unlink(missing_ok=True)

    offsets: list[float] = []
    elapsed = 0.0
    for chunk in chunks:
        offsets.append(elapsed)
        elapsed += chunk.duration

    logger.info("Path A: lanzando %d transcripciones API...", len(chunks))
    t0 = time.perf_counter()
    tasks = [asyncio.create_task(_transcribe_chunk(c)) for c in chunks]
    transcriptions = [await t for t in tasks]
    result.transcribe_seconds = time.perf_counter() - t0

    for text, offset, chunk in zip(transcriptions, offsets, chunks, strict=True):
        if text:
            result.entries.append(BenchmarkEntry(
                start=offset, end=offset + chunk.duration,
                text=text, translation="",
            ))

    logger.info(
        "Path A transcrito: %d entries (%.1fs)", len(result.entries), result.transcribe_seconds,
    )
    return result


async def _transcribe_path_b(
    client: AsyncOpenAI, audio_path: str, config: Config,
) -> BenchmarkResult:
    """Path B: transcribe con whisper-1 + verbose_json (sin traducir)."""
    tracker = CostTracker(
        whisper_model="whisper-1",
        translation_model=config.gpt_model,
        source_lang=config.source_lang,
        target_lang=config.target_lang,
    )
    result = BenchmarkResult(model="whisper-1", tracker=tracker)

    t0 = time.perf_counter()
    segments = await transcribe_verbose(client, Path(audio_path), config.source_lang)
    result.transcribe_seconds = time.perf_counter() - t0

    if not segments:
        logger.warning("Path B: transcribe_verbose retorno vacio")
        return result

    tracker.add_whisper(_BENCHMARK_DURATION)

    for seg in segments:
        result.entries.append(BenchmarkEntry(
            start=seg.start, end=seg.end,
            text=seg.text, translation="",
        ))

    logger.info(
        "Path B transcrito: %d entries (%.1fs)", len(result.entries), result.transcribe_seconds,
    )
    return result


def _faster_whisper_available() -> bool:
    """Chequea si faster-whisper esta instalado."""
    try:
        import faster_whisper  # noqa: F401

        return True
    except ImportError:
        return False


async def _transcribe_local_subprocess(
    audio_path: str,
    config: Config,
    model_size: str = "turbo",
    model_name: str = "faster-whisper-local",
) -> BenchmarkResult:
    """Transcribe con faster-whisper en subprocess aislado.

    ctranslate2 aborta el proceso al liberar recursos CUDA, asi que
    corremos faster-whisper en un proceso hijo que puede morir sin afectar
    al proceso principal.
    """
    import sys

    tracker = CostTracker(
        whisper_model="faster-whisper-local",
        translation_model=config.gpt_model,
        source_lang=config.source_lang,
        target_lang=config.target_lang,
    )
    result = BenchmarkResult(model=model_name, tracker=tracker)

    # Script inline que ejecuta faster-whisper y escribe JSON a stdout.
    # os._exit(0) evita la limpieza de Python que triggerea el abort de ctranslate2.
    script = f"""
import json, os, sys, time
sys.path.insert(0, {str(Path(__file__).resolve().parent.parent)!r})
from livesubs.local_whisper import LocalWhisperModel
from pathlib import Path

model = LocalWhisperModel(model_size={model_size!r})
t0 = time.perf_counter()
segments = model.transcribe(Path({audio_path!r}), {config.source_lang!r})
elapsed = time.perf_counter() - t0

data = {{
    "elapsed": elapsed,
    "segments": [{{"text": s.text, "start": s.start, "end": s.end}} for s in segments],
}}
sys.stdout.write(json.dumps(data))
sys.stdout.flush()
os._exit(0)
"""

    t0 = time.perf_counter()
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-c", script,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    wall_time = time.perf_counter() - t0

    if proc.returncode != 0:
        logger.error("%s subprocess fallo (exit %d)", model_name, proc.returncode)
        if stderr:
            for line in stderr.decode(errors="replace").splitlines():
                logger.error("%s stderr: %s", model_name, line)
        return result

    if stderr:
        for line in stderr.decode(errors="replace").splitlines():
            logger.debug("%s stderr: %s", model_name, line)

    try:
        data = json.loads(stdout.decode())
    except (json.JSONDecodeError, UnicodeDecodeError):
        logger.error("%s: no se pudo parsear stdout del subprocess", model_name)
        return result

    result.transcribe_seconds = data["elapsed"]
    tracker.add_whisper(_BENCHMARK_DURATION)

    for seg in data["segments"]:
        result.entries.append(BenchmarkEntry(
            start=seg["start"], end=seg["end"],
            text=seg["text"], translation="",
        ))

    logger.info(
        "%s transcrito: %d entries (%.1fs subprocess, %.1fs wall)",
        model_name, len(result.entries), result.transcribe_seconds, wall_time,
    )
    return result


# -- Traduccion --

async def _translate_entries(
    client: AsyncOpenAI, result: BenchmarkResult, config: Config,
) -> None:
    """Traduce todas las entries de un BenchmarkResult, secuencialmente."""
    t0 = time.perf_counter()
    translated_entries: list[BenchmarkEntry] = []
    for entry in result.entries:
        if entry.translation:
            translated_entries.append(entry)
            continue
        translated = await translate_text(
            client, entry.text, config.source_lang, config.target_lang,
            config.gpt_model, tracker=result.tracker,
        )
        if translated:
            entry.translation = translated
            translated_entries.append(entry)

    result.translate_seconds = time.perf_counter() - t0
    result.entries = translated_entries
    logger.info(
        "%s traducido: %d entries (%.1fs)",
        result.model, len(result.entries), result.translate_seconds,
    )


# -- Metricas --

def _count_hallucinations(entries: list[BenchmarkEntry]) -> int:
    """Detecta alucinaciones: repeticiones consecutivas, ★, texto < 2 chars."""
    count = 0
    prev_text = ""
    for entry in entries:
        text = entry.text.strip()
        if len(text) < 2 or "★" in text or text == prev_text:
            count += 1
        prev_text = text
    return count


def _count_short_segments(entries: list[BenchmarkEntry], min_chars: int = 3) -> int:
    """Cuenta segmentos con texto muy corto."""
    return sum(1 for e in entries if len(e.text.strip()) < min_chars)


def _total_chars(entries: list[BenchmarkEntry]) -> int:
    return sum(len(e.text) for e in entries)


def _find_entry_near(entries: list[BenchmarkEntry], target_time: float) -> BenchmarkEntry | None:
    """Encuentra el entry mas cercano a target_time."""
    if not entries:
        return None
    return min(entries, key=lambda e: abs(e.start - target_time))


# -- Generacion de reporte --

def _generate_comparison(
    results: list[BenchmarkResult],
    config: Config,
    video_id: str,
    now: datetime,
) -> str:
    """Genera markdown comparativo entre N paths."""
    end_time = config.start_time + _BENCHMARK_DURATION
    mm_start, ss_start = divmod(config.start_time, 60)
    mm_end, ss_end = divmod(end_time, 60)

    models = [r.model for r in results]
    header = " | ".join(models)
    separator = " | ".join("---" for _ in models)

    lines: list[str] = [
        f"# Benchmark LiveSubs — {video_id}",
        "",
        f"**Fecha:** {now:%Y-%m-%d %H:%M} | "
        f"**Audio:** {mm_start:02d}:{ss_start:02d}-{mm_end:02d}:{ss_end:02d} | "
        f"**Idioma:** {config.source_lang} -> {config.target_lang}",
        "",
        "## Metricas",
        "",
        f"| Metrica | {header} |",
        f"|---------|{' | '.join('---' for _ in models)}|",
    ]

    # Filas de metricas
    def _row(label: str, values: list[str]) -> str:
        return f"| {label} | {' | '.join(values)} |"

    lines.append(_row(
        "Subtitulos generados", [str(len(r.entries)) for r in results],
    ))
    lines.append(_row(
        "Caracteres transcritos", [str(_total_chars(r.entries)) for r in results],
    ))
    lines.append(_row(
        "Alucinaciones detectadas",
        [str(_count_hallucinations(r.entries)) for r in results],
    ))
    lines.append(_row(
        "Segmentos vacios/cortos",
        [str(_count_short_segments(r.entries)) for r in results],
    ))
    lines.append(_row(
        "Tiempo transcripcion",
        [f"{r.transcribe_seconds:.1f}s" for r in results],
    ))
    lines.append(_row(
        "Tiempo traduccion",
        [f"{r.translate_seconds:.1f}s" for r in results],
    ))
    total_time = [r.transcribe_seconds + r.translate_seconds for r in results]
    lines.append(_row(
        "Tiempo total",
        [f"{t:.1f}s" for t in total_time],
    ))
    # RTF < 1.0 = mas rapido que real-time (viable para live)
    lines.append(_row(
        "Real-time factor",
        [f"{r.transcribe_seconds / _BENCHMARK_DURATION:.2f}x" for r in results],
    ))
    avg_per_entry = [
        f"{r.transcribe_seconds / len(r.entries):.2f}s" if r.entries else "—"
        for r in results
    ]
    lines.append(_row("Promedio transcripcion/entry", avg_per_entry))
    lines.append(_row(
        "Costo whisper",
        [f"${r.tracker._whisper_cost():.4f}" for r in results],
    ))
    lines.append(_row(
        "Costo traduccion",
        [f"${sum(r.tracker._translation_cost()):.4f}" for r in results],
    ))

    totals = [
        r.tracker._whisper_cost() + sum(r.tracker._translation_cost())
        for r in results
    ]
    lines.append(_row("Costo total", [f"${t:.4f}" for t in totals]))
    lines.append("")

    # Muestras lado a lado cada ~2 minutos
    lines.append("## Comparacion lado a lado")
    lines.append("")

    sample_interval = 120  # 2 minutos
    max_samples = _BENCHMARK_DURATION // sample_interval
    for i in range(max_samples):
        target = float(i * sample_interval)
        nearest = [_find_entry_near(r.entries, target) for r in results]

        if all(e is None for e in nearest):
            continue

        mm, ss = divmod(int(target), 60)
        lines.append(f"### ~{mm:02d}:{ss:02d}")
        lines.append("")
        lines.append(f"| | {header} |")
        lines.append(f"|---|{separator}|")

        texts = [
            _escape_md_table(e.text) if e else "\u2014" for e in nearest
        ]
        translations = [
            _escape_md_table(e.translation) if e else "\u2014" for e in nearest
        ]

        lines.append(f"| {config.source_lang.upper()} | {' | '.join(texts)} |")
        lines.append(f"| {config.target_lang.upper()} | {' | '.join(translations)} |")
        lines.append("")

    # Costos totales
    lines.append("## Costos")
    lines.append("")
    for r, total in zip(results, totals, strict=True):
        lines.append(f"- **{r.model}:** ${total:.4f}")
    lines.append(f"- **Total benchmark:** ${sum(totals):.4f}")
    lines.append("")

    return "\n".join(lines)
