"""Benchmark: compara modelos de transcripcion sobre el mismo audio."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from openai import AsyncOpenAI

from autosubs.config import LOGS_DIR, YT_ID_RE, Config
from autosubs.costs import CostTracker, escape_md_table
from autosubs.stream import download_audio, set_cookies_browser
from autosubs.transcribe import transcribe_audio, transcribe_verbose
from autosubs.translate import translate_text
from autosubs.vad import SAMPLE_RATE, AudioChunk, VadSegmenter

logger = logging.getLogger(__name__)

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
    if m := YT_ID_RE.search(config.url):
        video_id = m.group(1)

    cp_path = _checkpoint_path(video_id, config.start_time)
    cached = _load_checkpoint(cp_path)

    # Si hay cache valido, intentar resumir
    if cached:
        segment_file, results = cached
        models_done = {r.model for r in results}

        # Modelos que vamos a correr esta sesion (segun disponibilidad)
        expected = {"gpt-4o-transcribe"}
        if _faster_whisper_available():
            expected |= {"faster-whisper-turbo", "faster-whisper-large-v3"}
        if _cohere_asr_available():
            expected |= {"cohere-transcribe-03-2026"}
        if _gemini_available():
            expected |= {"gemini-2.5-flash", "gemini-2.5-pro"}

        missing = expected - models_done
        has_real_entries = any(r.entries for r in results)

        if config.benchmark_no_translate:
            done = not missing and has_real_entries
        else:
            all_translated = all(not _result_needs_translation(r) for r in results)
            done = not missing and all_translated and has_real_entries

        if done:
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

    tmp_dir = Path(tempfile.mkdtemp(prefix="autosubs_bench_"))
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

        # Cohere Transcribe (open-weight, HF Transformers, GPU local)
        if _cohere_asr_available() and "cohere-transcribe-03-2026" not in models_done:
            logger.info(
                "Transcribiendo cohere-transcribe-03-2026 (%ds desde %ds)",
                _BENCHMARK_DURATION, config.start_time,
            )
            result_cohere = await _transcribe_cohere_subprocess(
                segment_file, config, model_name="cohere-transcribe-03-2026",
            )
            if result_cohere.entries:
                results.append(result_cohere)
                _save_checkpoint(cp_path, results, segment_file)
            else:
                logger.warning("cohere fallo, no guardado en checkpoint (se reintentara)")

        # Gemini 2.5 Flash y Pro (audio multimodal nativo)
        if _gemini_available():
            for gemini_model in ("gemini-2.5-flash", "gemini-2.5-pro"):
                if gemini_model not in models_done:
                    logger.info(
                        "Transcribiendo %s (%ds desde %ds)",
                        gemini_model, _BENCHMARK_DURATION, config.start_time,
                    )
                    result_g = await _transcribe_gemini(
                        segment_file, config, model_name=gemini_model,
                    )
                    if result_g.entries:
                        results.append(result_g)
                        _save_checkpoint(cp_path, results, segment_file)
                    else:
                        logger.warning(
                            "%s fallo, no guardado en checkpoint", gemini_model,
                        )
        elif not os.environ.get("GEMINI_API_KEY"):
            logger.info("GEMINI_API_KEY no configurada, saltando Gemini")
        else:
            logger.warning(
                "google-genai no instalado (instalar con: uv sync --extra gemini)",
            )

        # Fallback Path B (solo si no hay faster-whisper)
        if not has_local and "whisper-1" not in models_done:
            result_b = await _transcribe_path_b(client, segment_file, config)
            results.append(result_b)
            _save_checkpoint(cp_path, results, segment_file)

        logger.info(
            "Transcripcion completa: %s",
            ", ".join(f"{r.model}={len(r.entries)} entries" for r in results),
        )

        # 4. Traducir (solo los que faltan, salvo --benchmark-no-translate)
        if config.benchmark_no_translate:
            logger.info("Saltando traduccion (--benchmark-no-translate)")
        else:
            for result in results:
                if _result_needs_translation(result):
                    logger.info(
                        "Traduciendo %s (%d entries)...",
                        result.model, len(result.entries),
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
            cost = result.tracker.whisper_cost() + sum(result.tracker.translation_cost())
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


# -- Benchmark de traductores --

# (model_id, provider) — provider determina base_url + api key
_TRANSLATORS_TO_TEST: list[tuple[str, str]] = [
    ("gpt-4.1", "openai"),
    ("deepseek/deepseek-chat", "openrouter"),  # V3, mas estable que V4 Flash en rate limits
    ("deepseek/deepseek-v4-pro", "openrouter"),
    ("moonshotai/kimi-k2", "openrouter"),
    ("anthropic/claude-haiku-4.5", "openrouter"),
]

_PROVIDER_CONFIG = {
    "openai": {
        "base_url": None,  # default OpenAI
        "api_key_env": "OPENAI_API_KEY",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
    },
}


def _make_client(provider: str) -> AsyncOpenAI | None:
    """Crea cliente OpenAI-compatible para el provider especificado."""
    cfg = _PROVIDER_CONFIG.get(provider)
    if cfg is None:
        return None
    api_key = os.environ.get(cfg["api_key_env"])
    if not api_key:
        return None
    # Timeout 30s por request para detectar problemas rapido en benchmark
    if cfg["base_url"]:
        return AsyncOpenAI(api_key=api_key, base_url=cfg["base_url"], timeout=30.0)
    return AsyncOpenAI(api_key=api_key, timeout=30.0)


async def run_e2e_benchmark(config: Config) -> None:
    """Test E2E: Gemini 2.5 Pro hace ASR + traduccion en una sola call."""
    set_cookies_browser(config.cookies_browser)

    if not _gemini_available():
        logger.error("GEMINI_API_KEY no configurada o google-genai no instalado")
        return

    video_id = ""
    if m := YT_ID_RE.search(config.url):
        video_id = m.group(1)

    # Reusar audio cacheado del benchmark de transcripcion
    segment_file = LOGS_DIR / f"segment_{video_id}_{config.start_time}.mp3"
    if not segment_file.exists():
        logger.error(
            "Audio cacheado no encontrado: %s. Corre primero --benchmark.",
            segment_file,
        )
        return

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    logger.info("E2E test: Gemini 2.5 Pro audio -> %s", config.target_lang)

    t0 = time.perf_counter()
    myfile = await asyncio.to_thread(client.files.upload, file=str(segment_file))

    prompt = _GEMINI_E2E_PROMPT.format(
        source_lang=config.source_lang, target_lang=config.target_lang,
    )
    response = await asyncio.to_thread(
        client.models.generate_content,
        model="gemini-2.5-pro",
        contents=[prompt, myfile],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema={
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "start": {"type": "NUMBER"},
                        "end": {"type": "NUMBER"},
                        "text": {"type": "STRING"},
                    },
                    "required": ["start", "end", "text"],
                },
            },
        ),
    )
    elapsed = time.perf_counter() - t0

    try:
        await asyncio.to_thread(client.files.delete, name=myfile.name)
    except Exception as e:
        logger.debug("No se pudo borrar file remoto: %s", e)

    try:
        segments = json.loads(response.text)
    except (json.JSONDecodeError, AttributeError) as e:
        logger.error("Error parseando respuesta E2E: %s", e)
        return

    # Tracking de tokens (vienen en response.usage_metadata)
    usage = getattr(response, "usage_metadata", None)
    audio_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
    output_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0

    # Pricing Gemini 2.5 Pro: audio $1/M, output $10/M (≤200k context)
    audio_cost = audio_tokens * 1.00 / 1_000_000
    output_cost = output_tokens * 10.00 / 1_000_000
    total_cost = audio_cost + output_cost

    logger.info(
        "E2E completado: %d segments, %.1fs, %d audio tokens + %d output tokens, $%.4f",
        len(segments), elapsed, audio_tokens, output_tokens, total_cost,
    )

    # Generar reporte
    md = _generate_e2e_report(
        segments, config, video_id, datetime.now(),
        elapsed, audio_tokens, output_tokens, total_cost,
    )
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    output_path = LOGS_DIR / f"benchmark_e2e_{video_id}_{now:%Y-%m-%d_%H-%M-%S}.md"
    output_path.write_text(md, encoding="utf-8")
    logger.info("Reporte E2E guardado en: %s", output_path)


def _generate_e2e_report(
    segments: list[dict],
    config: Config,
    video_id: str,
    now: datetime,
    elapsed: float,
    audio_tokens: int,
    output_tokens: int,
    total_cost: float,
) -> str:
    """Markdown del benchmark E2E."""
    end_time = config.start_time + _BENCHMARK_DURATION
    mm_start, ss_start = divmod(config.start_time, 60)
    mm_end, ss_end = divmod(end_time, 60)

    # Extrapolar a 1 hora y 2 horas
    factor_1h = 3600 / _BENCHMARK_DURATION
    factor_2h = 7200 / _BENCHMARK_DURATION

    lines = [
        f"# Benchmark E2E (Gemini 2.5 Pro audio→ES) — {video_id}",
        "",
        f"**Fecha:** {now:%Y-%m-%d %H:%M} | "
        f"**Audio:** {mm_start:02d}:{ss_start:02d}-{mm_end:02d}:{ss_end:02d} | "
        f"**Modelo:** gemini-2.5-pro | "
        f"**Pipeline:** audio→{config.target_lang} (single call)",
        "",
        "## Metricas",
        "",
        f"- **Segments generados**: {len(segments)}",
        f"- **Tiempo total**: {elapsed:.1f}s",
        f"- **Audio tokens**: {audio_tokens:,}",
        f"- **Output tokens**: {output_tokens:,}",
        f"- **Costo total**: ${total_cost:.4f}",
        f"- **Costo extrapolado 1h**: ${total_cost * factor_1h:.4f}",
        f"- **Costo extrapolado 2h**: ${total_cost * factor_2h:.4f}",
        "",
        "## Comparacion vs Setup A (Pro ASR + Haiku traduccion)",
        "",
        "| Setup | $/2h | Calls | Latencia | Confiabilidad |",
        "|-------|------|-------|----------|---------------|",
        f"| **E2E (este)** | ${total_cost * factor_2h:.2f} | 1 | {elapsed:.0f}s | TBD |",
        "| Setup A (Pro+Haiku) | $0.43 | 2 | ~5min* | Confirmada |",
        "",
        "*Setup A: 81s ASR + 220s traduccion = 5min, pero permite parallelism real-time",
        "",
        "## Muestra de segments",
        "",
        "| Tiempo | ES |",
        "|--------|----|",
    ]

    # Mostrar primeros 30 segments
    for seg in segments[:30]:
        start = seg.get("start", 0)
        mm, ss = divmod(int(start), 60)
        # Soporta tanto schema viejo (ja/es) como nuevo (text)
        text = seg.get("text") or seg.get("es") or ""
        lines.append(f"| {mm:02d}:{ss:02d} | {escape_md_table(text)} |")

    lines.extend([
        "",
        f"*({len(segments)} segments totales, mostrando primeros 30)*",
        "",
        "## Spot-checks vs Setup A (Haiku traduccion)",
        "",
        "Buscar en estos timestamps clave si E2E mantiene calidad:",
        "- ~05:59 'la palabra abdominales' (meta-referencia)",
        "- ~07:56 PRIMERA PERSONA test (mi peso vs su peso)",
        "- ~10:03 'gym' vs 'Jim' test cultural",
        "",
        "Comparar manualmente con `benchmark_translators_*.md` previo.",
    ])

    return "\n".join(lines)


def _translator_cp_path(video_id: str, start_time: int) -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    return LOGS_DIR / f"benchmark_translators_cache_{video_id}_{start_time}.json"


def _save_translator_checkpoint(
    path: Path,
    results: list[tuple[str, list[str], CostTracker, float]],
) -> None:
    data = {
        "results": [
            {
                "model": m,
                "translations": tr,
                "input_tokens": t.translation_prompt_tokens,
                "output_tokens": t.translation_completion_tokens,
                "elapsed_seconds": e,
            }
            for m, tr, t, e in results
        ]
    }
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    logger.info("Translator checkpoint guardado: %s (%d traductores)", path.name, len(results))


def _load_translator_checkpoint(
    path: Path, source_lang: str, target_lang: str,
) -> list[tuple[str, list[str], CostTracker, float]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("No se pudo cargar translator checkpoint: %s", e)
        return []
    results = []
    for r in data.get("results", []):
        tracker = CostTracker(
            translation_model=r["model"],
            source_lang=source_lang,
            target_lang=target_lang,
            translation_prompt_tokens=r["input_tokens"],
            translation_completion_tokens=r["output_tokens"],
        )
        results.append((r["model"], r["translations"], tracker, r["elapsed_seconds"]))
    return results


async def run_translator_benchmark(config: Config) -> None:
    """Compara N traductores sobre la misma transcripcion cacheada (gpt-4o)."""
    set_cookies_browser(config.cookies_browser)

    video_id = ""
    if m := YT_ID_RE.search(config.url):
        video_id = m.group(1)

    cp_path = _checkpoint_path(video_id, config.start_time)
    cached = _load_checkpoint(cp_path)
    if cached is None:
        logger.error(
            "No hay checkpoint en %s. Corre primero --benchmark normal.", cp_path,
        )
        return
    _, transcribe_results = cached

    # Buscar gpt-4o-transcribe como referencia (mejor calidad confirmada)
    reference: BenchmarkResult | None = next(
        (r for r in transcribe_results if r.model == "gpt-4o-transcribe"),
        None,
    )
    if reference is None or not reference.entries:
        logger.error("No hay transcripcion de gpt-4o-transcribe en cache")
        return

    logger.info(
        "Usando %d entries de gpt-4o-transcribe como referencia",
        len(reference.entries),
    )

    # Cargar checkpoint de traductores ya completados
    tcp_path = _translator_cp_path(video_id, config.start_time)
    translator_results = _load_translator_checkpoint(
        tcp_path, config.source_lang, config.target_lang,
    )
    models_done = {m for m, _, _, _ in translator_results}
    if models_done:
        logger.info("Traductores ya cacheados: %s", ", ".join(models_done))

    # Filtrar traductores con API key disponible y NO cacheados
    available: list[tuple[str, str, AsyncOpenAI]] = []
    for model_id, provider in _TRANSLATORS_TO_TEST:
        if model_id in models_done:
            continue
        client = _make_client(provider)
        if client is None:
            logger.warning(
                "Sin API key para %s, saltando %s",
                _PROVIDER_CONFIG[provider]["api_key_env"], model_id,
            )
            continue
        available.append((model_id, provider, client))

    if not available and not translator_results:
        logger.error("Ningun traductor disponible. Configura OPENROUTER_API_KEY.")
        return

    if available:
        logger.info(
            "Traductores a correr: %s",
            ", ".join(m for m, _, _ in available),
        )

    # Traducir con cada modelo (smoke test primero)
    for model_id, _provider, client in available:
        tracker = CostTracker(
            translation_model=model_id,
            source_lang=config.source_lang,
            target_lang=config.target_lang,
        )
        # Smoke test: traducir primera entry para detectar problemas
        logger.info("Smoke test %s...", model_id)
        smoke_t0 = time.perf_counter()
        smoke_text = reference.entries[0].text if reference.entries else "test"
        try:
            smoke_result = await translate_text(
                client, smoke_text, config.source_lang, config.target_lang,
                model_id, tracker=tracker,
            )
        except Exception as e:
            logger.error("Smoke test %s fallo: %s", model_id, e)
            continue
        smoke_elapsed = time.perf_counter() - smoke_t0
        if smoke_result is None:
            logger.error("%s no devolvio traduccion en smoke test, saltando", model_id)
            continue
        logger.info(
            "Smoke test OK: '%s' -> '%s' (%.1fs)",
            smoke_text[:30], smoke_result[:30], smoke_elapsed,
        )

        logger.info(
            "Traduciendo con %s (%d entries)...", model_id, len(reference.entries),
        )
        t0 = time.perf_counter()
        translations: list[str] = [smoke_result]  # primera ya traducida en smoke
        for i, entry in enumerate(reference.entries[1:], start=1):
            translated = await translate_text(
                client, entry.text, config.source_lang, config.target_lang,
                model_id, tracker=tracker,
            )
            translations.append(translated or "")
            if (i + 1) % 30 == 0:
                logger.info("  %s: %d/%d", model_id, i + 1, len(reference.entries))
        elapsed = time.perf_counter() - t0
        translator_results.append((model_id, translations, tracker, elapsed))
        _save_translator_checkpoint(tcp_path, translator_results)
        cost = sum(tracker.translation_cost())
        logger.info(
            "%s completado: %.1fs, $%.4f", model_id, elapsed, cost,
        )

    # Generar reporte
    md = _generate_translator_comparison(
        reference, translator_results, config, video_id, datetime.now(),
    )
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    output_path = LOGS_DIR / f"benchmark_translators_{video_id}_{now:%Y-%m-%d_%H-%M-%S}.md"
    output_path.write_text(md, encoding="utf-8")
    logger.info("Benchmark guardado en: %s", output_path)

    # Resumen costos
    total = 0.0
    for model_id, _, tracker, _ in translator_results:
        cost = sum(tracker.translation_cost())
        total += cost
        logger.info("  %s: $%.4f", model_id, cost)
    logger.info("Total benchmark traductores: $%.4f", total)


def _generate_translator_comparison(
    reference: BenchmarkResult,
    translator_results: list[tuple[str, list[str], CostTracker, float]],
    config: Config,
    video_id: str,
    now: datetime,
) -> str:
    """Markdown comparando N traductores sobre la misma transcripcion."""
    end_time = config.start_time + _BENCHMARK_DURATION
    mm_start, ss_start = divmod(config.start_time, 60)
    mm_end, ss_end = divmod(end_time, 60)

    models = [m for m, _, _, _ in translator_results]
    header = " | ".join(models)
    sep = " | ".join("---" for _ in models)

    lines: list[str] = [
        f"# Benchmark Traductores — {video_id}",
        "",
        f"**Fecha:** {now:%Y-%m-%d %H:%M} | "
        f"**Audio:** {mm_start:02d}:{ss_start:02d}-{mm_end:02d}:{ss_end:02d} | "
        f"**Idioma:** {config.source_lang} -> {config.target_lang}",
        "",
        f"**Transcripcion source:** gpt-4o-transcribe ({len(reference.entries)} entries)",
        "",
        "## Metricas",
        "",
        f"| Metrica | {header} |",
        f"|---------|{sep}|",
    ]

    def _row(label: str, vals: list[str]) -> str:
        return f"| {label} | {' | '.join(vals)} |"

    lines.append(_row(
        "Tiempo total",
        [f"{e:.1f}s" for _, _, _, e in translator_results],
    ))
    lines.append(_row(
        "Tokens input",
        [f"{t.translation_prompt_tokens:,}" for _, _, t, _ in translator_results],
    ))
    lines.append(_row(
        "Tokens output",
        [f"{t.translation_completion_tokens:,}" for _, _, t, _ in translator_results],
    ))
    lines.append(_row(
        "Costo total",
        [f"${sum(t.translation_cost()):.4f}" for _, _, t, _ in translator_results],
    ))
    # Extrapolar a 2h de stream (seg actual = 1200s, 2h = 7200s, factor 6)
    lines.append(_row(
        "Costo extrapolado 2h",
        [f"${sum(t.translation_cost()) * 6:.4f}" for _, _, t, _ in translator_results],
    ))
    lines.append("")

    # Sample side-by-side cada 2 minutos
    lines.append("## Comparacion lado a lado")
    lines.append("")

    sample_interval = 120
    max_samples = _BENCHMARK_DURATION // sample_interval
    for i in range(max_samples):
        target = float(i * sample_interval)
        # Find entry idx closest to target
        best_idx = -1
        best_diff = float("inf")
        for j, e in enumerate(reference.entries):
            diff = abs(e.start - target)
            if diff < best_diff:
                best_diff = diff
                best_idx = j
        if best_idx < 0:
            continue

        entry = reference.entries[best_idx]
        mm, ss = divmod(int(entry.start), 60)
        lines.append(f"### ~{mm:02d}:{ss:02d}")
        lines.append("")
        lines.append(f"**JA:** {escape_md_table(entry.text)}")
        lines.append("")
        lines.append("| Traductor | ES |")
        lines.append("|---|---|")
        for model_id, translations, _, _ in translator_results:
            tr = translations[best_idx] if best_idx < len(translations) else ""
            lines.append(f"| {model_id} | {escape_md_table(tr)} |")
        lines.append("")

    return "\n".join(lines)


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

    # ffmpeg -> PCM 16kHz mono con EQ de transcripcion
    from autosubs.audio import TRANSCRIPTION_AUDIO_FILTER

    cmd = [
        "ffmpeg", "-loglevel", "warning",
        "-i", audio_path,
        "-af", TRANSCRIPTION_AUDIO_FILTER,
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
    return importlib.util.find_spec("faster_whisper") is not None


def _gemini_available() -> bool:
    """Chequea si google-genai esta instalado y hay API key."""
    if importlib.util.find_spec("google.genai") is None:
        return False
    return bool(os.environ.get("GEMINI_API_KEY"))


def _cohere_asr_available() -> bool:
    """Chequea si transformers + torch CUDA estan disponibles."""
    if importlib.util.find_spec("transformers") is None:
        return False
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


_GEMINI_PROMPT = (
    "Transcribe este audio en {lang}. "
    "Devuelve SOLO un array JSON de segmentos con timestamps en segundos. "
    'Formato: [{{"start": 0.0, "end": 5.5, "text": "..."}}]. '
    "No agregues explicaciones, comentarios ni codigo markdown. "
    "Solo el JSON."
)

_GEMINI_E2E_PROMPT = (
    "Escucha este audio en {source_lang} y traducelo directamente a {target_lang}. "
    "Devuelve SOLO un array JSON con segmentos de subtitulos. "
    'Formato: [{{"start": 0.0, "end": 5.5, "text": "traduccion en {target_lang}"}}]. '
    "Para subtitulos naturales: cuando el hablante hable de si mismo en primera "
    "persona (comun en japones por la elision del sujeto), usa primera persona "
    "en la traduccion (mi, yo, etc), no tercera persona (su, el/ella). "
    "No agregues explicaciones ni codigo markdown. Solo el JSON."
)


async def _transcribe_gemini(
    audio_path: str, config: Config, model_name: str = "gemini-2.5-flash",
) -> BenchmarkResult:
    """Transcribe con Gemini multimodal (audio nativo)."""
    from google import genai
    from google.genai import types

    tracker = CostTracker(
        whisper_model=model_name,
        translation_model=config.gpt_model,
        source_lang=config.source_lang,
        target_lang=config.target_lang,
    )
    result = BenchmarkResult(model=model_name, tracker=tracker)

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    t0 = time.perf_counter()
    try:
        myfile = await asyncio.to_thread(client.files.upload, file=audio_path)

        prompt = _GEMINI_PROMPT.format(lang=config.source_lang)
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=model_name,
            contents=[prompt, myfile],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema={
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "start": {"type": "NUMBER"},
                            "end": {"type": "NUMBER"},
                            "text": {"type": "STRING"},
                        },
                        "required": ["start", "end", "text"],
                    },
                },
            ),
        )

        # Cleanup remoto
        try:
            await asyncio.to_thread(client.files.delete, name=myfile.name)
        except Exception as e:
            logger.debug("No se pudo borrar file remoto de Gemini: %s", e)

        result.transcribe_seconds = time.perf_counter() - t0
        tracker.add_whisper(_BENCHMARK_DURATION)

        segments = json.loads(response.text)
        for seg in segments:
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            result.entries.append(BenchmarkEntry(
                start=float(seg["start"]),
                end=float(seg["end"]),
                text=text,
                translation="",
            ))

        logger.info(
            "%s transcrito: %d entries (%.1fs)",
            model_name, len(result.entries), result.transcribe_seconds,
        )
    except Exception as e:
        logger.error("%s fallo: %s", model_name, e)

    return result


async def _transcribe_cohere_subprocess(
    audio_path: str,
    config: Config,
    model_name: str = "cohere-transcribe-03-2026",
) -> BenchmarkResult:
    """Transcribe con Cohere Transcribe (HF Transformers) en subprocess aislado."""
    import sys

    tracker = CostTracker(
        whisper_model=model_name,
        translation_model=config.gpt_model,
        source_lang=config.source_lang,
        target_lang=config.target_lang,
    )
    result = BenchmarkResult(model=model_name, tracker=tracker)

    script = f"""
import json, os, re, sys, time
sys.path.insert(0, {str(Path(__file__).resolve().parent.parent)!r})
from dotenv import load_dotenv
load_dotenv()
import librosa
import torch
from transformers import AutoProcessor, CohereAsrForConditionalGeneration

token = os.environ.get("HF_TOKEN")
processor = AutoProcessor.from_pretrained(
    "CohereLabs/cohere-transcribe-03-2026", token=token,
)
model = CohereAsrForConditionalGeneration.from_pretrained(
    "CohereLabs/cohere-transcribe-03-2026",
    device_map="auto",
    torch_dtype=torch.float16,
    token=token,
)
model.eval()

# Cargar audio a 16kHz mono
audio, sr = librosa.load({audio_path!r}, sr=16000, mono=True)
chunk_seconds = 30
chunk_samples = chunk_seconds * sr

t0 = time.perf_counter()
texts = []
for i in range(0, len(audio), chunk_samples):
    chunk = audio[i:i + chunk_samples]
    if len(chunk) < sr:  # ignorar chunks <1s
        continue
    inputs = processor(
        chunk, sampling_rate=sr, return_tensors="pt",
        language={config.source_lang!r},
    )
    inputs = inputs.to(model.device, dtype=model.dtype)
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=256)
    text = processor.decode(outputs[0], skip_special_tokens=True).strip()
    if text:
        texts.append(text)
elapsed = time.perf_counter() - t0

full_text = " ".join(texts).strip()
duration = {_BENCHMARK_DURATION!r}

# Split por puntos japoneses (\\u3002 \\uff0e \\uff01 \\uff1f) y occidentales
parts = re.split(r"(?<=[\\u3002\\uff0e.!?\\uff01\\uff1f])\\s*", full_text)
sentences = [p.strip() for p in parts if p and p.strip()]

segments = []
if sentences:
    per = duration / len(sentences)
    for j, s in enumerate(sentences):
        segments.append({{
            "start": round(j * per, 2),
            "end": round((j + 1) * per, 2),
            "text": s,
        }})

sys.stdout.write(json.dumps({{"elapsed": elapsed, "segments": segments}}))
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
            for line in stderr.decode(errors="replace").splitlines()[-10:]:
                logger.error("%s stderr: %s", model_name, line)
        return result

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
from autosubs.local_whisper import LocalWhisperModel
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
        f"# Benchmark AutoSubs — {video_id}",
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
        [f"${r.tracker.whisper_cost():.4f}" for r in results],
    ))
    lines.append(_row(
        "Costo traduccion",
        [f"${sum(r.tracker.translation_cost()):.4f}" for r in results],
    ))

    totals = [
        r.tracker.whisper_cost() + sum(r.tracker.translation_cost())
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
            escape_md_table(e.text) if e else "\u2014" for e in nearest
        ]
        translations = [
            escape_md_table(e.translation) if e else "\u2014" for e in nearest
        ]

        lines.append(f"| {config.source_lang.upper()} | {' | '.join(texts)} |")
        if not config.benchmark_no_translate:
            lines.append(
                f"| {config.target_lang.upper()} | {' | '.join(translations)} |",
            )
        lines.append("")

    # Costos totales
    lines.append("## Costos")
    lines.append("")
    for r, total in zip(results, totals, strict=True):
        lines.append(f"- **{r.model}:** ${total:.4f}")
    lines.append(f"- **Total benchmark:** ${sum(totals):.4f}")
    lines.append("")

    return "\n".join(lines)
