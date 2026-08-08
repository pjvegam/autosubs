from __future__ import annotations

import asyncio
import collections
import contextlib
import json
import logging
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from openai import AsyncOpenAI

from autosubs.audio import AudioSegmenter
from autosubs.capture import SystemAudioCapture
from autosubs.config import LOGS_DIR, YT_ID_RE, Config
from autosubs.costs import CostTracker
from autosubs.overlay import SubtitleOverlay, run_overlay_loop
from autosubs.player import MPVPlayer
from autosubs.stream import (
    cut_video,
    download_audio,
    download_vod,
    get_stream_info,
    get_stream_urls,
    set_cookies_browser,
    split_audio,
)
from autosubs.subtitles import SubtitleManager, srt_last_end_time
from autosubs.transcribe import (
    TranscriptSegment,
    transcribe_audio,
    transcribe_gemini_chunk,
    transcribe_verbose,
)
from autosubs.translate import summarize_session, translate_text
from autosubs.vad import AudioChunk

if TYPE_CHECKING:
    from autosubs.diarize import DiarizationEngine
    from autosubs.local_whisper import LocalWhisperModel

logger = logging.getLogger(__name__)

MIN_SPEECH_RATIO = 0.15


def _video_download_complete(path: Path) -> bool:
    """True si el .mkv es una descarga terminada y no una a medias.

    yt-dlp mantiene un `<archivo>.ytdl` con el indice del fragmento en curso y
    lo borra al terminar. Sin ese chequeo, un .mkv parcial pasa por cache
    completo: la descarga no se reanuda nunca y el audio se lee de un archivo
    truncado, con lo que la transcripcion se corta a los pocos minutos.
    """
    if not path.exists():
        return False
    return not path.with_name(f"{path.name}.ytdl").exists()


async def _downloaded_seconds(path: Path) -> float:
    """Segundos de video ya disponibles en una descarga en curso.

    yt-dlp va concatenando fragmentos MPEG-TS, que ffprobe lee sin problema
    aunque el archivo siga creciendo. Retorna 0.0 si todavia no es legible.
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    stdout, _ = await proc.communicate()
    if proc.returncode != 0:
        return 0.0
    try:
        return float(stdout.decode().strip())
    except ValueError:
        return 0.0


async def _wait_until_downloaded(
    path: Path, target_seconds: float, download: asyncio.Task, *, margin: float = 30.0,
) -> bool:
    """Espera a que la descarga cubra target_seconds en vez del VOD entero.

    Saltar a un punto solo necesita el video hasta ese punto: esperar el archivo
    completo son horas para un VOD largo. Retorna False si la descarga muere
    antes de llegar.
    """
    needed = target_seconds + margin
    while True:
        if (available := await _downloaded_seconds(path)) >= needed:
            logger.info("Descargado hasta %.0fs, suficiente para saltar", available)
            return True
        if download.done():
            # Termino la descarga: o esta completo o fallo
            return await _downloaded_seconds(path) >= target_seconds
        logger.info(
            "Esperando descarga: %.0fs de %.0fs necesarios...", available, needed,
        )
        await asyncio.sleep(5)

# Circuit breaker para Gemini: si falla N veces consecutivas, lo "sacamos de
# juego" por X minutos y todas las requests van directo a gpt-4o-transcribe.
_GEMINI_FAIL_THRESHOLD = 5
_GEMINI_COOLDOWN_SECONDS = 300  # 5 min


@dataclass
class _GeminiCircuit:
    consecutive_failures: int = 0
    cooldown_until: float = 0.0  # epoch seconds

    def is_open(self) -> bool:
        """True si Gemini esta 'fuera de juego' temporalmente."""
        import time
        return time.time() < self.cooldown_until

    def record_success(self) -> None:
        self.consecutive_failures = 0

    def record_failure(self) -> None:
        import time
        self.consecutive_failures += 1
        if self.consecutive_failures >= _GEMINI_FAIL_THRESHOLD:
            self.cooldown_until = time.time() + _GEMINI_COOLDOWN_SECONDS
            self.consecutive_failures = 0  # reset al activar cooldown
            mins = _GEMINI_COOLDOWN_SECONDS // 60
            logger.warning(
                "Gemini desactivado por %d min tras %d fallos consecutivos. "
                "Usando gpt-4o-transcribe.",
                mins, _GEMINI_FAIL_THRESHOLD,
            )


def _get_local_model_size(whisper_model: str) -> str | None:
    """Mapea whisper_model CLI a model_size de faster-whisper, o None si es API."""
    match whisper_model:
        case "local-turbo":
            return "turbo"
        case "local-v3":
            return "large-v3"
        case _:
            return None


def _is_gemini_asr(whisper_model: str) -> bool:
    """True si el modelo es Gemini (audio multimodal)."""
    return whisper_model.startswith("gemini-")


def _create_translate_client(gpt_model: str, openai_api_key: str) -> AsyncOpenAI:
    """Crea cliente OpenAI-compatible segun el modelo.

    - Modelos con "/" (ej: anthropic/claude-haiku-4.5) → OpenRouter
    - Resto → OpenAI directo
    """
    import os
    if "/" in gpt_model:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                f"OPENROUTER_API_KEY no configurada para modelo '{gpt_model}'. "
                "Agregala al .env o usa un modelo OpenAI (ej: gpt-4.1)",
            )
        return AsyncOpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    return AsyncOpenAI(api_key=openai_api_key)


async def _load_local_whisper(model_size: str) -> LocalWhisperModel:
    """Carga modelo faster-whisper en GPU (blocking → to_thread)."""
    from autosubs.local_whisper import LocalWhisperModel

    return await asyncio.to_thread(LocalWhisperModel, model_size)


# El titulo traducido va en el system prompt de CADA subtitulo. Una traduccion
# degenerada (p.ej. '!' repetido 100 veces) contaminaria ese contexto y dispara
# el consumo de tokens durante todo el stream. Por eso se sanitiza y se trunca.
_MAX_TITLE_TRANSLATION = 200


def _sanitize_title(text: str) -> str:
    """Colapsa runs de 4+ repeticiones y trunca a un largo razonable."""
    collapsed = re.sub(r"(.)\1{3,}", r"\1\1\1", text).strip()
    return collapsed[:_MAX_TITLE_TRANSLATION].rstrip()


def _load_cached_title(video_id: str, original: str, target_lang: str) -> str | None:
    """Lee la traduccion del titulo cacheada si coincide original e idioma."""
    if not video_id:
        return None
    path = LOGS_DIR / f"{video_id}.title.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if data.get("original") == original and data.get("target") == target_lang:
        return data.get("translated") or None
    return None


def _save_cached_title(
    video_id: str, original: str, target_lang: str, translated: str,
) -> None:
    """Persiste la traduccion del titulo para no re-traducir en re-ejecuciones."""
    if not video_id:
        return
    payload = {"original": original, "target": target_lang, "translated": translated}
    with contextlib.suppress(OSError):
        (LOGS_DIR / f"{video_id}.title.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8",
        )


async def _get_stream_context(
    url: str, client: AsyncOpenAI, config: Config,
    tracker: CostTracker | None = None,
) -> tuple[bool, str, str]:
    """Extrae is_live, contexto de metadata y video_id en una sola llamada a yt-dlp.

    Traduce el titulo para dar contexto util a GPT.
    """
    try:
        info = await get_stream_info(url)
        live = info.get("is_live", False)
        video_id = info.get("id", "")
        title = info.get("title", "")
        channel = info.get("channel", "") or info.get("uploader", "")

        if title:
            use_cache = not config.no_cache
            translated_title = (
                _load_cached_title(video_id, title, config.target_lang)
                if use_cache else None
            )
            if translated_title is None:
                raw = await translate_text(
                    client, title, config.source_lang, config.target_lang,
                    config.gpt_model, tracker=tracker,
                )
                if raw:
                    translated_title = _sanitize_title(raw)
                    if use_cache:
                        _save_cached_title(
                            video_id, title, config.target_lang, translated_title,
                        )
            if translated_title:
                title = f"{title} ({translated_title})"

        parts: list[str] = []
        if channel:
            parts.append(f"Channel: {channel}")
        if title:
            parts.append(f"Title: {title}")
        if config.speakers > 1:
            parts.append(
                f"There are {config.speakers} speakers in this conversation. "
                "Different speakers may have different speech styles."
            )
        if config.speaker_gender:
            gender_label = "female" if config.speaker_gender == "f" else "male"
            parts.append(
                f"The main speaker is {gender_label}. "
                f"Use {gender_label} grammatical gender consistently."
            )
        context = "\n".join(parts)
        return live, context, video_id
    except Exception:
        logger.warning("No se pudo obtener metadata del stream, asumiendo VOD sin contexto")
        return False, "", ""


def _should_skip_chunk(chunk: AudioChunk) -> bool:
    """Descarta chunks con poca voz para evitar alucinaciones de Whisper."""
    if chunk.speech_ratio < MIN_SPEECH_RATIO:
        logger.debug(
            "Saltando %s: speech_ratio=%.0f%% < %.0f%%",
            chunk.path.name, chunk.speech_ratio * 100, MIN_SPEECH_RATIO * 100,
        )
        return True
    return False


async def _transcribe_worker(
    client: AsyncOpenAI,
    config: Config,
    audio_queue: asyncio.Queue,
    text_queue: asyncio.Queue,
    tracker: CostTracker | None = None,
    start_time_offset: float = 0.0,
    max_seconds: float = 0.0,
    local_model: LocalWhisperModel | None = None,
) -> None:
    """Lee chunks de audio, los transcribe secuencialmente (para lives)."""
    elapsed = start_time_offset
    last_text: str | None = None
    repeat_count = 0
    gemini_circuit = _GeminiCircuit()
    while True:
        chunk: AudioChunk | None = await audio_queue.get()
        if chunk is None:
            await text_queue.put(None)
            break

        start_time = elapsed
        elapsed += chunk.duration

        if max_seconds > 0 and elapsed >= max_seconds:
            mm, _ = divmod(int(max_seconds), 60)
            logger.info("Limite de %d minutos alcanzado, deteniendo traduccion", mm)
            with contextlib.suppress(OSError):
                chunk.path.unlink()
            await text_queue.put(None)
            break

        if _should_skip_chunk(chunk):
            with contextlib.suppress(OSError):
                chunk.path.unlink()
            continue

        try:
            if local_model is not None:
                segments = await asyncio.to_thread(
                    local_model.transcribe, chunk.path, config.source_lang,
                )
                text = " ".join(seg.text for seg in segments) if segments else None
            elif _is_gemini_asr(config.whisper_model):
                if gemini_circuit.is_open():
                    # Circuit breaker activo: skip Gemini, ir directo a gpt-4o
                    text = await transcribe_audio(
                        client, chunk.path, config.source_lang, "gpt-4o-transcribe",
                    )
                else:
                    text = await transcribe_gemini_chunk(
                        chunk.path, config.source_lang, config.whisper_model,
                    )
                    if text is None:
                        gemini_circuit.record_failure()
                        logger.warning(
                            "Gemini fallo en %s, fallback a gpt-4o-transcribe",
                            chunk.path.name,
                        )
                        text = await transcribe_audio(
                            client, chunk.path, config.source_lang,
                            "gpt-4o-transcribe",
                        )
                    else:
                        gemini_circuit.record_success()
            else:
                text = await transcribe_audio(
                    client, chunk.path, config.source_lang, config.whisper_model,
                )
        except Exception as e:
            logger.error("Error inesperado transcribiendo %s: %s", chunk.path.name, e)
            text = None

        with contextlib.suppress(OSError):
            chunk.path.unlink()

        if text:
            if text == last_text:
                repeat_count += 1
                logger.debug("Descartado repetido #%d: '%s'", repeat_count, text[:40])
                continue
            last_text = text
            repeat_count = 0
            if tracker:
                tracker.add_whisper(chunk.duration)
            await text_queue.put((start_time, chunk.duration, text))


async def _transcribe_worker_parallel(
    client: AsyncOpenAI,
    config: Config,
    audio_queue: asyncio.Queue,
    text_queue: asyncio.Queue,
    max_concurrent: int = 3,
    tracker: CostTracker | None = None,
    start_time_offset: float = 0.0,
    max_seconds: float = 0.0,
    local_model: LocalWhisperModel | None = None,
) -> None:
    """Transcribe multiples chunks en paralelo, mantiene orden (para VODs)."""
    semaphore = asyncio.Semaphore(max_concurrent)
    task_queue: asyncio.Queue[tuple[asyncio.Task[str | None], float] | None] = asyncio.Queue()
    gemini_circuit = _GeminiCircuit()

    async def _do_transcribe(audio_path: Path) -> str | None:
        async with semaphore:
            try:
                if _is_gemini_asr(config.whisper_model):
                    if gemini_circuit.is_open():
                        return await transcribe_audio(
                            client, audio_path, config.source_lang,
                            "gpt-4o-transcribe",
                        )
                    text = await transcribe_gemini_chunk(
                        audio_path, config.source_lang, config.whisper_model,
                    )
                    if text is None:
                        gemini_circuit.record_failure()
                        logger.warning(
                            "Gemini fallo en %s, fallback a gpt-4o-transcribe",
                            audio_path.name,
                        )
                        text = await transcribe_audio(
                            client, audio_path, config.source_lang,
                            "gpt-4o-transcribe",
                        )
                    else:
                        gemini_circuit.record_success()
                    return text
                return await transcribe_audio(
                    client, audio_path, config.source_lang, config.whisper_model,
                )
            except Exception as e:
                logger.error("Error transcribiendo %s: %s", audio_path.name, e)
                return None
            finally:
                with contextlib.suppress(OSError):
                    audio_path.unlink()

    async def _receiver() -> None:
        while True:
            chunk: AudioChunk | None = await audio_queue.get()
            if chunk is None:
                await task_queue.put(None)
                break
            if _should_skip_chunk(chunk):
                with contextlib.suppress(OSError):
                    chunk.path.unlink()
                # Poner None como task para mantener el tracking de elapsed
                await task_queue.put((None, chunk.duration))
                continue
            task = asyncio.create_task(_do_transcribe(chunk.path))
            await task_queue.put((task, chunk.duration))

    receiver = asyncio.create_task(_receiver())

    elapsed = start_time_offset
    limit_reached = False
    while True:
        item = await task_queue.get()
        if item is None:
            break
        task, duration = item
        if task is not None:
            text = await task
            if text:
                if tracker:
                    tracker.add_whisper(duration)
                await text_queue.put((elapsed, duration, text))
        elapsed += duration
        if max_seconds > 0 and elapsed >= max_seconds:
            mm, _ = divmod(int(max_seconds), 60)
            logger.info("Limite de %d minutos alcanzado, deteniendo traduccion", mm)
            limit_reached = True
            break

    if not limit_reached:
        await receiver
    await text_queue.put(None)


async def _translate_worker(
    client: AsyncOpenAI,
    config: Config,
    text_queue: asyncio.Queue,
    translated_queue: asyncio.Queue,
    stream_context: str = "",
    diarizer: DiarizationEngine | None = None,
    tracker: CostTracker | None = None,
) -> None:
    """Lee texto transcrito, lo traduce y pone en la queue."""
    history: collections.deque[str] = collections.deque(maxlen=3)
    session_summary: str = ""
    summary_buffer: list[str] = []

    while True:
        item = await text_queue.get()
        if item is None:
            await translated_queue.put(None)
            break

        start_time, duration, text = item

        # Diarizacion: obtener speaker dominante para este segmento
        speaker_tag = ""
        if diarizer is not None:
            speaker = await asyncio.to_thread(
                diarizer.get_speaker, start_time, start_time + duration,
            )
            if speaker is not None:
                speaker_tag = f"[Speaker {speaker + 1}]"

        context_parts: list[str] = []
        if stream_context:
            context_parts.append(stream_context)
        if session_summary:
            context_parts.append(f"Current topic: {session_summary}")
        if speaker_tag:
            context_parts.append(f"Current speaker: {speaker_tag}")
        if history:
            context_parts.append("Recent translations:\n" + "\n".join(history))
        context = "\n".join(context_parts)

        try:
            translated = await translate_text(
                client, text, config.source_lang, config.target_lang,
                config.gpt_model, context=context, tracker=tracker,
            )
        except Exception as e:
            logger.error("Error inesperado traduciendo: %s", e)
            translated = None

        if translated:
            if tracker:
                tracker.add_entry(start_time, duration, text, translated)
            if speaker_tag:
                translated = f"{speaker_tag} {translated}"
            history.append(translated)
            summary_buffer.append(translated)
            await translated_queue.put((start_time, duration, translated))

            if len(summary_buffer) >= 10:
                try:
                    session_summary = await summarize_session(
                        client, summary_buffer, config.source_lang,
                    )
                    logger.info("Resumen de sesion actualizado: %s", session_summary[:80])
                except Exception as e:
                    logger.warning("Error generando resumen de sesion: %s", e)
                summary_buffer.clear()


async def _subtitle_writer(
    config: Config,
    translated_queue: asyncio.Queue,
    sub_manager: SubtitleManager,
    player: MPVPlayer,
    live: bool,
    *,
    srt_preloaded: bool = False,
) -> None:
    """Lee texto traducido y lo muestra en MPV.

    Live: usa show-text (OSD) con sync a la posicion de MPV.
    Si el pipeline va adelantado, espera a que el video alcance.
    Si va atrasado, garantiza un tiempo minimo de lectura entre subtitulos.
    VOD: usa archivo SRT con timestamps exactos.
    """
    first_subtitle = not srt_preloaded
    min_display = config.segment_duration * 0.6
    last_shown: float | None = None
    loop = asyncio.get_running_loop()
    while True:
        item = await translated_queue.get()
        if item is None:
            break

        start_time, duration, text = item
        mm, ss = divmod(int(start_time), 60)

        try:
            logger.info("Subtitulo [%02d:%02d]: %s", mm, ss, text)

            if live:
                duration_ms = int(duration * 1000)
                end_time = start_time + duration
                mpv_pos = await player.get_time_pos()
                if mpv_pos is not None and mpv_pos < end_time:
                    delay = end_time - mpv_pos
                    if delay < 30:
                        logger.debug("Sync: esperando %.1fs para alinear subtitulo", delay)
                        await asyncio.sleep(delay)
                elif last_shown is not None:
                    elapsed = loop.time() - last_shown
                    if elapsed < min_display:
                        wait = min_display - elapsed
                        logger.debug("Lectura minima: esperando %.1fs", wait)
                        await asyncio.sleep(wait)
                await player.show_text(text, duration_ms)
                last_shown = loop.time()
            else:
                end_time = start_time + duration
                sub_manager.add_entry(start_time, end_time, text)

                if first_subtitle:
                    await player.sub_add(sub_manager.srt_path)
                    first_subtitle = False
                else:
                    await player.sub_reload(sub_manager.srt_path)
        except Exception as e:
            logger.warning("Error escribiendo subtitulo: %s", e)


async def _overlay_subtitle_writer(
    config: Config,
    translated_queue: asyncio.Queue,
    overlay: SubtitleOverlay,
) -> None:
    """Lee texto traducido y lo muestra en el overlay (modo subs-only/live)."""
    while True:
        item = await translated_queue.get()
        if item is None:
            break

        start_time, _, text = item
        mm, ss = divmod(int(start_time), 60)
        try:
            logger.info("Subtitulo [%02d:%02d]: %s", mm, ss, text)
            overlay.update_text(text)
        except Exception as e:
            logger.warning("Error mostrando subtitulo en overlay: %s", e)


async def _monitor_mpv(player: MPVPlayer, segmenter: AudioSegmenter) -> None:
    """Monitorea MPV. Si cierra, detiene ffmpeg para cascadear el shutdown."""
    while True:
        if not player.is_running():
            logger.info("MPV cerrado por el usuario")
            await segmenter.stop()
            return
        await asyncio.sleep(1)


async def _monitor_overlay(
    overlay: SubtitleOverlay, segmenter: AudioSegmenter | SystemAudioCapture,
) -> None:
    """Monitorea el overlay. Si se cierra, detiene la fuente de audio."""
    while overlay.is_running():
        await asyncio.sleep(1)
    logger.info("Overlay cerrado por el usuario")
    await segmenter.stop()


async def _create_diarizer(*, vod: bool = False) -> DiarizationEngine:
    """Crea el engine de diarizacion (carga modelo, ~5s)."""
    from autosubs.diarize import DiarizationEngine, SortformerModel

    model = await asyncio.to_thread(SortformerModel, vod=vod)
    engine = DiarizationEngine(model, wait_timeout=30.0 if vod else 5.0)
    engine.start()
    return engine


async def _run_cached_pipeline(
    config: Config, srt_path: Path, video_id: str = "",
) -> None:
    """Abre MPV con video + SRT cacheado, sin re-traducir."""
    logger.info("Usando SRT cacheado, saltando transcripcion/traduccion")

    # Intentar usar video cacheado
    cached_video = LOGS_DIR / f"{video_id}.mkv" if video_id else None

    player = MPVPlayer(config.ipc_pipe)
    try:
        if cached_video and _video_download_complete(cached_video):
            logger.info("Video encontrado en cache: %s", cached_video.name)
            await player.start(str(cached_video))
            await player.connect()
            await player.sub_add(srt_path)
            # SRT tiene timestamps absolutos, seekear video a la posicion
            if config.start_time > 0:
                await player.command(
                    "set_property", "time-pos", str(config.start_time),
                )
        else:
            logger.info("Video no cacheado, obteniendo URLs del stream...")
            video_url, mpv_audio_url, _ = await get_stream_urls(config.url)
            await player.start(video_url, audio_url=mpv_audio_url)
            await player.connect()
            await player.sub_add(srt_path)

        logger.info("MPV reproduciendo con SRT cache: %s", srt_path.name)
        await player.wait()
    finally:
        await player.close()
    logger.info("autosubs finalizado (cache)")


async def run_pipeline(config: Config) -> None:
    """Orquesta todo el pipeline de subtitulos."""
    logger.info("Iniciando autosubs...")
    set_cookies_browser(config.cookies_browser)

    if config.benchmark_e2e:
        from autosubs.benchmark import run_e2e_benchmark
        await run_e2e_benchmark(config)
        return

    if config.fill_gaps:
        from autosubs.recover import fill_gaps as recover_fill_gaps
        await recover_fill_gaps(config)
        return

    if config.benchmark_translators:
        from autosubs.benchmark import run_translator_benchmark
        await run_translator_benchmark(config)
        return

    if config.benchmark:
        from autosubs.benchmark import run_benchmark
        await run_benchmark(config)
        return

    if config.system_audio:
        await _run_system_audio_pipeline(config)
        return

    # SRT cache: determinar cuanto hay traducido
    cache_end = 0.0
    if not config.no_cache and (m := YT_ID_RE.search(config.url)):
        vid = m.group(1)
        cached_srt = LOGS_DIR / f"{vid}.srt"
        if cached_srt.exists():
            cache_end = srt_last_end_time(cached_srt)
            if cache_end > 0:
                mm, ss = divmod(int(cache_end), 60)
                logger.info("SRT cache: %02d:%02d traducido", mm, ss)

    client = _create_translate_client(config.gpt_model, config.api_key)
    tracker = CostTracker(
        whisper_model=config.whisper_model,
        translation_model=config.gpt_model,
        source_lang=config.source_lang,
        target_lang=config.target_lang,
    )
    live, stream_context, video_id = await _get_stream_context(
        config.url, client, config, tracker,
    )
    logger.info("Modo: %s", "LIVE" if live else "VOD")

    # Modo batch-vod: pre-procesar audio completo con chunks grandes
    if not live and config.batch_vod:
        await _run_batch_vod_pipeline(config, stream_context, tracker, video_id)
        return

    if config.start_time > 0:
        if live:
            logger.warning("start-time ignorado: no aplica a streams en vivo")
        else:
            mm, ss = divmod(config.start_time, 60)
            logger.info("Inicio en %02d:%02d (%ds)", mm, ss, config.start_time)
    if stream_context:
        logger.info("Contexto del stream: %s", stream_context.replace("\n", " | "))

    await _run_mpv_pipeline(config, live, stream_context, tracker, video_id,
                            cache_end=cache_end)


async def _run_batch_vod_pipeline(
    config: Config, stream_context: str = "",
    tracker: CostTracker | None = None, video_id: str = "",
) -> None:
    """Pipeline batch-streaming para VODs: chunks grandes + verbose_json.

    Descarga audio completo, splitea en chunks de 20min, y procesa en streaming:
    chunk 0 se transcribe+traduce mientras MPV ya reproduce. Los siguientes
    chunks se procesan en background y se agregan al SRT en vivo.
    """
    logger.info("Modo batch-vod: descargando audio...")

    tmp_dir = Path(tempfile.mkdtemp(prefix="autosubs_batch_"))
    srt_path = tmp_dir / "autosubs.srt"
    asr_client = AsyncOpenAI(api_key=config.api_key)
    translate_client = _create_translate_client(config.gpt_model, config.api_key)
    chunk_seconds = 1200  # 20 min
    player: MPVPlayer | None = None

    # batch-vod usa whisper-1 (unico que soporta verbose_json)
    if tracker:
        tracker.whisper_model = "whisper-1"

    try:
        # Fase 1: Descargar audio y splitear
        audio_file = str(tmp_dir / "audio.mp3")
        chunks_dir = str(tmp_dir / "chunks")
        await download_audio(config.url, audio_file)
        chunks = await split_audio(audio_file, chunks_dir, chunk_seconds)
        Path(audio_file).unlink(missing_ok=True)

        # Limitar chunks si --max-minutes
        max_seconds = config.max_minutes * 60.0
        if max_seconds > 0:
            max_chunks = max(1, int(max_seconds / chunk_seconds) + 1)
            if len(chunks) > max_chunks:
                for extra in chunks[max_chunks:]:
                    extra.unlink(missing_ok=True)
                chunks = chunks[:max_chunks]
                logger.info(
                    "Limitado a %d chunks (~%d min) por --max-minutes",
                    max_chunks, config.max_minutes,
                )

        # Transcribir primer chunk antes de abrir MPV
        logger.info("Transcribiendo chunk 1/%d...", len(chunks))
        first_segs = await transcribe_verbose(
            asr_client, chunks[0], config.source_lang,
        )
        chunks[0].unlink(missing_ok=True)

        if not first_segs:
            logger.warning("Primer chunk sin segments, abortando")
            return

        if tracker:
            duration = first_segs[-1].end if first_segs else 0
            tracker.add_whisper(duration)
        logger.info("Chunk 1/%d transcrito: %d segments", len(chunks), len(first_segs))

        # Traducir primer chunk y generar SRT inicial
        sub_manager = SubtitleManager(srt_path)
        history: collections.deque[str] = collections.deque(maxlen=5)
        total_translated = 0

        async def _translate_and_write(
            segments: list[TranscriptSegment],
        ) -> None:
            nonlocal total_translated
            for seg in segments:
                if max_seconds > 0 and seg.start >= max_seconds:
                    break
                context_parts: list[str] = []
                if stream_context:
                    context_parts.append(stream_context)
                if history:
                    context_parts.append(
                        "Recent translations:\n" + "\n".join(history),
                    )
                context = "\n".join(context_parts)

                translated = await translate_text(
                    translate_client, seg.text, config.source_lang,
                    config.target_lang, config.gpt_model,
                    context=context, tracker=tracker,
                )
                if translated:
                    if tracker:
                        tracker.add_entry(
                            seg.start, seg.end - seg.start,
                            seg.text, translated,
                        )
                    history.append(translated)
                    sub_manager.add_entry(seg.start, seg.end, translated)
                    total_translated += 1

        logger.info("Traduciendo chunk 1/%d...", len(chunks))
        await _translate_and_write(first_segs)
        logger.info(
            "Chunk 1 listo: %d subtitulos. Abriendo video...",
            total_translated,
        )

        # Abrir MPV con SRT inicial
        player = MPVPlayer(config.ipc_pipe)
        if config.start_time > 0:
            full_video = str(tmp_dir / "full.mkv")
            cut_file = str(tmp_dir / "video.mkv")
            await download_vod(config.url, full_video)
            await cut_video(full_video, cut_file, config.start_time)
            Path(full_video).unlink(missing_ok=True)
            await player.start(cut_file)
            await player.connect()
            await player.sub_add(srt_path)
            await player.command(
                "set_property", "sub-delay", str(-config.start_time),
            )
        else:
            video_url, mpv_audio_url, _ = await get_stream_urls(config.url)
            await player.start(video_url, audio_url=mpv_audio_url)
            await player.connect()
            await player.sub_add(srt_path)

        # Procesar chunks restantes en background mientras MPV reproduce
        async def _process_remaining() -> None:
            for i, chunk_path in enumerate(chunks[1:], start=1):
                if not player.is_running():
                    # Limpiar chunks restantes
                    for c in chunks[i:]:
                        c.unlink(missing_ok=True)
                    break

                logger.info("Transcribiendo chunk %d/%d...", i + 1, len(chunks))
                segs = await transcribe_verbose(
                    asr_client, chunk_path, config.source_lang,
                )
                chunk_path.unlink(missing_ok=True)
                if not segs:
                    continue

                offset = i * chunk_seconds
                for seg in segs:
                    seg.start += offset
                    seg.end += offset

                if tracker:
                    duration = segs[-1].end - offset if segs else 0
                    tracker.add_whisper(duration)
                logger.info(
                    "Chunk %d/%d transcrito: %d segments",
                    i + 1, len(chunks), len(segs),
                )

                await _translate_and_write(segs)
                await player.sub_reload(srt_path)
                logger.info(
                    "Chunk %d/%d traducido. Total: %d subtitulos",
                    i + 1, len(chunks), total_translated,
                )

        if len(chunks) > 1:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(_process_remaining())
                tg.create_task(_monitor_mpv_simple(player))
        else:
            await player.wait()

        # Guardar cache al final
        if video_id and srt_path.exists():
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            cache_srt = LOGS_DIR / f"{video_id}.srt"
            shutil.copy2(srt_path, cache_srt)
            logger.info("SRT cacheado en: %s", cache_srt)

    except BaseException as e:
        if isinstance(e, ExceptionGroup):
            for exc in e.exceptions:
                if not isinstance(exc, asyncio.CancelledError):
                    logger.error("Error en pipeline batch-vod: %s", exc)
        elif not isinstance(e, (KeyboardInterrupt, asyncio.CancelledError)):
            logger.error("Error en pipeline batch-vod: %s", e)
    finally:
        if player is not None:
            await player.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.debug("Directorio temporal limpiado: %s", tmp_dir)
        if tracker:
            logger.info("\n%s", tracker.summary())
            if video_id:
                tracker.save_srt(LOGS_DIR, video_id=video_id)
            tracker.save_markdown(LOGS_DIR)
        logger.info("autosubs finalizado (batch-vod)")


async def _monitor_mpv_simple(player: MPVPlayer) -> None:
    """Espera a que MPV cierre."""
    await player.wait()


async def _run_mpv_pipeline(
    config: Config, live: bool, stream_context: str = "",
    tracker: CostTracker | None = None, video_id: str = "",
    cache_end: float = 0.0,
) -> None:
    """Pipeline con MPV: video + subtitulos SRT (modo VOD)."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="autosubs_"))
    audio_dir = tmp_dir / "audio"

    vod_start = config.start_time if not live else 0
    translate_from = max(cache_end, float(vod_start))
    bg_download: asyncio.Task | None = None

    # SRT: escribir directo al cache si hay video_id
    if video_id:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        srt_path = LOGS_DIR / f"{video_id}.srt"
        sub_manager = SubtitleManager(srt_path, append=cache_end > 0)
    else:
        srt_path = tmp_dir / "autosubs.srt"
        sub_manager = SubtitleManager(srt_path)

    if cache_end > 0:
        mm, ss = divmod(int(translate_from), 60)
        logger.info("Continuando traduccion desde %02d:%02d", mm, ss)

    # Obtener URLs para streaming (MPV + ffmpeg audio)
    logger.info("Obteniendo URLs del stream...")
    video_url, mpv_audio_url, ffmpeg_audio_url = await get_stream_urls(
        config.url,
    )

    # Video local: necesario para seek, opcional como cache. Dos condiciones
    # distintas, no confundirlas:
    #   video_complete  -> el archivo esta entero y se puede leer el audio de el
    #   video_playable  -> alcanza para que MPV salte al punto pedido, aunque
    #                      la descarga siga en curso
    # El audio nunca sale de un archivo a medias: ffmpeg llegaria al final de lo
    # descargado y cortaria la transcripcion ahi.
    cached_video: Path | None = None
    video_complete = False
    video_playable = False
    if not live and video_id:
        cached_video = LOGS_DIR / f"{video_id}.mkv"
        if _video_download_complete(cached_video):
            logger.info("Video encontrado en cache: %s", cached_video.name)
            video_complete = video_playable = True
        else:
            if cached_video.exists():
                # Descarga interrumpida: yt-dlp reanuda desde el fragmento
                # anotado en el .ytdl, no hay que borrar lo ya bajado.
                logger.info(
                    "Descarga incompleta en cache (%.0f MB), reanudando: %s",
                    cached_video.stat().st_size / (1024 * 1024), cached_video.name,
                )
            logger.info("Descargando VOD a cache en background: %s", cached_video.name)
            bg_download = asyncio.create_task(
                download_vod(config.url, str(cached_video)),
            )
            if vod_start > 0:
                # Solo hace falta el video hasta el punto pedido, no entero
                video_playable = await _wait_until_downloaded(
                    cached_video, vod_start, bg_download,
                )
                if not video_playable:
                    logger.warning(
                        "La descarga no llego a %ds, MPV reproduce del stream", vod_start,
                    )

    # Cliente OpenAI para transcripcion API (gpt-4o-transcribe etc).
    # Cliente potencialmente OpenRouter para traduccion segun gpt_model.
    asr_client = AsyncOpenAI(api_key=config.api_key)
    translate_client = _create_translate_client(config.gpt_model, config.api_key)
    player = MPVPlayer(config.ipc_pipe)
    diarizer: DiarizationEngine | None = None

    if config.diarize:
        logger.info("Cargando modelo de diarizacion...")
        diarizer = await _create_diarizer()

    # Cargar modelo local de whisper si se usa local-turbo o local-v3
    local_model: LocalWhisperModel | None = None
    if model_size := _get_local_model_size(config.whisper_model):
        local_model = await _load_local_whisper(model_size)

    pcm_callback = diarizer.push_pcm if diarizer else None

    # Audio: archivo local solo si esta COMPLETO; si hay descarga en background
    # ffmpeg lee del stream (la URL) para no chocar con yt-dlp escribiendo el .mkv
    audio_source = ffmpeg_audio_url
    if video_complete:
        audio_source = str(cached_video)

    segmenter = AudioSegmenter(
        audio_source, audio_dir, config.segment_duration,
        vad_aggressiveness=config.vad_aggressiveness,
        pcm_callback=pcm_callback,
        start_time=int(translate_from),
    )

    audio_queue: asyncio.Queue = asyncio.Queue(maxsize=5)
    text_queue: asyncio.Queue = asyncio.Queue(maxsize=5)
    translated_queue: asyncio.Queue = asyncio.Queue(maxsize=5)

    # VODs usan transcripcion paralela, excepto con whisper local (GPU no paralela)
    transcriber = _transcribe_worker if live or local_model else _transcribe_worker_parallel

    try:
        use_local = video_playable and vod_start > 0
        if use_local:
            await player.start(str(cached_video))
        else:
            await player.start(video_url, audio_url=mpv_audio_url, live=live)
        await player.connect()
        if live:
            await player.setup_osd_for_subs()

        # Pre-cargar SRT con contenido cacheado
        if cache_end > 0:
            await player.sub_add(srt_path)

        if vod_start > 0:
            await player.command("set_property", "time-pos", str(vod_start))
        await segmenter.start()

        # Sincronizar timestamps SRT con la posicion del video en MPV.
        mpv_pos = await player.get_time_pos() or 0.0
        time_offset = max(mpv_pos, translate_from)
        if time_offset > 1.0 and translate_from == 0:
            logger.info(
                "Offset A/V detectado: MPV en %.1fs, ajustando SRT", time_offset,
            )

        logger.info("Pipeline MPV en marcha. Subtitulos en: %s", srt_path)

        async with asyncio.TaskGroup() as tg:
            tg.create_task(segmenter.poll_chunks(audio_queue))
            tg.create_task(
                transcriber(
                    asr_client, config, audio_queue, text_queue,
                    tracker=tracker, start_time_offset=time_offset,
                    max_seconds=config.max_minutes * 60.0,
                    local_model=local_model,
                )
            )
            tg.create_task(
                _translate_worker(
                    translate_client, config, text_queue, translated_queue,
                    stream_context=stream_context,
                    diarizer=diarizer,
                    tracker=tracker,
                )
            )
            tg.create_task(
                _subtitle_writer(
                    config, translated_queue, sub_manager, player, live,
                    srt_preloaded=cache_end > 0,
                )
            )
            tg.create_task(_monitor_mpv(player, segmenter))

    except BaseException as e:
        if isinstance(e, ExceptionGroup):
            for exc in e.exceptions:
                if not isinstance(exc, asyncio.CancelledError):
                    logger.error("Error en pipeline: %s", exc)
        elif not isinstance(e, (KeyboardInterrupt, asyncio.CancelledError)):
            logger.error("Error en pipeline: %s", e)
    finally:
        if diarizer is not None:
            await diarizer.stop()
        await segmenter.stop()
        await player.close()
        # Esperar descarga en background si sigue corriendo
        if bg_download is not None:
            if not bg_download.done():
                logger.info("Esperando descarga de video en background...")
                try:
                    await asyncio.wait_for(bg_download, timeout=300)
                    logger.info("Video cacheado en: %s", cached_video)
                except (TimeoutError, Exception) as dl_err:
                    logger.warning("Descarga en background fallo: %s", dl_err)
                    bg_download.cancel()
            elif bg_download.exception() is None:
                logger.info("Video cacheado en: %s", cached_video)
            else:
                logger.warning("Descarga en background fallo: %s", bg_download.exception())
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.debug("Directorio temporal limpiado: %s", tmp_dir)
        if tracker:
            logger.info("\n%s", tracker.summary())
            if not video_id:
                tracker.save_srt(LOGS_DIR)
            tracker.save_markdown(LOGS_DIR)
        logger.info("autosubs finalizado")


async def _run_system_audio_pipeline(config: Config) -> None:
    """Pipeline con audio del sistema: captura loopback WASAPI + overlay."""
    logger.info("Modo audio del sistema (loopback WASAPI)")

    tmp_dir = Path(tempfile.mkdtemp(prefix="autosubs_"))
    audio_dir = tmp_dir / "audio"

    asr_client = AsyncOpenAI(api_key=config.api_key)
    translate_client = _create_translate_client(config.gpt_model, config.api_key)
    tracker = CostTracker(
        whisper_model=config.whisper_model,
        translation_model=config.gpt_model,
        source_lang=config.source_lang,
        target_lang=config.target_lang,
    )
    overlay = SubtitleOverlay()
    diarizer: DiarizationEngine | None = None

    if config.diarize:
        logger.info("Cargando modelo de diarizacion...")
        diarizer = await _create_diarizer()

    # Cargar modelo local de whisper si se usa local-turbo o local-v3
    local_model: LocalWhisperModel | None = None
    if model_size := _get_local_model_size(config.whisper_model):
        local_model = await _load_local_whisper(model_size)

    pcm_callback = diarizer.push_pcm if diarizer else None

    capture = SystemAudioCapture(
        audio_dir, config.segment_duration,
        vad_aggressiveness=config.vad_aggressiveness,
        pcm_callback=pcm_callback,
    )

    audio_queue: asyncio.Queue = asyncio.Queue(maxsize=5)
    text_queue: asyncio.Queue = asyncio.Queue(maxsize=5)
    translated_queue: asyncio.Queue = asyncio.Queue(maxsize=5)

    try:
        overlay.start()
        await capture.start()

        logger.info("Pipeline audio del sistema en marcha")

        async with asyncio.TaskGroup() as tg:
            tg.create_task(run_overlay_loop(overlay))
            tg.create_task(capture.poll_chunks(audio_queue))
            tg.create_task(
                _transcribe_worker(
                    asr_client, config, audio_queue, text_queue,
                    tracker=tracker, local_model=local_model,
                )
            )
            tg.create_task(
                _translate_worker(
                    translate_client, config, text_queue, translated_queue,
                    diarizer=diarizer,
                    tracker=tracker,
                )
            )
            tg.create_task(
                _overlay_subtitle_writer(config, translated_queue, overlay)
            )
            tg.create_task(_monitor_overlay(overlay, capture))

    except BaseException as e:
        if isinstance(e, ExceptionGroup):
            for exc in e.exceptions:
                if not isinstance(exc, asyncio.CancelledError):
                    logger.error("Error en pipeline audio sistema: %s", exc)
        elif not isinstance(e, (KeyboardInterrupt, asyncio.CancelledError)):
            logger.error("Error en pipeline audio sistema: %s", e)
    finally:
        if diarizer is not None:
            await diarizer.stop()
        await capture.stop()
        overlay.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.debug("Directorio temporal limpiado: %s", tmp_dir)
        logger.info("\n%s", tracker.summary())
        tracker.save_srt(LOGS_DIR)
        tracker.save_markdown(LOGS_DIR)
        logger.info("autosubs finalizado")
