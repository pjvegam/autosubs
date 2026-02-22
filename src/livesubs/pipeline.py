from __future__ import annotations

import asyncio
import collections
import contextlib
import logging
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from openai import AsyncOpenAI

from livesubs.audio import AudioSegmenter
from livesubs.capture import SystemAudioCapture
from livesubs.config import Config
from livesubs.costs import CostTracker
from livesubs.overlay import SubtitleOverlay, run_overlay_loop
from livesubs.player import MPVPlayer
from livesubs.stream import get_audio_url, get_stream_info, get_stream_urls
from livesubs.subtitles import SubtitleManager
from livesubs.transcribe import transcribe_audio
from livesubs.translate import translate_text
from livesubs.vad import AudioChunk

if TYPE_CHECKING:
    from livesubs.diarize import DiarizationEngine

logger = logging.getLogger(__name__)

LOGS_DIR = Path("livesubs_logs")
MIN_SPEECH_RATIO = 0.15


async def _get_stream_context(
    url: str, client: AsyncOpenAI, config: Config,
    tracker: CostTracker | None = None,
) -> tuple[bool, str]:
    """Extrae is_live y contexto de metadata en una sola llamada a yt-dlp.

    Traduce el titulo para dar contexto util a GPT.
    """
    try:
        info = await get_stream_info(url)
        live = info.get("is_live", False)
        title = info.get("title", "")
        channel = info.get("channel", "") or info.get("uploader", "")

        if title:
            translated_title = await translate_text(
                client, title, config.source_lang, config.target_lang,
                config.gpt_model, tracker=tracker,
            )
            if translated_title:
                title = f"{title} ({translated_title})"

        parts = [f"Canal: {channel}" if channel else "", f"Titulo: {title}" if title else ""]
        context = "\n".join(p for p in parts if p)
        return live, context
    except Exception:
        logger.warning("No se pudo obtener metadata del stream, asumiendo VOD sin contexto")
        return False, ""


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
) -> None:
    """Lee chunks de audio, los transcribe secuencialmente (para lives)."""
    elapsed = 0.0
    while True:
        chunk: AudioChunk | None = await audio_queue.get()
        if chunk is None:
            await text_queue.put(None)
            break

        start_time = elapsed
        elapsed += chunk.duration

        if _should_skip_chunk(chunk):
            with contextlib.suppress(OSError):
                chunk.path.unlink()
            continue

        try:
            text = await transcribe_audio(
                client, chunk.path, config.source_lang, config.whisper_model,
            )
        except Exception as e:
            logger.error("Error inesperado transcribiendo %s: %s", chunk.path.name, e)
            text = None

        with contextlib.suppress(OSError):
            chunk.path.unlink()

        if text:
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
) -> None:
    """Transcribe multiples chunks en paralelo, mantiene orden (para VODs)."""
    semaphore = asyncio.Semaphore(max_concurrent)
    task_queue: asyncio.Queue[tuple[asyncio.Task[str | None], float] | None] = asyncio.Queue()

    async def _do_transcribe(audio_path: Path) -> str | None:
        async with semaphore:
            try:
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

    elapsed = 0.0
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
        if speaker_tag:
            context_parts.append(f"Hablante actual: {speaker_tag}")
        if history:
            context_parts.append("Traducciones recientes:\n" + "\n".join(history))
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
            await translated_queue.put((start_time, duration, translated))


async def _subtitle_writer(
    config: Config,
    translated_queue: asyncio.Queue,
    sub_manager: SubtitleManager,
    player: MPVPlayer,
    live: bool,
) -> None:
    """Lee texto traducido y lo muestra en MPV.

    Live: usa show-text (OSD) con sync a la posicion de MPV.
    Si el pipeline va adelantado, espera a que el video alcance.
    Si va atrasado, garantiza un tiempo minimo de lectura entre subtitulos.
    VOD: usa archivo SRT con timestamps exactos.
    """
    first_subtitle = True
    min_display = config.segment_duration * 0.6
    last_shown: float | None = None
    loop = asyncio.get_event_loop()
    while True:
        item = await translated_queue.get()
        if item is None:
            break

        start_time, duration, text = item
        mm, ss = divmod(int(start_time), 60)

        try:
            logger.info("Subtitulo [%02d:%02d]: %s", mm, ss, text[:80])

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
            logger.info("Subtitulo [%02d:%02d]: %s", mm, ss, text[:80])
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
    from livesubs.diarize import DiarizationEngine, SortformerModel

    model = await asyncio.to_thread(SortformerModel, vod=vod)
    engine = DiarizationEngine(model, wait_timeout=30.0 if vod else 5.0)
    engine.start()
    return engine


async def run_pipeline(config: Config) -> None:
    """Orquesta todo el pipeline de subtitulos."""
    logger.info("Iniciando livesubs...")

    if config.system_audio:
        await _run_system_audio_pipeline(config)
        return

    client = AsyncOpenAI(api_key=config.api_key)
    tracker = CostTracker(
        whisper_model=config.whisper_model,
        translation_model=config.gpt_model,
        source_lang=config.source_lang,
        target_lang=config.target_lang,
    )
    live, stream_context = await _get_stream_context(config.url, client, config, tracker)
    logger.info("Modo: %s", "LIVE" if live else "VOD")
    if stream_context:
        logger.info("Contexto del stream: %s", stream_context.replace("\n", " | "))

    # --subs-only fuerza overlay (sin MPV). Lives usan MPV con show-text.
    use_overlay = config.subs_only

    if use_overlay:
        await _run_overlay_pipeline(config, live, stream_context, tracker)
    else:
        await _run_mpv_pipeline(config, live, stream_context, tracker)


async def _run_overlay_pipeline(
    config: Config, live: bool, stream_context: str = "",
    tracker: CostTracker | None = None,
) -> None:
    """Pipeline con overlay: solo audio + subtitulos en ventana transparente."""
    logger.info("Modo overlay: solo subtitulos")

    tmp_dir = Path(tempfile.mkdtemp(prefix="livesubs_"))
    audio_dir = tmp_dir / "audio"

    client = AsyncOpenAI(api_key=config.api_key)
    overlay = SubtitleOverlay()
    diarizer: DiarizationEngine | None = None

    if config.diarize:
        logger.info("Cargando modelo de diarizacion...")
        diarizer = await _create_diarizer()

    pcm_callback = diarizer.push_pcm if diarizer else None

    audio_queue: asyncio.Queue = asyncio.Queue(maxsize=5)
    text_queue: asyncio.Queue = asyncio.Queue(maxsize=5)
    translated_queue: asyncio.Queue = asyncio.Queue(maxsize=5)

    # Transcripcion paralela para VODs, secuencial para lives
    transcriber = _transcribe_worker if live else _transcribe_worker_parallel

    # Lives: yt-dlp pipe a ffmpeg (maneja reconexiones automaticamente)
    # VODs con --subs-only: URL directa a ffmpeg
    if live:
        segmenter = AudioSegmenter(
            "", audio_dir, config.segment_duration,
            vad_aggressiveness=config.vad_aggressiveness,
            pcm_callback=pcm_callback,
        )
    else:
        logger.info("Obteniendo URL de audio...")
        audio_url = await get_audio_url(config.url)
        segmenter = AudioSegmenter(
            audio_url, audio_dir, config.segment_duration,
            vad_aggressiveness=config.vad_aggressiveness,
            pcm_callback=pcm_callback,
        )

    try:
        overlay.start()
        if live:
            await segmenter.start_live(config.url)
        else:
            await segmenter.start()

        logger.info("Pipeline overlay en marcha")

        async with asyncio.TaskGroup() as tg:
            tg.create_task(run_overlay_loop(overlay))
            tg.create_task(segmenter.poll_chunks(audio_queue))
            tg.create_task(
                transcriber(client, config, audio_queue, text_queue, tracker=tracker)
            )
            tg.create_task(
                _translate_worker(
                    client, config, text_queue, translated_queue,
                    stream_context=stream_context,
                    diarizer=diarizer,
                    tracker=tracker,
                )
            )
            tg.create_task(
                _overlay_subtitle_writer(config, translated_queue, overlay)
            )
            tg.create_task(_monitor_overlay(overlay, segmenter))

    except BaseException as e:
        if isinstance(e, ExceptionGroup):
            for exc in e.exceptions:
                if not isinstance(exc, asyncio.CancelledError):
                    logger.error("Error en pipeline overlay: %s", exc)
        elif not isinstance(e, (KeyboardInterrupt, asyncio.CancelledError)):
            logger.error("Error en pipeline overlay: %s", e)
    finally:
        if diarizer is not None:
            await diarizer.stop()
        await segmenter.stop()
        overlay.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.debug("Directorio temporal limpiado: %s", tmp_dir)
        if tracker:
            logger.info("\n%s", tracker.summary())
            tracker.save_markdown(LOGS_DIR)
        logger.info("livesubs finalizado")


async def _run_mpv_pipeline(
    config: Config, live: bool, stream_context: str = "",
    tracker: CostTracker | None = None,
) -> None:
    """Pipeline con MPV: video + subtitulos SRT (modo VOD)."""
    logger.info("Obteniendo URLs del stream...")
    video_url, mpv_audio_url, ffmpeg_audio_url = await get_stream_urls(config.url)

    tmp_dir = Path(tempfile.mkdtemp(prefix="livesubs_"))
    srt_path = tmp_dir / "livesubs.srt"
    audio_dir = tmp_dir / "audio"

    client = AsyncOpenAI(api_key=config.api_key)
    player = MPVPlayer(config.ipc_pipe)
    diarizer: DiarizationEngine | None = None

    if config.diarize:
        logger.info("Cargando modelo de diarizacion...")
        diarizer = await _create_diarizer()

    pcm_callback = diarizer.push_pcm if diarizer else None

    segmenter = AudioSegmenter(
        ffmpeg_audio_url, audio_dir, config.segment_duration,
        vad_aggressiveness=config.vad_aggressiveness,
        pcm_callback=pcm_callback,
    )
    sub_manager = SubtitleManager(srt_path)

    audio_queue: asyncio.Queue = asyncio.Queue(maxsize=5)
    text_queue: asyncio.Queue = asyncio.Queue(maxsize=5)
    translated_queue: asyncio.Queue = asyncio.Queue(maxsize=5)

    # VODs usan transcripcion paralela para reducir latencia inicial
    transcriber = _transcribe_worker if live else _transcribe_worker_parallel

    try:
        await player.start(video_url, audio_url=mpv_audio_url, live=live)
        await player.connect()
        if live:
            await player.setup_osd_for_subs()
        await segmenter.start()

        logger.info("Pipeline MPV en marcha. Subtitulos en: %s", srt_path)

        async with asyncio.TaskGroup() as tg:
            tg.create_task(segmenter.poll_chunks(audio_queue))
            tg.create_task(
                transcriber(client, config, audio_queue, text_queue, tracker=tracker)
            )
            tg.create_task(
                _translate_worker(
                    client, config, text_queue, translated_queue,
                    stream_context=stream_context,
                    diarizer=diarizer,
                    tracker=tracker,
                )
            )
            tg.create_task(
                _subtitle_writer(
                    config, translated_queue, sub_manager, player, live,
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
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.debug("Directorio temporal limpiado: %s", tmp_dir)
        if tracker:
            logger.info("\n%s", tracker.summary())
            tracker.save_markdown(LOGS_DIR)
        logger.info("livesubs finalizado")


async def _run_system_audio_pipeline(config: Config) -> None:
    """Pipeline con audio del sistema: captura loopback WASAPI + overlay."""
    logger.info("Modo audio del sistema (loopback WASAPI)")

    tmp_dir = Path(tempfile.mkdtemp(prefix="livesubs_"))
    audio_dir = tmp_dir / "audio"

    client = AsyncOpenAI(api_key=config.api_key)
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
                _transcribe_worker(client, config, audio_queue, text_queue, tracker=tracker)
            )
            tg.create_task(
                _translate_worker(
                    client, config, text_queue, translated_queue,
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
        tracker.save_markdown(LOGS_DIR)
        logger.info("livesubs finalizado")
