import asyncio
import contextlib
import logging
import shutil
import tempfile
from pathlib import Path

from openai import AsyncOpenAI

from livesubs.audio import AudioSegmenter
from livesubs.capture import SystemAudioCapture
from livesubs.config import Config
from livesubs.overlay import SubtitleOverlay, run_overlay_loop
from livesubs.player import MPVPlayer
from livesubs.stream import get_audio_url, get_stream_urls, is_live
from livesubs.subtitles import SubtitleManager
from livesubs.transcribe import transcribe_audio
from livesubs.translate import translate_text

logger = logging.getLogger(__name__)


async def _transcribe_worker(
    client: AsyncOpenAI,
    config: Config,
    audio_queue: asyncio.Queue,
    text_queue: asyncio.Queue,
    segment_duration: int,
) -> None:
    """Lee chunks de audio, los transcribe secuencialmente (para lives)."""
    chunk_index = 0
    while True:
        audio_path: Path | None = await audio_queue.get()
        if audio_path is None:
            await text_queue.put(None)
            break

        start_time = chunk_index * segment_duration
        try:
            text = await transcribe_audio(
                client, audio_path, config.source_lang, config.whisper_model,
            )
        except Exception as e:
            logger.error("Error inesperado transcribiendo %s: %s", audio_path.name, e)
            text = None

        with contextlib.suppress(OSError):
            audio_path.unlink()

        if text:
            await text_queue.put((start_time, text))
        chunk_index += 1


async def _transcribe_worker_parallel(
    client: AsyncOpenAI,
    config: Config,
    audio_queue: asyncio.Queue,
    text_queue: asyncio.Queue,
    segment_duration: int,
    max_concurrent: int = 3,
) -> None:
    """Transcribe multiples chunks en paralelo, mantiene orden (para VODs)."""
    semaphore = asyncio.Semaphore(max_concurrent)
    task_queue: asyncio.Queue[asyncio.Task[str | None] | None] = asyncio.Queue()

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
            audio_path: Path | None = await audio_queue.get()
            if audio_path is None:
                await task_queue.put(None)
                break
            await task_queue.put(asyncio.create_task(_do_transcribe(audio_path)))

    receiver = asyncio.create_task(_receiver())

    chunk_index = 0
    while True:
        item = await task_queue.get()
        if item is None:
            break
        text = await item
        if text:
            start_time = chunk_index * segment_duration
            await text_queue.put((start_time, text))
        chunk_index += 1

    await receiver
    await text_queue.put(None)


async def _translate_worker(
    client: AsyncOpenAI,
    config: Config,
    text_queue: asyncio.Queue,
    translated_queue: asyncio.Queue,
) -> None:
    """Lee texto transcrito, lo traduce y pone en la queue."""
    while True:
        item = await text_queue.get()
        if item is None:
            await translated_queue.put(None)
            break

        start_time, text = item
        try:
            translated = await translate_text(
                client, text, config.source_lang, config.target_lang, config.gpt_model,
            )
        except Exception as e:
            logger.error("Error inesperado traduciendo: %s", e)
            translated = None

        if translated:
            await translated_queue.put((start_time, translated))


async def _subtitle_writer(
    config: Config,
    translated_queue: asyncio.Queue,
    sub_manager: SubtitleManager,
    player: MPVPlayer,
    live: bool,
) -> None:
    """Lee texto traducido y lo muestra en MPV.

    Live: usa show-text (OSD) para mostrar inmediatamente en pantalla.
    VOD: usa archivo SRT con timestamps exactos.
    """
    first_subtitle = True
    duration_ms = config.segment_duration * 1000
    while True:
        item = await translated_queue.get()
        if item is None:
            break

        start_time, text = item

        try:
            logger.info("Subtitulo: %s", text[:80])

            if live:
                await player.show_text(text, duration_ms)
            else:
                end_time = start_time + config.segment_duration
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

        _, text = item
        try:
            logger.info("Subtitulo: %s", text[:80])
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


async def run_pipeline(config: Config) -> None:
    """Orquesta todo el pipeline de subtitulos."""
    logger.info("Iniciando livesubs...")

    if config.system_audio:
        await _run_system_audio_pipeline(config)
        return

    live = await is_live(config.url)
    logger.info("Modo: %s", "LIVE" if live else "VOD")

    # --subs-only fuerza overlay (sin MPV). Lives usan MPV con show-text.
    use_overlay = config.subs_only

    if use_overlay:
        await _run_overlay_pipeline(config, live)
    else:
        await _run_mpv_pipeline(config, live)


async def _run_overlay_pipeline(config: Config, live: bool) -> None:
    """Pipeline con overlay: solo audio + subtitulos en ventana transparente."""
    logger.info("Modo overlay: solo subtitulos")

    tmp_dir = Path(tempfile.mkdtemp(prefix="livesubs_"))
    audio_dir = tmp_dir / "audio"

    client = AsyncOpenAI(api_key=config.api_key)
    overlay = SubtitleOverlay()

    audio_queue: asyncio.Queue = asyncio.Queue(maxsize=5)
    text_queue: asyncio.Queue = asyncio.Queue(maxsize=5)
    translated_queue: asyncio.Queue = asyncio.Queue(maxsize=5)

    # Transcripcion paralela para VODs, secuencial para lives
    transcriber = _transcribe_worker if live else _transcribe_worker_parallel

    # Lives: yt-dlp pipe a ffmpeg (maneja reconexiones automaticamente)
    # VODs con --subs-only: URL directa a ffmpeg
    if live:
        segmenter = AudioSegmenter("", audio_dir, config.segment_duration)
    else:
        logger.info("Obteniendo URL de audio...")
        audio_url = await get_audio_url(config.url)
        segmenter = AudioSegmenter(audio_url, audio_dir, config.segment_duration)

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
                transcriber(
                    client, config, audio_queue, text_queue, config.segment_duration,
                )
            )
            tg.create_task(
                _translate_worker(client, config, text_queue, translated_queue)
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
        await segmenter.stop()
        overlay.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.debug("Directorio temporal limpiado: %s", tmp_dir)
        logger.info("livesubs finalizado")


async def _run_mpv_pipeline(config: Config, live: bool) -> None:
    """Pipeline con MPV: video + subtitulos SRT (modo VOD)."""
    logger.info("Obteniendo URLs del stream...")
    video_url, mpv_audio_url, ffmpeg_audio_url = await get_stream_urls(config.url)

    tmp_dir = Path(tempfile.mkdtemp(prefix="livesubs_"))
    srt_path = tmp_dir / "livesubs.srt"
    audio_dir = tmp_dir / "audio"

    client = AsyncOpenAI(api_key=config.api_key)
    player = MPVPlayer(config.ipc_pipe)
    segmenter = AudioSegmenter(ffmpeg_audio_url, audio_dir, config.segment_duration)
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
                transcriber(
                    client, config, audio_queue, text_queue, config.segment_duration,
                )
            )
            tg.create_task(
                _translate_worker(client, config, text_queue, translated_queue)
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
        await segmenter.stop()
        await player.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.debug("Directorio temporal limpiado: %s", tmp_dir)
        logger.info("livesubs finalizado")


async def _run_system_audio_pipeline(config: Config) -> None:
    """Pipeline con audio del sistema: captura loopback WASAPI + overlay."""
    logger.info("Modo audio del sistema (loopback WASAPI)")

    tmp_dir = Path(tempfile.mkdtemp(prefix="livesubs_"))
    audio_dir = tmp_dir / "audio"

    client = AsyncOpenAI(api_key=config.api_key)
    overlay = SubtitleOverlay()
    capture = SystemAudioCapture(audio_dir, config.segment_duration)

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
                    client, config, audio_queue, text_queue, config.segment_duration,
                )
            )
            tg.create_task(
                _translate_worker(client, config, text_queue, translated_queue)
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
        await capture.stop()
        overlay.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.debug("Directorio temporal limpiado: %s", tmp_dir)
        logger.info("livesubs finalizado")
