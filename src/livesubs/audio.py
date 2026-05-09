import asyncio
import logging
import queue
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path

from livesubs.vad import AudioChunk, VadSegmenter

logger = logging.getLogger(__name__)

# Leer ~1s de audio por iteracion (16kHz mono 16-bit)
READ_CHUNK_BYTES = 16000 * 2

# EQ optimizado para transcripcion de voz (inspirado en FxSound "Transcription")
# - highpass 80Hz: elimina rumble/graves que no son voz
# - boost 250Hz +5dB: refuerza fundamental de voz
# - boost 4500Hz +8dB: claridad de consonantes (s, t, k)
# - lowpass 8000Hz: elimina ruido de alta frecuencia
# - acompressor: compresion dinamica para nivelar volumen (partes quietas mas audibles)
TRANSCRIPTION_AUDIO_FILTER = (
    "highpass=f=80,"
    "equalizer=f=250:t=h:w=100:g=5,"
    "equalizer=f=4500:t=h:w=1500:g=8,"
    "lowpass=f=8000,"
    "acompressor=threshold=-20dB:ratio=4:attack=5:release=50"
)


class AudioSegmenter:
    """Segmenta audio de un stream en chunks WAV usando ffmpeg + VAD."""

    def __init__(
        self,
        audio_url: str,
        output_dir: Path,
        segment_duration: int = 5,
        *,
        vad_aggressiveness: int = 2,
        pcm_callback: Callable[[bytes], None] | None = None,
        start_time: int = 0,
    ) -> None:
        self.audio_url = audio_url
        self.output_dir = output_dir
        self.segment_duration = segment_duration
        self._vad_aggressiveness = vad_aggressiveness
        self._pcm_callback = pcm_callback
        self._start_time = start_time
        self.process: asyncio.subprocess.Process | None = None
        self._ffmpeg_popen: subprocess.Popen | None = None
        self._ytdlp_popen: subprocess.Popen | None = None
        self._chunk_queue: queue.Queue[AudioChunk | None] | None = None
        self._reader_thread: threading.Thread | None = None

    def _make_vad_segmenter(self) -> VadSegmenter:
        min_dur = self.segment_duration * 0.6
        max_dur = self.segment_duration * 1.6
        return VadSegmenter(
            self.output_dir,
            min_duration=min_dur,
            target_duration=float(self.segment_duration),
            max_duration=max_dur,
            aggressiveness=self._vad_aggressiveness,
        )

    def _ffmpeg_continuous_args(self, input_source: str) -> list[str]:
        """Argumentos de ffmpeg para output PCM continuo a stdout.

        Aplica EQ de transcripcion: corta graves/agudos, boost en frecuencias
        de voz (250Hz, 4.5kHz) y compresion dinamica para normalizar volumen.
        """
        cmd = ["ffmpeg", "-loglevel", "warning"]
        if self._start_time > 0:
            cmd.extend(["-ss", str(self._start_time)])
        cmd.extend([
            "-i", input_source,
            "-af", TRANSCRIPTION_AUDIO_FILTER,
            "-ar", "16000",
            "-ac", "1",
            "-f", "s16le",
            "-acodec", "pcm_s16le",
            "pipe:1",
        ])
        return cmd

    async def start(self) -> None:
        """Lanza ffmpeg leyendo de URL directa, output PCM a stdout."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        cmd = self._ffmpeg_continuous_args(self.audio_url)
        logger.info("Lanzando ffmpeg para audio continuo...")
        self.process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    async def start_live(self, url: str) -> None:
        """Lanza yt-dlp → ffmpeg pipe para streams en vivo."""
        from livesubs.stream import ytdlp_pipe_audio_cmd

        self.output_dir.mkdir(parents=True, exist_ok=True)
        ytdlp_cmd = ytdlp_pipe_audio_cmd(url)
        ffmpeg_cmd = self._ffmpeg_continuous_args("pipe:0")

        logger.info("Lanzando yt-dlp → ffmpeg para audio en vivo...")

        self._chunk_queue = queue.Queue()

        def _start_pipeline() -> tuple[subprocess.Popen, subprocess.Popen]:
            ytdlp = subprocess.Popen(
                ytdlp_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            ffmpeg = subprocess.Popen(
                ffmpeg_cmd,
                stdin=ytdlp.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            ytdlp.stdout.close()
            return ytdlp, ffmpeg

        self._ytdlp_popen, self._ffmpeg_popen = await asyncio.to_thread(_start_pipeline)

        # Thread lector que alimenta VAD con el stdout de ffmpeg (blocking I/O)
        self._reader_thread = threading.Thread(
            target=self._live_reader_loop, daemon=True,
        )
        self._reader_thread.start()
        logger.info("Pipeline yt-dlp → ffmpeg iniciado")

    def _live_reader_loop(self) -> None:
        """Lee PCM de ffmpeg popen stdout y alimenta VAD (corre en thread)."""
        segmenter = self._make_vad_segmenter()
        stdout = self._ffmpeg_popen.stdout if self._ffmpeg_popen else None
        if stdout is None or self._chunk_queue is None:
            return

        try:
            while True:
                data = stdout.read(READ_CHUNK_BYTES)
                if not data:
                    break
                if self._pcm_callback is not None:
                    self._pcm_callback(data)
                for chunk in segmenter.feed(data):
                    self._chunk_queue.put(chunk)

            if remaining := segmenter.flush_remaining():
                self._chunk_queue.put(remaining)
        except Exception as e:
            logger.error("Error en reader thread: %s", e)
        finally:
            self._chunk_queue.put(None)

    async def _log_ffmpeg_status(self) -> None:
        """Lee stderr de ffmpeg y loguea info de diagnostico."""
        if self.process is not None:
            rc = self.process.returncode
            alive = rc is None
            logger.warning("ffmpeg: alive=%s, returncode=%s", alive, rc)
            if not alive and self.process.stderr:
                try:
                    data = await asyncio.wait_for(
                        self.process.stderr.read(65536), timeout=2,
                    )
                    if isinstance(data, bytes):
                        text = data.decode(errors="replace").strip()
                        for line in text.splitlines()[-5:]:
                            logger.warning("ffmpeg stderr: %s", line)
                except (TimeoutError, OSError, TypeError):
                    pass

    async def poll_chunks(self, async_queue: asyncio.Queue) -> None:
        """Lee PCM de stdout de ffmpeg, segmenta con VAD, encola AudioChunks."""
        if self._chunk_queue is not None:
            # Modo live: drena la queue del reader thread
            while True:
                chunk = await asyncio.to_thread(self._chunk_queue.get)
                if chunk is None:
                    break
                logger.debug("Chunk listo: %s (%.2fs)", chunk.path.name, chunk.duration)
                await async_queue.put(chunk)

            await async_queue.put(None)
            logger.info("Segmentacion finalizada (live)")
            return

        # Modo VOD: leer directamente del process async
        if self.process is None or self.process.stdout is None:
            await async_queue.put(None)
            return

        segmenter = self._make_vad_segmenter()
        chunk_count = 0

        while True:
            data = await self.process.stdout.read(READ_CHUNK_BYTES)
            if not data:
                break
            if self._pcm_callback is not None:
                self._pcm_callback(data)
            for chunk in segmenter.feed(data):
                logger.debug("Chunk listo: %s (%.2fs)", chunk.path.name, chunk.duration)
                await async_queue.put(chunk)
                chunk_count += 1

        if remaining := segmenter.flush_remaining():
            await async_queue.put(remaining)
            chunk_count += 1

        await self._log_ffmpeg_status()
        await async_queue.put(None)
        logger.info("Segmentacion finalizada (%d chunks procesados)", chunk_count)

    def is_running(self) -> bool:
        """Verifica si el pipeline de audio sigue corriendo."""
        if self._ffmpeg_popen is not None:
            return self._ffmpeg_popen.poll() is None
        if self.process is not None:
            return self.process.returncode is None
        return False

    async def stop(self) -> None:
        """Detiene todos los procesos de audio."""
        if self.process is not None and self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5)
            except TimeoutError:
                self.process.kill()

        for proc in [self._ffmpeg_popen, self._ytdlp_popen]:
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    await asyncio.to_thread(proc.wait, timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
