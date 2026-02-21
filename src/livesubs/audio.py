import asyncio
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class AudioSegmenter:
    """Segmenta audio de un stream en chunks WAV usando ffmpeg."""

    def __init__(
        self,
        audio_url: str,
        output_dir: Path,
        segment_duration: int = 10,
    ) -> None:
        self.audio_url = audio_url
        self.output_dir = output_dir
        self.segment_duration = segment_duration
        self.process: asyncio.subprocess.Process | None = None
        self._ffmpeg_popen: subprocess.Popen | None = None
        self._ytdlp_popen: subprocess.Popen | None = None
        self._chunk_index = 0

    def _ffmpeg_segment_args(self, input_source: str) -> list[str]:
        """Argumentos de ffmpeg para segmentar audio en WAV."""
        pattern = str(self.output_dir / "chunk_%04d.wav")
        return [
            "ffmpeg",
            "-loglevel", "warning",
            "-i", input_source,
            "-f", "segment",
            "-segment_time", str(self.segment_duration),
            "-ar", "16000",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            "-y",
            pattern,
        ]

    async def start(self) -> None:
        """Lanza ffmpeg leyendo de URL directa."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        cmd = self._ffmpeg_segment_args(self.audio_url)
        logger.info("Lanzando ffmpeg para segmentar audio...")
        self.process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )

    async def start_live(self, url: str) -> None:
        """Lanza yt-dlp → ffmpeg pipe para streams en vivo."""
        from livesubs.stream import ytdlp_pipe_audio_cmd

        self.output_dir.mkdir(parents=True, exist_ok=True)
        ytdlp_cmd = ytdlp_pipe_audio_cmd(url)
        ffmpeg_cmd = self._ffmpeg_segment_args("pipe:0")

        logger.info("Lanzando yt-dlp → ffmpeg para audio en vivo...")

        def _start_pipeline() -> tuple[subprocess.Popen, subprocess.Popen]:
            ytdlp = subprocess.Popen(
                ytdlp_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            ffmpeg = subprocess.Popen(
                ffmpeg_cmd,
                stdin=ytdlp.stdout,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            ytdlp.stdout.close()
            return ytdlp, ffmpeg

        self._ytdlp_popen, self._ffmpeg_popen = await asyncio.to_thread(_start_pipeline)
        logger.info("Pipeline yt-dlp → ffmpeg iniciado")

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

    async def poll_chunks(self, queue: asyncio.Queue) -> None:
        """Detecta nuevos chunks WAV y los pone en la queue."""
        stall_seconds = 0
        while True:
            expected = self.output_dir / f"chunk_{self._chunk_index:04d}.wav"
            next_chunk = self.output_dir / f"chunk_{self._chunk_index + 1:04d}.wav"

            if next_chunk.exists():
                logger.debug("Chunk listo: %s", expected.name)
                await queue.put(expected)
                self._chunk_index += 1
                stall_seconds = 0
            elif not self.is_running() and expected.exists():
                logger.debug("Ultimo chunk: %s", expected.name)
                await queue.put(expected)
                self._chunk_index += 1
                await self._log_ffmpeg_status()
                break
            elif not self.is_running():
                logger.warning("ffmpeg termino sin producir mas chunks")
                await self._log_ffmpeg_status()
                break
            else:
                stall_seconds += 1
                if stall_seconds == 30:
                    logger.warning("Sin chunks nuevos por 30s")
                    await self._log_ffmpeg_status()
                await asyncio.sleep(1)

        await queue.put(None)
        logger.info("Segmentacion finalizada (%d chunks procesados)", self._chunk_index)

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
