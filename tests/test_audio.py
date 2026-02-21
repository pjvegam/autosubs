import asyncio
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from livesubs.audio import AudioSegmenter


class TestAudioSegmenter:
    def test_init(self, tmp_path):
        seg = AudioSegmenter("https://audio.url", tmp_path / "audio", 10)
        assert seg.audio_url == "https://audio.url"
        assert seg.segment_duration == 10
        assert seg.process is None
        assert seg._ffmpeg_popen is None
        assert seg._ytdlp_popen is None

    async def test_start_creates_output_dir(self, tmp_path):
        audio_dir = tmp_path / "audio"
        seg = AudioSegmenter("https://audio.url", audio_dir, 10)

        mock_proc = AsyncMock()
        mock_proc.returncode = None

        with patch(
            "livesubs.audio.asyncio.create_subprocess_exec", return_value=mock_proc,
        ) as mock_exec:
            await seg.start()

        assert audio_dir.exists()
        assert seg.process is mock_proc

        # Verificar que ffmpeg se llamo con los argumentos correctos
        call_args = mock_exec.call_args[0]
        assert call_args[0] == "ffmpeg"
        assert "-f" in call_args
        assert "segment" in call_args
        assert "-segment_time" in call_args
        assert "10" in call_args
        assert "-ar" in call_args
        assert "16000" in call_args

    async def test_start_live_creates_pipeline(self, tmp_path):
        audio_dir = tmp_path / "audio"
        seg = AudioSegmenter("", audio_dir, 5)

        mock_ytdlp = MagicMock(spec=subprocess.Popen)
        mock_ytdlp.stdout = MagicMock()
        mock_ytdlp.poll.return_value = None

        mock_ffmpeg = MagicMock(spec=subprocess.Popen)
        mock_ffmpeg.poll.return_value = None

        with patch(
            "livesubs.audio.asyncio.to_thread",
            new_callable=AsyncMock,
            return_value=(mock_ytdlp, mock_ffmpeg),
        ):
            await seg.start_live("https://youtube.com/watch?v=test")

        assert audio_dir.exists()
        assert seg._ytdlp_popen is mock_ytdlp
        assert seg._ffmpeg_popen is mock_ffmpeg

    async def test_poll_chunks_detects_completed_chunks(self, tmp_path):
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        seg = AudioSegmenter("https://audio.url", audio_dir, 10)
        # Simular que ffmpeg ya no esta corriendo
        seg.process = AsyncMock()
        seg.process.returncode = 0  # ya termino

        # Crear dos chunks
        (audio_dir / "chunk_0000.wav").write_bytes(b"audio0")
        (audio_dir / "chunk_0001.wav").write_bytes(b"audio1")

        queue: asyncio.Queue = asyncio.Queue()
        await seg.poll_chunks(queue)

        chunks = []
        while not queue.empty():
            item = await queue.get()
            chunks.append(item)

        # Deberia tener chunk_0000, chunk_0001, y None sentinel
        assert len(chunks) == 3
        assert chunks[0].name == "chunk_0000.wav"
        assert chunks[1].name == "chunk_0001.wav"
        assert chunks[2] is None

    async def test_poll_chunks_sends_sentinel_when_no_chunks(self, tmp_path):
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        seg = AudioSegmenter("https://audio.url", audio_dir, 10)
        seg.process = AsyncMock()
        seg.process.returncode = 0  # ffmpeg already finished, no chunks

        queue: asyncio.Queue = asyncio.Queue()
        await seg.poll_chunks(queue)

        item = await queue.get()
        assert item is None

    def test_is_running_false_when_no_process(self):
        seg = AudioSegmenter("url", Path("/tmp"), 10)
        assert seg.is_running() is False

    def test_is_running_true_when_running(self):
        seg = AudioSegmenter("url", Path("/tmp"), 10)
        seg.process = AsyncMock()
        seg.process.returncode = None
        assert seg.is_running() is True

    def test_is_running_false_when_finished(self):
        seg = AudioSegmenter("url", Path("/tmp"), 10)
        seg.process = AsyncMock()
        seg.process.returncode = 0
        assert seg.is_running() is False

    def test_is_running_live_mode(self):
        seg = AudioSegmenter("", Path("/tmp"), 5)
        seg._ffmpeg_popen = MagicMock(spec=subprocess.Popen)
        seg._ffmpeg_popen.poll.return_value = None
        assert seg.is_running() is True

        seg._ffmpeg_popen.poll.return_value = 0
        assert seg.is_running() is False

    async def test_stop_terminates_process(self):
        seg = AudioSegmenter("url", Path("/tmp"), 10)
        seg.process = AsyncMock()
        seg.process.returncode = None
        seg.process.terminate = MagicMock()  # terminate es sincrono
        seg.process.wait = AsyncMock()

        await seg.stop()

        seg.process.terminate.assert_called_once()

    async def test_stop_live_terminates_both(self):
        seg = AudioSegmenter("", Path("/tmp"), 5)

        mock_ffmpeg = MagicMock(spec=subprocess.Popen)
        mock_ffmpeg.poll.return_value = None
        mock_ytdlp = MagicMock(spec=subprocess.Popen)
        mock_ytdlp.poll.return_value = None

        seg._ffmpeg_popen = mock_ffmpeg
        seg._ytdlp_popen = mock_ytdlp

        with patch("livesubs.audio.asyncio.to_thread", new_callable=AsyncMock):
            await seg.stop()

        mock_ffmpeg.terminate.assert_called_once()
        mock_ytdlp.terminate.assert_called_once()
