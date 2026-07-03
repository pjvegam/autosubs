import asyncio
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from autosubs.audio import AudioSegmenter
from autosubs.vad import AudioChunk


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
            "autosubs.audio.asyncio.create_subprocess_exec", return_value=mock_proc,
        ) as mock_exec:
            await seg.start()

        assert audio_dir.exists()
        assert seg.process is mock_proc

        # Verificar que ffmpeg se llamo con output PCM a stdout
        call_args = mock_exec.call_args[0]
        assert call_args[0] == "ffmpeg"
        assert "-f" in call_args
        assert "s16le" in call_args
        assert "pipe:1" in call_args
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
        mock_ffmpeg.stdout = MagicMock()

        with patch(
            "autosubs.audio.asyncio.to_thread",
            new_callable=AsyncMock,
            return_value=(mock_ytdlp, mock_ffmpeg),
        ), patch("autosubs.audio.threading.Thread") as mock_thread:
            mock_thread.return_value = MagicMock()
            await seg.start_live("https://youtube.com/watch?v=test")

        assert audio_dir.exists()
        assert seg._ytdlp_popen is mock_ytdlp
        assert seg._ffmpeg_popen is mock_ffmpeg
        assert seg._chunk_queue is not None

    async def test_poll_chunks_vod_reads_from_process(self, tmp_path):
        """poll_chunks en modo VOD lee de process.stdout y usa VAD."""
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir(parents=True)

        seg = AudioSegmenter("https://audio.url", audio_dir, 5)
        seg.process = AsyncMock()
        seg.process.returncode = 0

        mock_chunk = AudioChunk(path=audio_dir / "chunk_0000.wav", duration=4.5)

        # Simular que stdout retorna datos y luego EOF
        pcm_data = b"\x00" * 32000  # 1s de audio
        seg.process.stdout = AsyncMock()
        seg.process.stdout.read = AsyncMock(side_effect=[pcm_data, b""])
        seg.process.stderr = AsyncMock()
        seg.process.stderr.read = AsyncMock(return_value=b"")

        with patch("autosubs.audio.AudioSegmenter._make_vad_segmenter") as mock_vad:
            mock_segmenter = MagicMock()
            mock_segmenter.feed.return_value = [mock_chunk]
            mock_segmenter.flush_remaining.return_value = None
            mock_vad.return_value = mock_segmenter

            queue: asyncio.Queue = asyncio.Queue()
            await seg.poll_chunks(queue)

        items = []
        while not queue.empty():
            items.append(await queue.get())

        assert len(items) == 2
        assert items[0] is mock_chunk
        assert items[1] is None

    async def test_poll_chunks_sends_sentinel_when_no_data(self, tmp_path):
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir(parents=True)

        seg = AudioSegmenter("https://audio.url", audio_dir, 10)
        seg.process = AsyncMock()
        seg.process.returncode = 0
        seg.process.stdout = AsyncMock()
        seg.process.stdout.read = AsyncMock(return_value=b"")
        seg.process.stderr = AsyncMock()
        seg.process.stderr.read = AsyncMock(return_value=b"")

        with patch("autosubs.audio.AudioSegmenter._make_vad_segmenter") as mock_vad:
            mock_segmenter = MagicMock()
            mock_segmenter.flush_remaining.return_value = None
            mock_vad.return_value = mock_segmenter

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

        with patch("autosubs.audio.asyncio.to_thread", new_callable=AsyncMock):
            await seg.stop()

        mock_ffmpeg.terminate.assert_called_once()
        mock_ytdlp.terminate.assert_called_once()

    async def test_pcm_callback_invoked_vod(self, tmp_path):
        """pcm_callback se invoca con cada bloque de PCM leido en modo VOD."""
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir(parents=True)

        received: list[bytes] = []

        def callback(data: bytes) -> None:
            received.append(data)

        seg = AudioSegmenter(
            "https://audio.url", audio_dir, 5,
            pcm_callback=callback,
        )
        seg.process = AsyncMock()
        seg.process.returncode = 0

        pcm_data = b"\x00" * 32000
        seg.process.stdout = AsyncMock()
        seg.process.stdout.read = AsyncMock(side_effect=[pcm_data, b""])
        seg.process.stderr = AsyncMock()
        seg.process.stderr.read = AsyncMock(return_value=b"")

        with patch("autosubs.audio.AudioSegmenter._make_vad_segmenter") as mock_vad:
            mock_segmenter = MagicMock()
            mock_segmenter.feed.return_value = []
            mock_segmenter.flush_remaining.return_value = None
            mock_vad.return_value = mock_segmenter

            queue: asyncio.Queue = asyncio.Queue()
            await seg.poll_chunks(queue)

        assert len(received) == 1
        assert received[0] == pcm_data

    def test_pcm_callback_none_by_default(self, tmp_path):
        seg = AudioSegmenter("https://audio.url", tmp_path, 5)
        assert seg._pcm_callback is None
