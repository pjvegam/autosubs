from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from autosubs.transcribe import TranscriptSegment, transcribe_verbose


def _make_process(stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0):
    """Crea un mock de asyncio.subprocess.Process."""
    proc = AsyncMock()
    proc.communicate.return_value = (stdout, stderr)
    proc.returncode = returncode
    proc.wait = AsyncMock(return_value=returncode)
    return proc


class TestDownloadAudio:
    async def test_downloads_audio_only(self, tmp_path):
        from autosubs.stream import download_audio

        output = str(tmp_path / "audio.mp3")
        Path(output).write_bytes(b"fake mp3 data" * 1000)

        proc = _make_process(returncode=0)
        with patch("autosubs.stream.asyncio.create_subprocess_exec", return_value=proc):
            await download_audio("https://example.com", output)

    async def test_raises_on_failure(self, tmp_path):
        from autosubs.stream import download_audio

        output = str(tmp_path / "audio.mp3")
        proc = _make_process(returncode=1)
        with (
            patch("autosubs.stream.asyncio.create_subprocess_exec", return_value=proc),
            pytest.raises(RuntimeError, match="yt-dlp audio download fallo"),
        ):
            await download_audio("https://example.com", output)

    async def test_raises_when_file_missing(self, tmp_path):
        from autosubs.stream import download_audio

        output = str(tmp_path / "nonexistent.mp3")
        proc = _make_process(returncode=0)
        with (
            patch("autosubs.stream.asyncio.create_subprocess_exec", return_value=proc),
            pytest.raises(RuntimeError, match="Archivo no encontrado"),
        ):
            await download_audio("https://example.com", output)

    async def test_uses_ba_format(self, tmp_path):
        from autosubs.stream import download_audio

        output = str(tmp_path / "audio.mp3")
        Path(output).write_bytes(b"fake")

        proc = _make_process(returncode=0)
        mock_target = "autosubs.stream.asyncio.create_subprocess_exec"
        with patch(mock_target, return_value=proc) as mock_exec:
            await download_audio("https://example.com", output)

        cmd = mock_exec.call_args[0]
        assert "-f" in cmd
        idx = cmd.index("-f")
        assert cmd[idx + 1] == "ba"


class TestSplitAudio:
    async def test_splits_audio_into_chunks(self, tmp_path):
        from autosubs.stream import split_audio

        input_file = str(tmp_path / "audio.mp3")
        output_dir = str(tmp_path / "chunks")

        chunks_path = Path(output_dir)
        chunks_path.mkdir(parents=True)
        (chunks_path / "chunk_000.mp3").write_bytes(b"chunk0")
        (chunks_path / "chunk_001.mp3").write_bytes(b"chunk1")

        proc = _make_process(returncode=0)
        with patch("autosubs.stream.asyncio.create_subprocess_exec", return_value=proc):
            result = await split_audio(input_file, output_dir, chunk_seconds=1200)

        assert len(result) == 2
        assert all(isinstance(p, Path) for p in result)
        assert result[0].name == "chunk_000.mp3"
        assert result[1].name == "chunk_001.mp3"

    async def test_raises_on_ffmpeg_failure(self, tmp_path):
        from autosubs.stream import split_audio

        proc = _make_process(returncode=1)
        with (
            patch("autosubs.stream.asyncio.create_subprocess_exec", return_value=proc),
            pytest.raises(RuntimeError, match="ffmpeg split fallo"),
        ):
            await split_audio(
                str(tmp_path / "audio.mp3"),
                str(tmp_path / "chunks"),
            )

    async def test_uses_correct_segment_time(self, tmp_path):
        from autosubs.stream import split_audio

        chunks_path = tmp_path / "chunks"
        chunks_path.mkdir()

        proc = _make_process(returncode=0)
        mock_target = "autosubs.stream.asyncio.create_subprocess_exec"
        with patch(mock_target, return_value=proc) as mock_exec:
            await split_audio(
                str(tmp_path / "a.mp3"), str(chunks_path), chunk_seconds=600,
            )

        cmd = mock_exec.call_args[0]
        assert "-segment_time" in cmd
        idx = cmd.index("-segment_time")
        assert cmd[idx + 1] == "600"


class TestTranscriptSegment:
    def test_dataclass_fields(self):
        seg = TranscriptSegment(text="Hello", start=0.0, end=5.0)
        assert seg.text == "Hello"
        assert seg.start == 0.0
        assert seg.end == 5.0


class TestTranscribeVerbose:
    async def test_returns_segments(self, tmp_path):
        audio_file = tmp_path / "chunk_000.mp3"
        audio_file.write_bytes(b"fake audio")

        mock_response = MagicMock()
        mock_response.segments = [
            {"text": "Hello world", "start": 0.0, "end": 3.5},
            {"text": "How are you", "start": 3.5, "end": 7.0},
        ]

        client = AsyncMock()
        client.audio.transcriptions.create.return_value = mock_response

        result = await transcribe_verbose(client, audio_file, "en")

        assert result is not None
        assert len(result) == 2
        assert result[0].text == "Hello world"
        assert result[0].start == 0.0
        assert result[0].end == 3.5
        assert result[1].text == "How are you"

    async def test_filters_hallucinations(self, tmp_path):
        audio_file = tmp_path / "chunk_000.mp3"
        audio_file.write_bytes(b"fake audio")

        mock_response = MagicMock()
        mock_response.segments = [
            {"text": "Good text", "start": 0.0, "end": 3.0},
            {"text": ".", "start": 3.0, "end": 4.0},
            {"text": " ", "start": 4.0, "end": 5.0},
        ]

        client = AsyncMock()
        client.audio.transcriptions.create.return_value = mock_response

        result = await transcribe_verbose(client, audio_file, "en")

        assert result is not None
        assert len(result) == 1
        assert result[0].text == "Good text"

    async def test_returns_none_on_all_hallucinations(self, tmp_path):
        audio_file = tmp_path / "chunk_000.mp3"
        audio_file.write_bytes(b"fake audio")

        mock_response = MagicMock()
        mock_response.segments = [
            {"text": ".", "start": 0.0, "end": 1.0},
        ]

        client = AsyncMock()
        client.audio.transcriptions.create.return_value = mock_response

        result = await transcribe_verbose(client, audio_file, "en")
        assert result is None

    async def test_returns_none_on_api_failure(self, tmp_path):
        audio_file = tmp_path / "chunk_000.mp3"
        audio_file.write_bytes(b"fake audio")

        client = AsyncMock()
        client.audio.transcriptions.create.side_effect = RuntimeError("API error")

        with patch("autosubs.retry.BASE_DELAY", 0.01):
            result = await transcribe_verbose(client, audio_file, "en")
        assert result is None

    async def test_uses_correct_params(self, tmp_path):
        audio_file = tmp_path / "chunk_000.mp3"
        audio_file.write_bytes(b"fake audio")

        mock_response = MagicMock()
        mock_response.segments = [
            {"text": "Test", "start": 0.0, "end": 1.0},
        ]

        client = AsyncMock()
        client.audio.transcriptions.create.return_value = mock_response

        await transcribe_verbose(client, audio_file, "ja")

        call_kwargs = client.audio.transcriptions.create.call_args.kwargs
        assert call_kwargs["model"] == "whisper-1"
        assert call_kwargs["language"] == "ja"
        assert call_kwargs["response_format"] == "verbose_json"

    async def test_handles_object_style_segments(self, tmp_path):
        audio_file = tmp_path / "chunk_000.mp3"
        audio_file.write_bytes(b"fake audio")

        seg_obj = MagicMock()
        seg_obj.text = "Object style text"
        seg_obj.start = 1.0
        seg_obj.end = 4.0

        mock_response = MagicMock()
        mock_response.segments = [seg_obj]

        client = AsyncMock()
        client.audio.transcriptions.create.return_value = mock_response

        result = await transcribe_verbose(client, audio_file, "en")

        assert result is not None
        assert len(result) == 1
        assert result[0].text == "Object style text"
