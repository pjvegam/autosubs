import json
from unittest.mock import AsyncMock, patch

import pytest

from livesubs.stream import (
    _ytdlp_base_args,
    get_audio_url,
    get_stream_info,
    get_stream_urls,
    is_live,
    set_cookies_browser,
    ytdlp_pipe_audio_cmd,
)


def _make_process(stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0):
    """Crea un mock de asyncio.subprocess.Process."""
    proc = AsyncMock()
    proc.communicate.return_value = (stdout, stderr)
    proc.returncode = returncode
    return proc


class TestCookiesBrowser:
    def setup_method(self):
        set_cookies_browser("")

    def teardown_method(self):
        set_cookies_browser("")

    def test_base_args_without_cookies(self):
        args = _ytdlp_base_args()
        assert "--cookies-from-browser" not in args

    def test_base_args_with_cookies(self):
        set_cookies_browser("firefox")
        args = _ytdlp_base_args()
        idx = args.index("--cookies-from-browser")
        assert args[idx + 1] == "firefox"

    def test_cookies_in_pipe_cmd(self):
        set_cookies_browser("chrome")
        cmd = ytdlp_pipe_audio_cmd("https://example.com")
        idx = cmd.index("--cookies-from-browser")
        assert cmd[idx + 1] == "chrome"

    def test_set_cookies_browser_overrides(self):
        set_cookies_browser("firefox")
        set_cookies_browser("chrome")
        args = _ytdlp_base_args()
        idx = args.index("--cookies-from-browser")
        assert args[idx + 1] == "chrome"


class TestYtdlpPipeAudioCmd:
    def test_returns_correct_format(self):
        cmd = ytdlp_pipe_audio_cmd("https://youtube.com/watch?v=test")
        assert cmd[0] == "yt-dlp"
        assert "-f" in cmd
        assert "wa*" in cmd
        assert "-o" in cmd
        assert "-" in cmd
        assert "https://youtube.com/watch?v=test" in cmd


class TestGetStreamInfo:
    async def test_returns_parsed_json(self):
        info = {"title": "Test Video", "is_live": False}
        proc = _make_process(stdout=json.dumps(info).encode())
        with patch("livesubs.stream.asyncio.create_subprocess_exec", return_value=proc):
            result = await get_stream_info("https://example.com")
        assert result == info

    async def test_raises_on_failure(self):
        proc = _make_process(stderr=b"ERROR: video not found", returncode=1)
        with (
            patch("livesubs.stream.asyncio.create_subprocess_exec", return_value=proc),
            pytest.raises(RuntimeError, match="yt-dlp fallo"),
        ):
            await get_stream_info("https://example.com")


class TestGetStreamUrls:
    async def test_dash_returns_three_urls(self):
        """DASH: video-only + audio-only separados."""
        video_proc = _make_process(stdout=b"https://video.url\n")
        audio_proc = _make_process(stdout=b"https://audio.url\n")

        call_count = 0

        async def mock_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return video_proc if call_count == 1 else audio_proc

        with patch("livesubs.stream.asyncio.create_subprocess_exec", side_effect=mock_create):
            video_url, mpv_audio, ffmpeg_audio = await get_stream_urls("https://example.com")

        assert video_url == "https://video.url"
        assert mpv_audio == "https://audio.url"
        assert ffmpeg_audio == "https://audio.url"

    async def test_hls_fallback_uses_light_stream_for_ffmpeg(self):
        """HLS: ba falla, usa bv* + wa* (stream ligero para ffmpeg)."""
        call_count = 0

        async def mock_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            cmd_args = args
            if call_count <= 2:
                # bv* ok, ba falla
                if "ba" in cmd_args:
                    return _make_process(stderr=b"no audio-only", returncode=1)
                return _make_process(stdout=b"https://video.url\n")
            # Segundo intento: bv* y wa*
            if "wa*" in cmd_args:
                return _make_process(stdout=b"https://light.url\n")
            return _make_process(stdout=b"https://video.url\n")

        with patch("livesubs.stream.asyncio.create_subprocess_exec", side_effect=mock_create):
            video_url, mpv_audio, ffmpeg_audio = await get_stream_urls("https://example.com")

        assert video_url == "https://video.url"
        assert mpv_audio is None
        assert ffmpeg_audio == "https://light.url"

    async def test_full_fallback_to_combined(self):
        """Si todo falla, usa b* para todo."""
        call_count = 0

        async def mock_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 4:
                # bv* + ba falla, bv* + wa* falla
                return _make_process(stderr=b"format error", returncode=1)
            # Fallback b*
            return _make_process(stdout=b"https://combined.url\n")

        with patch("livesubs.stream.asyncio.create_subprocess_exec", side_effect=mock_create):
            video_url, mpv_audio, ffmpeg_audio = await get_stream_urls("https://example.com")

        assert video_url == "https://combined.url"
        assert mpv_audio is None
        assert ffmpeg_audio == "https://combined.url"


class TestGetAudioUrl:
    async def test_returns_lightest_audio(self):
        """Usa wa* (audio mas ligero) primero."""
        proc = _make_process(stdout=b"https://light-audio.url\n")
        with patch("livesubs.stream.asyncio.create_subprocess_exec", return_value=proc):
            result = await get_audio_url("https://example.com")
        assert result == "https://light-audio.url"

    async def test_fallback_to_ba(self):
        """Si wa* falla, usa ba."""
        call_count = 0

        async def mock_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_process(stderr=b"no wa*", returncode=1)
            return _make_process(stdout=b"https://audio.url\n")

        with patch("livesubs.stream.asyncio.create_subprocess_exec", side_effect=mock_create):
            result = await get_audio_url("https://example.com")
        assert result == "https://audio.url"

    async def test_fallback_to_combined(self):
        """Si todo falla, usa b* combinado."""
        call_count = 0

        async def mock_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return _make_process(stderr=b"error", returncode=1)
            return _make_process(stdout=b"https://combined.url\n")

        with patch("livesubs.stream.asyncio.create_subprocess_exec", side_effect=mock_create):
            result = await get_audio_url("https://example.com")
        assert result == "https://combined.url"

    async def test_raises_when_all_fail(self):
        """RuntimeError si ningun formato funciona."""
        proc = _make_process(stderr=b"error", returncode=1)
        with (
            patch("livesubs.stream.asyncio.create_subprocess_exec", return_value=proc),
            pytest.raises(RuntimeError, match="No se pudo obtener URL de audio"),
        ):
            await get_audio_url("https://example.com")


class TestIsLive:
    async def test_live_stream(self):
        info = {"is_live": True}
        proc = _make_process(stdout=json.dumps(info).encode())
        with patch("livesubs.stream.asyncio.create_subprocess_exec", return_value=proc):
            assert await is_live("https://example.com") is True

    async def test_vod(self):
        info = {"is_live": False}
        proc = _make_process(stdout=json.dumps(info).encode())
        with patch("livesubs.stream.asyncio.create_subprocess_exec", return_value=proc):
            assert await is_live("https://example.com") is False

    async def test_missing_field_defaults_to_false(self):
        info = {"title": "Test"}
        proc = _make_process(stdout=json.dumps(info).encode())
        with patch("livesubs.stream.asyncio.create_subprocess_exec", return_value=proc):
            assert await is_live("https://example.com") is False

    async def test_error_defaults_to_false(self):
        proc = _make_process(stderr=b"error", returncode=1)
        with patch("livesubs.stream.asyncio.create_subprocess_exec", return_value=proc):
            assert await is_live("https://example.com") is False
