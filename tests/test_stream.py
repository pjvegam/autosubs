import asyncio
import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from autosubs.stream import (
    FRAGMENT_CONCURRENCY,
    _ytdlp_base_args,
    download_vod,
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


class TestDownloadVodCancel:
    """Cancelar la task debe matar a yt-dlp, no dejarlo descargando huerfano."""

    @staticmethod
    def _process(*, dies_on_terminate: bool = True) -> MagicMock:
        """yt-dlp descargando: el primer wait() no retorna nunca."""
        proc = MagicMock()
        proc.returncode = None
        calls = {"n": 0}

        async def _wait() -> int:
            calls["n"] += 1
            if calls["n"] == 1 or not dies_on_terminate:
                await asyncio.Event().wait()
            proc.returncode = 0
            return 0

        proc.wait = _wait
        return proc

    @staticmethod
    async def _cancel_during_download(proc: MagicMock, tmp_path) -> None:
        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            task = asyncio.create_task(
                download_vod("https://youtu.be/x", str(tmp_path / "v.mkv")),
            )
            await asyncio.sleep(0)  # deja que arranque y quede descargando
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    async def test_terminates_ytdlp_on_cancel(self, tmp_path):
        proc = self._process()
        await self._cancel_during_download(proc, tmp_path)
        proc.terminate.assert_called_once()
        proc.kill.assert_not_called()

    async def test_kills_ytdlp_if_terminate_is_ignored(self, tmp_path):
        proc = self._process(dies_on_terminate=False)

        async def _expire(coro, **_kwargs):
            coro.close()  # sin esto queda una corrutina sin await
            raise TimeoutError

        with patch("autosubs.stream.asyncio.wait_for", _expire):
            await self._cancel_during_download(proc, tmp_path)

        proc.terminate.assert_called_once()
        proc.kill.assert_called_once()

    async def test_survives_process_already_gone(self, tmp_path):
        proc = self._process()
        proc.terminate.side_effect = ProcessLookupError
        # No debe propagar ProcessLookupError, solo CancelledError
        await self._cancel_during_download(proc, tmp_path)


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


class TestConcurrenciaDeFragmentos:
    """Bajar fragmentos de uno en uno hace la descarga inutilizable.

    Medido el 2026-10-21 sobre un VOD de 5h que solo ofrece HLS: 126 KB/s con
    -N 1 contra 706 con -N 8. Las 17.5 GiB del 1080p60 pasan de 38 horas a 7.
    Los argumentos base NO lo llevan: en las extracciones (-j, -g) no pinta
    nada y solo ensucia el comando.
    """

    async def test_el_vod_baja_fragmentos_en_paralelo(self):
        proc = AsyncMock()
        proc.wait.return_value = 0
        with (
            patch("autosubs.stream.asyncio.create_subprocess_exec", return_value=proc) as spawn,
            patch("autosubs.stream.Path") as fake_path,
        ):
            fake_path.return_value.exists.return_value = True
            fake_path.return_value.stat.return_value.st_size = 1024
            await download_vod("https://example.com", "out.mkv")
        cmd = spawn.call_args[0]
        assert cmd[cmd.index("-N") + 1] == str(FRAGMENT_CONCURRENCY)
        assert FRAGMENT_CONCURRENCY > 1

    def test_las_extracciones_no_lo_llevan(self):
        assert "-N" not in _ytdlp_base_args()


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
        with patch("autosubs.stream.asyncio.create_subprocess_exec", return_value=proc):
            result = await get_stream_info("https://example.com")
        assert result == info

    async def test_raises_on_failure(self):
        proc = _make_process(stderr=b"ERROR: video not found", returncode=1)
        with (
            patch("autosubs.stream.asyncio.create_subprocess_exec", return_value=proc),
            pytest.raises(RuntimeError, match="yt-dlp metadata fallo"),
        ):
            await get_stream_info("https://example.com")


class TestAvisoDeExtraccionLenta:
    """Una extraccion de minutos sin decir nada parece un cuelgue.

    Medido el 2026-10-21: YouTube estreno el reproductor 55de7e60 y la primera
    extraccion tardo 6m09s resolviendo su reto de JS. El usuario solo vio el
    log de los modelos y creyo que el programa estaba muerto.
    """

    @staticmethod
    def _lento(demora: float):
        """yt-dlp que tarda `demora` en contestar."""
        proc = AsyncMock()
        proc.returncode = 0

        async def _communicate() -> tuple[bytes, bytes]:
            await asyncio.sleep(demora)
            return json.dumps({"id": "x"}).encode(), b""

        proc.communicate = _communicate
        return proc

    async def test_avisa_cuando_tarda(self, caplog):
        with (
            patch("autosubs.stream.SLOW_EXTRACTION_WARN", 0.01),
            patch(
                "autosubs.stream.asyncio.create_subprocess_exec",
                return_value=self._lento(0.05),
            ),
            caplog.at_level(logging.WARNING, logger="autosubs.stream"),
        ):
            await get_stream_info("https://example.com")
        assert "reto de JS" in caplog.text

    async def test_no_avisa_cuando_es_rapido(self, caplog):
        with (
            patch("autosubs.stream.SLOW_EXTRACTION_WARN", 5.0),
            patch(
                "autosubs.stream.asyncio.create_subprocess_exec",
                return_value=self._lento(0.0),
            ),
            caplog.at_level(logging.WARNING, logger="autosubs.stream"),
        ):
            await get_stream_info("https://example.com")
        assert caplog.text == ""

    async def test_el_aviso_no_sobrevive_a_la_llamada(self):
        """Cancelado el aviso, no queda una task suelta avisando despues."""
        with (
            patch("autosubs.stream.SLOW_EXTRACTION_WARN", 0.05),
            patch(
                "autosubs.stream.asyncio.create_subprocess_exec",
                return_value=self._lento(0.0),
            ),
        ):
            await get_stream_info("https://example.com")
        assert not [t for t in asyncio.all_tasks() if "_warn_slow" in repr(t)]


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

        with patch("autosubs.stream.asyncio.create_subprocess_exec", side_effect=mock_create):
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

        with patch("autosubs.stream.asyncio.create_subprocess_exec", side_effect=mock_create):
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

        with patch("autosubs.stream.asyncio.create_subprocess_exec", side_effect=mock_create):
            video_url, mpv_audio, ffmpeg_audio = await get_stream_urls("https://example.com")

        assert video_url == "https://combined.url"
        assert mpv_audio is None
        assert ffmpeg_audio == "https://combined.url"


class TestGetAudioUrl:
    async def test_returns_lightest_audio(self):
        """Usa wa* (audio mas ligero) primero."""
        proc = _make_process(stdout=b"https://light-audio.url\n")
        with patch("autosubs.stream.asyncio.create_subprocess_exec", return_value=proc):
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

        with patch("autosubs.stream.asyncio.create_subprocess_exec", side_effect=mock_create):
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

        with patch("autosubs.stream.asyncio.create_subprocess_exec", side_effect=mock_create):
            result = await get_audio_url("https://example.com")
        assert result == "https://combined.url"

    async def test_raises_when_all_fail(self):
        """RuntimeError si ningun formato funciona."""
        proc = _make_process(stderr=b"error", returncode=1)
        with (
            patch("autosubs.stream.asyncio.create_subprocess_exec", return_value=proc),
            pytest.raises(RuntimeError, match="No se pudo obtener URL de audio"),
        ):
            await get_audio_url("https://example.com")


class TestIsLive:
    async def test_live_stream(self):
        info = {"is_live": True}
        proc = _make_process(stdout=json.dumps(info).encode())
        with patch("autosubs.stream.asyncio.create_subprocess_exec", return_value=proc):
            assert await is_live("https://example.com") is True

    async def test_vod(self):
        info = {"is_live": False}
        proc = _make_process(stdout=json.dumps(info).encode())
        with patch("autosubs.stream.asyncio.create_subprocess_exec", return_value=proc):
            assert await is_live("https://example.com") is False

    async def test_missing_field_defaults_to_false(self):
        info = {"title": "Test"}
        proc = _make_process(stdout=json.dumps(info).encode())
        with patch("autosubs.stream.asyncio.create_subprocess_exec", return_value=proc):
            assert await is_live("https://example.com") is False

    async def test_error_defaults_to_false(self):
        proc = _make_process(stderr=b"error", returncode=1)
        with patch("autosubs.stream.asyncio.create_subprocess_exec", return_value=proc):
            assert await is_live("https://example.com") is False
