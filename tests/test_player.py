import json
from unittest.mock import AsyncMock, MagicMock, patch

from autosubs.player import MPVPlayer


class TestMPVPlayerInit:
    def test_initial_state(self):
        player = MPVPlayer(r"\\.\pipe\mpvsocket")
        assert player.ipc_pipe == r"\\.\pipe\mpvsocket"
        assert player.process is None
        assert player._pipe_handle is None
        assert player._request_id == 0
        assert player._connected is False

    def test_request_id_is_instance_scoped(self):
        p1 = MPVPlayer("pipe1")
        p2 = MPVPlayer("pipe2")
        assert p1._next_request_id() == 1
        assert p1._next_request_id() == 2
        assert p2._next_request_id() == 1  # independiente


class TestMPVPlayerIsRunning:
    def test_no_process(self):
        player = MPVPlayer("pipe")
        assert player.is_running() is False

    def test_running_process(self):
        player = MPVPlayer("pipe")
        player.process = MagicMock()
        player.process.poll.return_value = None
        assert player.is_running() is True

    def test_finished_process(self):
        player = MPVPlayer("pipe")
        player.process = MagicMock()
        player.process.poll.return_value = 0
        assert player.is_running() is False


class TestMPVPlayerStart:
    async def test_launches_mpv(self):
        player = MPVPlayer(r"\\.\pipe\mpvsocket")
        mock_popen = MagicMock()

        with patch("autosubs.player.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = mock_popen
            await player.start("https://video.url")

        assert player.process is mock_popen
        call_args = mock_thread.call_args
        cmd = call_args[0][1]
        assert cmd[0] == "mpv"
        assert r"--input-ipc-server=\\.\pipe\mpvsocket" in cmd
        assert "--force-window=yes" in cmd
        assert "--sub-visibility=yes" in cmd
        assert "https://video.url" in cmd

    async def test_includes_audio_file_when_provided(self):
        player = MPVPlayer("pipe")

        with patch("autosubs.player.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = MagicMock()
            await player.start("https://video.url", audio_url="https://audio.url")

        cmd = mock_thread.call_args[0][1]
        assert "--audio-file=https://audio.url" in cmd

    async def test_start_time_becomes_mpv_start_flag(self):
        # Se pasa como --start y no como seek por IPC: contra un stream HLS el
        # seek llega antes de que MPV tenga el indice y se pierde
        player = MPVPlayer("pipe")

        with patch("autosubs.player.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = MagicMock()
            await player.start("https://video.url", start_time=1113)

        cmd = mock_thread.call_args[0][1]
        assert "--start=1113" in cmd

    async def test_end_time_becomes_mpv_end_flag(self):
        player = MPVPlayer("pipe")

        with patch("autosubs.player.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = MagicMock()
            await player.start("https://video.url", start_time=2700, end_time=3300)

        cmd = mock_thread.call_args[0][1]
        assert "--start=2700" in cmd
        assert "--end=3300" in cmd

    async def test_no_end_flag_without_end_time(self):
        player = MPVPlayer("pipe")

        with patch("autosubs.player.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = MagicMock()
            await player.start("https://video.url", start_time=2700)

        cmd = mock_thread.call_args[0][1]
        assert not any(c.startswith("--end=") for c in cmd)

    async def test_no_start_flag_without_start_time(self):
        player = MPVPlayer("pipe")

        with patch("autosubs.player.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = MagicMock()
            await player.start("https://video.url")

        cmd = mock_thread.call_args[0][1]
        assert not any(c.startswith("--start=") for c in cmd)


class TestMPVPlayerSendCommand:
    async def test_send_command_returns_parsed_response(self):
        player = MPVPlayer("pipe")
        response = {"error": "success", "data": 42.5, "request_id": 1}

        pipe_mock = MagicMock()
        pipe_mock.readline.return_value = json.dumps(response).encode() + b"\n"
        pipe_mock.write = MagicMock()
        pipe_mock.flush = MagicMock()
        player._pipe_handle = pipe_mock

        with patch("autosubs.player.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.side_effect = lambda fn: fn()
            result = await player._send_command(["get_property", "time-pos"])

        assert result == response

    async def test_send_command_skips_events(self):
        player = MPVPlayer("pipe")
        event = json.dumps({"event": "audio-reconfig"}).encode() + b"\n"
        response = json.dumps({"error": "success", "request_id": 1}).encode() + b"\n"

        pipe_mock = MagicMock()
        # Primer readline devuelve evento, segundo devuelve respuesta
        pipe_mock.readline.side_effect = [event, response]
        pipe_mock.write = MagicMock()
        pipe_mock.flush = MagicMock()
        player._pipe_handle = pipe_mock

        with patch("autosubs.player.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.side_effect = lambda fn: fn()
            result = await player._send_command(["sub-add", "/tmp/test.srt"])

        assert result == {"error": "success", "request_id": 1}

    async def test_send_command_returns_none_when_not_connected(self):
        player = MPVPlayer("pipe")
        result = await player._send_command(["test"])
        assert result is None


class TestMPVPlayerShowText:
    async def test_show_text_sends_correct_command(self):
        player = MPVPlayer("pipe")
        player._send_command = AsyncMock(return_value={"error": "success"})
        await player.show_text("Hola mundo", 5000)
        player._send_command.assert_called_once_with(
            ["expand-properties", "show-text", "Hola mundo", "5000"]
        )


class TestMPVPlayerGetTimePos:
    async def test_returns_time_on_success(self):
        player = MPVPlayer("pipe")
        player._send_command = AsyncMock(
            return_value={"error": "success", "data": 42.5}
        )
        result = await player.get_time_pos()
        assert result == 42.5

    async def test_returns_none_on_error(self):
        player = MPVPlayer("pipe")
        player._send_command = AsyncMock(
            return_value={"error": "property not found"}
        )
        result = await player.get_time_pos()
        assert result is None

    async def test_returns_none_on_no_response(self):
        player = MPVPlayer("pipe")
        player._send_command = AsyncMock(return_value=None)
        result = await player.get_time_pos()
        assert result is None


class TestMPVPlayerClose:
    async def test_close_cleans_up(self):
        player = MPVPlayer("pipe")
        player._pipe_handle = MagicMock()
        player.process = MagicMock()
        player.process.poll.return_value = None

        with patch("autosubs.player.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = None
            await player.close()

        assert player._pipe_handle is None
        player.process.terminate.assert_called_once()

    async def test_close_without_connection(self):
        player = MPVPlayer("pipe")
        await player.close()  # No deberia lanzar error

    async def test_wait_returns_negative_without_process(self):
        player = MPVPlayer("pipe")
        result = await player.wait()
        assert result == -1


class TestArranqueEnPausa:
    """MPV reproduce desde que se lanza; el pipeline tarda en estar listo."""

    async def _cmd(self, **kwargs):
        player = MPVPlayer("pipe")
        with patch(
            "autosubs.player.asyncio.to_thread", new_callable=AsyncMock,
        ) as mock_thread:
            mock_thread.return_value = MagicMock()
            await player.start("https://video.url", **kwargs)
        return mock_thread.call_args[0][1]

    async def test_sin_pausa_no_agrega_la_opcion(self):
        assert "--pause=yes" not in await self._cmd()

    async def test_con_pausa_arranca_detenido(self):
        assert "--pause=yes" in await self._cmd(pause=True)

    async def test_la_pausa_convive_con_start_y_end(self):
        cmd = await self._cmd(pause=True, start_time=90, end_time=120)
        assert "--pause=yes" in cmd
        assert "--start=90" in cmd
        assert "--end=120" in cmd

    async def test_resume_quita_la_pausa(self):
        player = MPVPlayer("pipe")
        player._send_command = AsyncMock(return_value={"error": "success"})
        await player.resume()
        player._send_command.assert_awaited_once_with(["set", "pause", "no"])
