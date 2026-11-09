import asyncio
import os
import shutil
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from autosubs.audio import (
    TRANSCRIPTION_AUDIO_FILTER,
    AudioSegmenter,
    ensure_ca_bundle,
)
from autosubs.vad import AudioChunk


class TestTimelineIntegrity:
    """Los timestamps salen de sumar duraciones de chunks, asi que el PCM tiene
    que cubrir el timeline completo: un hueco perdido adelanta todo lo que sigue."""

    def test_filter_fills_timeline_gaps(self):
        assert TRANSCRIPTION_AUDIO_FILTER.startswith("aresample=async=1,")

    def test_copyts_presente(self):
        """Sin -copyts el aresample de arriba no ve los huecos y no rellena."""
        seg = AudioSegmenter("/tmp/v.mkv", Path("/tmp/x"))
        assert "-copyts" in seg._ffmpeg_continuous_args("/tmp/v.mkv")

    @pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="necesita ffmpeg")
    def test_hueco_de_pts_se_rellena_de_verdad(self, tmp_path):
        """El test de arriba mira la cadena; este mide el PCM que sale.

        Dos TS de 10s pegados a nivel de bytes, el segundo con PTS +40s: es
        exactamente lo que ve ffmpeg cuando un segmento HLS no se descarga.
        """
        partes = []
        for i, offset in enumerate((0, 40)):
            parte = tmp_path / f"p{i}.ts"
            subprocess.run(
                ["ffmpeg", "-v", "error", "-y",
                 "-f", "lavfi", "-i", "sine=f=440:d=10:r=16000",
                 "-c:a", "aac", "-output_ts_offset", str(offset), str(parte)],
                check=True,
            )
            partes.append(parte.read_bytes())
        gap_ts = tmp_path / "gap.ts"
        gap_ts.write_bytes(b"".join(partes))

        seg = AudioSegmenter(str(gap_ts), tmp_path)
        cmd = seg._ffmpeg_continuous_args(str(gap_ts))
        cmd[cmd.index("pipe:1")] = str(raw := tmp_path / "out.raw")
        subprocess.run(cmd, check=True, capture_output=True)

        # 16kHz mono s16le = 32000 bytes/s. 10 + 30 de hueco + 10 = 50s
        segundos = raw.stat().st_size / 32000
        assert segundos == pytest.approx(50, abs=1.5), (
            f"el hueco no se relleno: {segundos:.1f}s de PCM en vez de ~50s"
        )

    def test_vod_desde_cero_rellena_el_hueco_de_delante(self):
        """Si la pista empieza tarde, sin esto todo sale adelantado."""
        seg = AudioSegmenter("/tmp/v.mkv", Path("/tmp/x"))
        af = seg._ffmpeg_continuous_args("/tmp/v.mkv")
        assert "first_pts=0" in af[af.index("-af") + 1]

    def test_con_ss_no_rellena_desde_cero(self):
        """Buscando un punto, el PCM arranca ahi: rellenar serian 900s de silencio."""
        seg = AudioSegmenter("/tmp/v.mkv", Path("/tmp/x"), start_time=900)
        af = seg._ffmpeg_continuous_args("/tmp/v.mkv")
        assert "first_pts=0" not in af[af.index("-af") + 1]

    def test_en_vivo_no_rellena_desde_cero(self):
        """En vivo se entra por el extremo; delante esta toda la transmision."""
        seg = AudioSegmenter("https://live", Path("/tmp/x"), live=True)
        af = seg._ffmpeg_continuous_args("https://live")
        assert "first_pts=0" not in af[af.index("-af") + 1]

    @pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="necesita ffmpeg")
    def test_arranque_tardio_de_la_pista_se_rellena_de_verdad(self, tmp_path):
        """El hueco inicial es el que mas desincroniza, y el que no se veia.

        Un TS de 10s cuyo primer PTS es 40s: es lo que ve ffmpeg cuando la
        pista de audio no empieza donde empieza el video. Medido en un VOD
        real del usuario ese hueco era de 595s, o sea 10 minutos de desfase.
        """
        tarde = tmp_path / "tarde.ts"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y",
             "-f", "lavfi", "-i", "sine=f=440:d=10:r=16000",
             "-c:a", "aac", "-output_ts_offset", "40", str(tarde)],
            check=True,
        )

        seg = AudioSegmenter(str(tarde), tmp_path)
        cmd = seg._ffmpeg_continuous_args(str(tarde))
        cmd[cmd.index("pipe:1")] = str(raw := tmp_path / "tarde.raw")
        subprocess.run(cmd, check=True, capture_output=True)

        segundos = raw.stat().st_size / 32000
        assert segundos == pytest.approx(50, abs=1.5), (
            f"el arranque no se relleno: {segundos:.1f}s de PCM en vez de ~50s"
        )

    def test_http_source_reconnects_instead_of_skipping(self):
        seg = AudioSegmenter("https://example.com/stream.m3u8", Path("/tmp/x"))
        cmd = seg._ffmpeg_continuous_args("https://example.com/stream.m3u8")
        assert "-reconnect" in cmd
        assert "-reconnect_streamed" in cmd

    def test_local_file_gets_no_reconnect_flags(self):
        seg = AudioSegmenter("/tmp/video.mkv", Path("/tmp/x"))
        cmd = seg._ffmpeg_continuous_args("/tmp/video.mkv")
        assert "-reconnect" not in cmd

    def test_seek_precedes_input(self):
        seg = AudioSegmenter("/tmp/v.mkv", Path("/tmp/x"), start_time=900)
        cmd = seg._ffmpeg_continuous_args("/tmp/v.mkv")
        assert cmd.index("-ss") < cmd.index("-i")
        assert cmd[cmd.index("-ss") + 1] == "900"


class TestEnsureCaBundle:
    def test_sets_ssl_cert_file_from_certifi(self, monkeypatch):
        import certifi

        monkeypatch.delenv("SSL_CERT_FILE", raising=False)
        ensure_ca_bundle()
        assert Path(os.environ["SSL_CERT_FILE"]) == Path(certifi.where())

    def test_bundle_file_actually_exists(self, monkeypatch):
        monkeypatch.delenv("SSL_CERT_FILE", raising=False)
        ensure_ca_bundle()
        assert Path(os.environ["SSL_CERT_FILE"]).is_file()

    def test_does_not_override_user_setting(self, monkeypatch):
        monkeypatch.setenv("SSL_CERT_FILE", "/ruta/del/usuario.pem")
        ensure_ca_bundle()
        assert os.environ["SSL_CERT_FILE"] == "/ruta/del/usuario.pem"

    def test_survives_missing_certifi(self, monkeypatch):
        monkeypatch.delenv("SSL_CERT_FILE", raising=False)
        with patch.dict("sys.modules", {"certifi": None}):
            ensure_ca_bundle()  # no debe lanzar
        assert "SSL_CERT_FILE" not in os.environ


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
