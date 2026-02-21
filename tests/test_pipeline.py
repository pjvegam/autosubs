import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from livesubs.config import Config
from livesubs.pipeline import (
    _monitor_mpv,
    _monitor_overlay,
    _overlay_subtitle_writer,
    _subtitle_writer,
    _transcribe_worker,
    _transcribe_worker_parallel,
    _translate_worker,
)


def _make_config(**kwargs) -> Config:
    defaults = dict(url="https://example.com", api_key="sk-test", segment_duration=5)
    defaults.update(kwargs)
    return Config(**defaults)


class TestTranscribeWorker:
    async def test_processes_chunks_and_sends_sentinel(self):
        config = _make_config()
        client = AsyncMock()

        audio_queue: asyncio.Queue = asyncio.Queue()
        text_queue: asyncio.Queue = asyncio.Queue()

        chunk = Path("/tmp/fake_chunk.wav")

        with patch("livesubs.pipeline.transcribe_audio", new_callable=AsyncMock) as mock_transcribe:
            mock_transcribe.return_value = "texto transcrito"
            with patch.object(Path, "unlink"):
                await audio_queue.put(chunk)
                await audio_queue.put(None)

                await _transcribe_worker(client, config, audio_queue, text_queue, 5)

        item = await text_queue.get()
        assert item == (0, "texto transcrito")
        sentinel = await text_queue.get()
        assert sentinel is None

    async def test_skips_failed_transcription(self):
        config = _make_config()
        client = AsyncMock()

        audio_queue: asyncio.Queue = asyncio.Queue()
        text_queue: asyncio.Queue = asyncio.Queue()

        with patch("livesubs.pipeline.transcribe_audio", new_callable=AsyncMock) as mock_transcribe:
            mock_transcribe.return_value = None
            with patch.object(Path, "unlink"):
                await audio_queue.put(Path("/tmp/chunk.wav"))
                await audio_queue.put(None)

                await _transcribe_worker(client, config, audio_queue, text_queue, 5)

        sentinel = await text_queue.get()
        assert sentinel is None

    async def test_handles_transcription_exception(self):
        config = _make_config()
        client = AsyncMock()

        audio_queue: asyncio.Queue = asyncio.Queue()
        text_queue: asyncio.Queue = asyncio.Queue()

        with patch("livesubs.pipeline.transcribe_audio", new_callable=AsyncMock) as mock_transcribe:
            mock_transcribe.side_effect = RuntimeError("boom")
            with patch.object(Path, "unlink"):
                await audio_queue.put(Path("/tmp/chunk.wav"))
                await audio_queue.put(None)

                await _transcribe_worker(client, config, audio_queue, text_queue, 5)

        sentinel = await text_queue.get()
        assert sentinel is None

    async def test_increments_start_time(self):
        config = _make_config()
        client = AsyncMock()

        audio_queue: asyncio.Queue = asyncio.Queue()
        text_queue: asyncio.Queue = asyncio.Queue()

        with patch("livesubs.pipeline.transcribe_audio", new_callable=AsyncMock) as mock_transcribe:
            mock_transcribe.return_value = "texto"
            with patch.object(Path, "unlink"):
                await audio_queue.put(Path("/tmp/c0.wav"))
                await audio_queue.put(Path("/tmp/c1.wav"))
                await audio_queue.put(Path("/tmp/c2.wav"))
                await audio_queue.put(None)

                await _transcribe_worker(client, config, audio_queue, text_queue, 5)

        items = []
        while not text_queue.empty():
            items.append(await text_queue.get())

        assert items[0] == (0, "texto")
        assert items[1] == (5, "texto")
        assert items[2] == (10, "texto")
        assert items[3] is None


class TestTranslateWorker:
    async def test_translates_and_forwards(self):
        config = _make_config()
        client = AsyncMock()

        text_queue: asyncio.Queue = asyncio.Queue()
        translated_queue: asyncio.Queue = asyncio.Queue()

        with patch("livesubs.pipeline.translate_text", new_callable=AsyncMock) as mock_translate:
            mock_translate.return_value = "traducido"
            await text_queue.put((0, "original"))
            await text_queue.put(None)

            await _translate_worker(client, config, text_queue, translated_queue)

        item = await translated_queue.get()
        assert item == (0, "traducido")
        sentinel = await translated_queue.get()
        assert sentinel is None

    async def test_skips_failed_translation(self):
        config = _make_config()
        client = AsyncMock()

        text_queue: asyncio.Queue = asyncio.Queue()
        translated_queue: asyncio.Queue = asyncio.Queue()

        with patch("livesubs.pipeline.translate_text", new_callable=AsyncMock) as mock_translate:
            mock_translate.return_value = None
            await text_queue.put((0, "original"))
            await text_queue.put(None)

            await _translate_worker(client, config, text_queue, translated_queue)

        sentinel = await translated_queue.get()
        assert sentinel is None

    async def test_handles_translation_exception(self):
        config = _make_config()
        client = AsyncMock()

        text_queue: asyncio.Queue = asyncio.Queue()
        translated_queue: asyncio.Queue = asyncio.Queue()

        with patch("livesubs.pipeline.translate_text", new_callable=AsyncMock) as mock_translate:
            mock_translate.side_effect = RuntimeError("boom")
            await text_queue.put((0, "original"))
            await text_queue.put(None)

            await _translate_worker(client, config, text_queue, translated_queue)

        sentinel = await translated_queue.get()
        assert sentinel is None


class TestSubtitleWriter:
    async def test_writes_subtitle_vod(self, tmp_path):
        config = _make_config()
        srt_path = tmp_path / "test.srt"

        from livesubs.subtitles import SubtitleManager
        sub_manager = SubtitleManager(srt_path)

        player = AsyncMock()
        player.sub_add = AsyncMock()
        player.sub_reload = AsyncMock()

        translated_queue: asyncio.Queue = asyncio.Queue()
        await translated_queue.put((0.0, "Hola mundo"))
        await translated_queue.put(None)

        await _subtitle_writer(config, translated_queue, sub_manager, player, live=False)

        content = srt_path.read_text(encoding="utf-8")
        assert "Hola mundo" in content
        assert "00:00:00,000 --> 00:00:05,000" in content
        # Primer subtitulo usa sub_add, no sub_reload
        player.sub_add.assert_called_once_with(srt_path)

    async def test_second_subtitle_reloads(self, tmp_path):
        config = _make_config()
        srt_path = tmp_path / "test.srt"

        from livesubs.subtitles import SubtitleManager
        sub_manager = SubtitleManager(srt_path)

        player = AsyncMock()
        player.sub_add = AsyncMock()
        player.sub_reload = AsyncMock()

        translated_queue: asyncio.Queue = asyncio.Queue()
        await translated_queue.put((0.0, "Primero"))
        await translated_queue.put((5.0, "Segundo"))
        await translated_queue.put(None)

        await _subtitle_writer(config, translated_queue, sub_manager, player, live=False)

        player.sub_add.assert_called_once()
        player.sub_reload.assert_called_once_with(srt_path)


    async def test_live_uses_show_text(self, tmp_path):
        config = _make_config()
        srt_path = tmp_path / "test.srt"

        from livesubs.subtitles import SubtitleManager
        sub_manager = SubtitleManager(srt_path)

        player = AsyncMock()
        player.show_text = AsyncMock()

        translated_queue: asyncio.Queue = asyncio.Queue()
        await translated_queue.put((0.0, "En vivo"))
        await translated_queue.put(None)

        await _subtitle_writer(config, translated_queue, sub_manager, player, live=True)

        player.show_text.assert_called_once_with("En vivo", 5000)
        content = srt_path.read_text(encoding="utf-8")
        assert content == ""


class TestOverlaySubtitleWriter:
    async def test_updates_overlay_text(self):
        config = _make_config()
        overlay = MagicMock()

        translated_queue: asyncio.Queue = asyncio.Queue()
        await translated_queue.put((0.0, "Subtitulo overlay"))
        await translated_queue.put(None)

        await _overlay_subtitle_writer(config, translated_queue, overlay)

        overlay.update_text.assert_called_once_with("Subtitulo overlay")

    async def test_multiple_updates(self):
        config = _make_config()
        overlay = MagicMock()

        translated_queue: asyncio.Queue = asyncio.Queue()
        await translated_queue.put((0.0, "Primero"))
        await translated_queue.put((5.0, "Segundo"))
        await translated_queue.put(None)

        await _overlay_subtitle_writer(config, translated_queue, overlay)

        assert overlay.update_text.call_count == 2
        overlay.update_text.assert_any_call("Primero")
        overlay.update_text.assert_any_call("Segundo")

    async def test_handles_exception_gracefully(self):
        config = _make_config()
        overlay = MagicMock()
        overlay.update_text.side_effect = RuntimeError("tkinter error")

        translated_queue: asyncio.Queue = asyncio.Queue()
        await translated_queue.put((0.0, "texto"))
        await translated_queue.put(None)

        # No deberia lanzar excepcion
        await _overlay_subtitle_writer(config, translated_queue, overlay)


class TestTranscribeWorkerParallel:
    async def test_processes_chunks_and_sends_sentinel(self):
        config = _make_config()
        client = AsyncMock()

        audio_queue: asyncio.Queue = asyncio.Queue()
        text_queue: asyncio.Queue = asyncio.Queue()

        chunk = Path("/tmp/fake_chunk.wav")

        with patch("livesubs.pipeline.transcribe_audio", new_callable=AsyncMock) as mock_transcribe:
            mock_transcribe.return_value = "texto paralelo"
            with patch.object(Path, "unlink"):
                await audio_queue.put(chunk)
                await audio_queue.put(None)

                await _transcribe_worker_parallel(
                    client, config, audio_queue, text_queue, 5, max_concurrent=2,
                )

        item = await text_queue.get()
        assert item == (0, "texto paralelo")
        sentinel = await text_queue.get()
        assert sentinel is None

    async def test_preserves_order_with_multiple_chunks(self):
        config = _make_config()
        client = AsyncMock()

        audio_queue: asyncio.Queue = asyncio.Queue()
        text_queue: asyncio.Queue = asyncio.Queue()

        with patch("livesubs.pipeline.transcribe_audio", new_callable=AsyncMock) as mock_transcribe:
            mock_transcribe.side_effect = ["texto_0", "texto_1", "texto_2"]
            with patch.object(Path, "unlink"):
                await audio_queue.put(Path("/tmp/c0.wav"))
                await audio_queue.put(Path("/tmp/c1.wav"))
                await audio_queue.put(Path("/tmp/c2.wav"))
                await audio_queue.put(None)

                await _transcribe_worker_parallel(
                    client, config, audio_queue, text_queue, 5, max_concurrent=3,
                )

        items = []
        while not text_queue.empty():
            items.append(await text_queue.get())

        assert items[0] == (0, "texto_0")
        assert items[1] == (5, "texto_1")
        assert items[2] == (10, "texto_2")
        assert items[3] is None

    async def test_skips_failed_transcription(self):
        config = _make_config()
        client = AsyncMock()

        audio_queue: asyncio.Queue = asyncio.Queue()
        text_queue: asyncio.Queue = asyncio.Queue()

        with patch("livesubs.pipeline.transcribe_audio", new_callable=AsyncMock) as mock_transcribe:
            # Chunk 0 falla, chunk 1 funciona
            mock_transcribe.side_effect = [None, "texto_1"]
            with patch.object(Path, "unlink"):
                await audio_queue.put(Path("/tmp/c0.wav"))
                await audio_queue.put(Path("/tmp/c1.wav"))
                await audio_queue.put(None)

                await _transcribe_worker_parallel(
                    client, config, audio_queue, text_queue, 5, max_concurrent=2,
                )

        # Solo chunk 1 deberia estar (chunk 0 fallo)
        item = await text_queue.get()
        assert item == (5, "texto_1")
        sentinel = await text_queue.get()
        assert sentinel is None

    async def test_handles_exception_gracefully(self):
        config = _make_config()
        client = AsyncMock()

        audio_queue: asyncio.Queue = asyncio.Queue()
        text_queue: asyncio.Queue = asyncio.Queue()

        with patch("livesubs.pipeline.transcribe_audio", new_callable=AsyncMock) as mock_transcribe:
            mock_transcribe.side_effect = RuntimeError("boom")
            with patch.object(Path, "unlink"):
                await audio_queue.put(Path("/tmp/chunk.wav"))
                await audio_queue.put(None)

                await _transcribe_worker_parallel(
                    client, config, audio_queue, text_queue, 5, max_concurrent=2,
                )

        sentinel = await text_queue.get()
        assert sentinel is None


class TestMonitorMPV:
    async def test_stops_segmenter_when_mpv_closes(self):
        player = MagicMock()
        player.is_running.return_value = False

        segmenter = AsyncMock()
        segmenter.stop = AsyncMock()

        await _monitor_mpv(player, segmenter)
        segmenter.stop.assert_called_once()


class TestMonitorOverlay:
    async def test_stops_segmenter_when_overlay_closes(self):
        overlay = MagicMock()
        overlay.is_running.return_value = False

        segmenter = AsyncMock()
        segmenter.stop = AsyncMock()

        await _monitor_overlay(overlay, segmenter)
        segmenter.stop.assert_called_once()
