import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from livesubs.config import Config
from livesubs.costs import CostTracker
from livesubs.pipeline import (
    _get_stream_context,
    _monitor_mpv,
    _monitor_overlay,
    _overlay_subtitle_writer,
    _subtitle_writer,
    _transcribe_worker,
    _transcribe_worker_parallel,
    _translate_worker,
)
from livesubs.vad import AudioChunk


def _make_config(**kwargs) -> Config:
    defaults = dict(
        url="https://example.com",
        api_key="sk-test",
        segment_duration=5,
        whisper_model="gpt-4o-transcribe",  # explicit para no depender de env vars
        gpt_model="gpt-4.1",
    )
    defaults.update(kwargs)
    return Config(**defaults)


def _make_chunk(path: Path | str = "/tmp/fake_chunk.wav", duration: float = 5.0) -> AudioChunk:
    return AudioChunk(path=Path(path), duration=duration)


class TestTranscribeWorker:
    async def test_processes_chunks_and_sends_sentinel(self):
        config = _make_config()
        client = AsyncMock()

        audio_queue: asyncio.Queue = asyncio.Queue()
        text_queue: asyncio.Queue = asyncio.Queue()

        chunk = _make_chunk(duration=4.2)

        with patch("livesubs.pipeline.transcribe_audio", new_callable=AsyncMock) as mock_transcribe:
            mock_transcribe.return_value = "texto transcrito"
            with patch.object(Path, "unlink"):
                await audio_queue.put(chunk)
                await audio_queue.put(None)

                await _transcribe_worker(client, config, audio_queue, text_queue)

        item = await text_queue.get()
        assert item == (0, 4.2, "texto transcrito")
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
                await audio_queue.put(_make_chunk())
                await audio_queue.put(None)

                await _transcribe_worker(client, config, audio_queue, text_queue)

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
                await audio_queue.put(_make_chunk())
                await audio_queue.put(None)

                await _transcribe_worker(client, config, audio_queue, text_queue)

        sentinel = await text_queue.get()
        assert sentinel is None

    async def test_accumulates_real_timestamps(self):
        config = _make_config()
        client = AsyncMock()

        audio_queue: asyncio.Queue = asyncio.Queue()
        text_queue: asyncio.Queue = asyncio.Queue()

        with patch("livesubs.pipeline.transcribe_audio", new_callable=AsyncMock) as mock_transcribe:
            mock_transcribe.side_effect = ["texto uno", "texto dos", "texto tres"]
            with patch.object(Path, "unlink"):
                await audio_queue.put(_make_chunk(duration=4.2))
                await audio_queue.put(_make_chunk(duration=5.8))
                await audio_queue.put(_make_chunk(duration=3.1))
                await audio_queue.put(None)

                await _transcribe_worker(client, config, audio_queue, text_queue)

        items = []
        while not text_queue.empty():
            items.append(await text_queue.get())

        assert items[0] == (0, 4.2, "texto uno")
        assert items[1][0] == pytest.approx(4.2)
        assert items[1][1] == 5.8
        assert items[2][0] == pytest.approx(10.0)
        assert items[2][1] == 3.1
        assert items[3] is None


    async def test_tracker_records_whisper_duration(self):
        config = _make_config()
        client = AsyncMock()
        tracker = CostTracker(whisper_model="gpt-4o-transcribe")

        audio_queue: asyncio.Queue = asyncio.Queue()
        text_queue: asyncio.Queue = asyncio.Queue()

        with patch("livesubs.pipeline.transcribe_audio", new_callable=AsyncMock) as mock_transcribe:
            mock_transcribe.side_effect = ["texto uno", "texto dos"]
            with patch.object(Path, "unlink"):
                await audio_queue.put(_make_chunk(duration=4.2))
                await audio_queue.put(_make_chunk(duration=5.8))
                await audio_queue.put(None)

                await _transcribe_worker(
                    client, config, audio_queue, text_queue, tracker=tracker,
                )

        assert tracker.whisper_seconds == pytest.approx(10.0)

    async def test_tracker_not_updated_on_failed_transcription(self):
        config = _make_config()
        client = AsyncMock()
        tracker = CostTracker(whisper_model="gpt-4o-transcribe")

        audio_queue: asyncio.Queue = asyncio.Queue()
        text_queue: asyncio.Queue = asyncio.Queue()

        with patch("livesubs.pipeline.transcribe_audio", new_callable=AsyncMock) as mock_transcribe:
            mock_transcribe.return_value = None
            with patch.object(Path, "unlink"):
                await audio_queue.put(_make_chunk(duration=5.0))
                await audio_queue.put(None)

                await _transcribe_worker(
                    client, config, audio_queue, text_queue, tracker=tracker,
                )

        assert tracker.whisper_seconds == 0.0


class TestTranslateWorker:
    async def test_translates_and_forwards(self):
        config = _make_config()
        client = AsyncMock()

        text_queue: asyncio.Queue = asyncio.Queue()
        translated_queue: asyncio.Queue = asyncio.Queue()

        with patch("livesubs.pipeline.translate_text", new_callable=AsyncMock) as mock_translate:
            mock_translate.return_value = "traducido"
            await text_queue.put((0, 4.5, "original"))
            await text_queue.put(None)

            await _translate_worker(client, config, text_queue, translated_queue)

        item = await translated_queue.get()
        assert item == (0, 4.5, "traducido")
        sentinel = await translated_queue.get()
        assert sentinel is None

    async def test_skips_failed_translation(self):
        config = _make_config()
        client = AsyncMock()

        text_queue: asyncio.Queue = asyncio.Queue()
        translated_queue: asyncio.Queue = asyncio.Queue()

        with patch("livesubs.pipeline.translate_text", new_callable=AsyncMock) as mock_translate:
            mock_translate.return_value = None
            await text_queue.put((0, 5.0, "original"))
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
            await text_queue.put((0, 5.0, "original"))
            await text_queue.put(None)

            await _translate_worker(client, config, text_queue, translated_queue)

        sentinel = await translated_queue.get()
        assert sentinel is None

    async def test_passes_stream_context(self):
        config = _make_config()
        client = AsyncMock()

        text_queue: asyncio.Queue = asyncio.Queue()
        translated_queue: asyncio.Queue = asyncio.Queue()

        with patch("livesubs.pipeline.translate_text", new_callable=AsyncMock) as mock_translate:
            mock_translate.return_value = "traducido"
            await text_queue.put((0, 4.5, "original"))
            await text_queue.put(None)

            await _translate_worker(
                client, config, text_queue, translated_queue,
                stream_context="Channel: Test\nTitle: Mi Stream",
            )

        call_kwargs = mock_translate.call_args.kwargs
        assert "Channel: Test" in call_kwargs["context"]
        assert "Title: Mi Stream" in call_kwargs["context"]

    async def test_accumulates_history_in_context(self):
        config = _make_config()
        client = AsyncMock()

        text_queue: asyncio.Queue = asyncio.Queue()
        translated_queue: asyncio.Queue = asyncio.Queue()

        with patch("livesubs.pipeline.translate_text", new_callable=AsyncMock) as mock_translate:
            mock_translate.side_effect = ["trad_1", "trad_2", "trad_3"]
            await text_queue.put((0, 3.0, "texto_1"))
            await text_queue.put((3.0, 3.0, "texto_2"))
            await text_queue.put((6.0, 3.0, "texto_3"))
            await text_queue.put(None)

            await _translate_worker(client, config, text_queue, translated_queue)

        calls = mock_translate.call_args_list
        # Primera llamada: sin historial
        assert calls[0].kwargs.get("context", "") == ""
        # Segunda llamada: historial con trad_1
        assert "trad_1" in calls[1].kwargs["context"]
        # Tercera llamada: historial con trad_1 y trad_2
        assert "trad_1" in calls[2].kwargs["context"]
        assert "trad_2" in calls[2].kwargs["context"]

    async def test_history_maxlen_is_three(self):
        config = _make_config()
        client = AsyncMock()

        text_queue: asyncio.Queue = asyncio.Queue()
        translated_queue: asyncio.Queue = asyncio.Queue()

        with patch("livesubs.pipeline.translate_text", new_callable=AsyncMock) as mock_translate:
            mock_translate.side_effect = ["t1", "t2", "t3", "t4", "t5"]
            for i in range(5):
                await text_queue.put((i * 3.0, 3.0, f"texto_{i}"))
            await text_queue.put(None)

            await _translate_worker(client, config, text_queue, translated_queue)

        # La 5ta llamada (index 4) debe tener t2, t3, t4 (no t1)
        last_call_context = mock_translate.call_args_list[4].kwargs["context"]
        assert "t1" not in last_call_context
        assert "t2" in last_call_context
        assert "t3" in last_call_context
        assert "t4" in last_call_context

    async def test_with_diarizer_adds_speaker_tag(self):
        config = _make_config()
        client = AsyncMock()

        diarizer = MagicMock()
        diarizer.get_speaker.return_value = 1  # Speaker 2 (0-indexed)

        text_queue: asyncio.Queue = asyncio.Queue()
        translated_queue: asyncio.Queue = asyncio.Queue()

        with patch("livesubs.pipeline.translate_text", new_callable=AsyncMock) as mock_translate:
            mock_translate.return_value = "traducido"
            await text_queue.put((0, 4.5, "original"))
            await text_queue.put(None)

            await _translate_worker(
                client, config, text_queue, translated_queue,
                diarizer=diarizer,
            )

        item = await translated_queue.get()
        assert item[2] == "[Speaker 2] traducido"

        # Verificar que el contexto incluye info del speaker
        call_kwargs = mock_translate.call_args.kwargs
        assert "Current speaker: [Speaker 2]" in call_kwargs["context"]

    async def test_without_diarizer_no_speaker_tag(self):
        config = _make_config()
        client = AsyncMock()

        text_queue: asyncio.Queue = asyncio.Queue()
        translated_queue: asyncio.Queue = asyncio.Queue()

        with patch("livesubs.pipeline.translate_text", new_callable=AsyncMock) as mock_translate:
            mock_translate.return_value = "traducido"
            await text_queue.put((0, 4.5, "original"))
            await text_queue.put(None)

            await _translate_worker(client, config, text_queue, translated_queue)

        item = await translated_queue.get()
        assert item[2] == "traducido"
        assert "[Speaker" not in item[2]

    async def test_diarizer_returns_none_no_tag(self):
        config = _make_config()
        client = AsyncMock()

        diarizer = MagicMock()
        diarizer.get_speaker.return_value = None

        text_queue: asyncio.Queue = asyncio.Queue()
        translated_queue: asyncio.Queue = asyncio.Queue()

        with patch("livesubs.pipeline.translate_text", new_callable=AsyncMock) as mock_translate:
            mock_translate.return_value = "traducido"
            await text_queue.put((0, 4.5, "original"))
            await text_queue.put(None)

            await _translate_worker(
                client, config, text_queue, translated_queue,
                diarizer=diarizer,
            )

        item = await translated_queue.get()
        assert item[2] == "traducido"
        assert "[Speaker" not in item[2]

    async def test_tracker_passed_to_translate_text(self):
        config = _make_config()
        client = AsyncMock()
        tracker = CostTracker(translation_model="gpt-4.1")

        text_queue: asyncio.Queue = asyncio.Queue()
        translated_queue: asyncio.Queue = asyncio.Queue()

        with patch("livesubs.pipeline.translate_text", new_callable=AsyncMock) as mock_translate:
            mock_translate.return_value = "traducido"
            await text_queue.put((0, 4.5, "original"))
            await text_queue.put(None)

            await _translate_worker(
                client, config, text_queue, translated_queue, tracker=tracker,
            )

        call_kwargs = mock_translate.call_args.kwargs
        assert call_kwargs["tracker"] is tracker

    async def test_tracker_records_entries(self):
        config = _make_config()
        client = AsyncMock()
        tracker = CostTracker(translation_model="gpt-4.1")

        text_queue: asyncio.Queue = asyncio.Queue()
        translated_queue: asyncio.Queue = asyncio.Queue()

        with patch("livesubs.pipeline.translate_text", new_callable=AsyncMock) as mock_translate:
            mock_translate.side_effect = ["traducido_1", "traducido_2"]
            await text_queue.put((0, 4.5, "original_1"))
            await text_queue.put((4.5, 5.0, "original_2"))
            await text_queue.put(None)

            await _translate_worker(
                client, config, text_queue, translated_queue, tracker=tracker,
            )

        assert len(tracker.entries) == 2
        assert tracker.entries[0].transcription == "original_1"
        assert tracker.entries[0].translation == "traducido_1"
        assert tracker.entries[1].transcription == "original_2"
        assert tracker.entries[1].translation == "traducido_2"

    async def test_tracker_entry_uses_clean_translation_without_speaker_tag(self):
        config = _make_config()
        client = AsyncMock()
        tracker = CostTracker(translation_model="gpt-4.1")

        diarizer = MagicMock()
        diarizer.get_speaker.return_value = 0

        text_queue: asyncio.Queue = asyncio.Queue()
        translated_queue: asyncio.Queue = asyncio.Queue()

        with patch("livesubs.pipeline.translate_text", new_callable=AsyncMock) as mock_translate:
            mock_translate.return_value = "traducido"
            await text_queue.put((0, 4.5, "original"))
            await text_queue.put(None)

            await _translate_worker(
                client, config, text_queue, translated_queue,
                diarizer=diarizer, tracker=tracker,
            )

        # Entry guarda traduccion limpia, sin speaker tag
        assert tracker.entries[0].translation == "traducido"
        # Pero la queue tiene el tag
        item = await translated_queue.get()
        assert item[2] == "[Speaker 1] traducido"


class TestGetStreamContext:
    def _make_client_config(self):
        client = AsyncMock()
        config = _make_config()
        return client, config

    async def test_returns_live_and_context(self):
        client, config = self._make_client_config()
        info = {"is_live": True, "id": "abc123", "title": "Mi Stream", "channel": "TestChannel"}
        mock_translate = AsyncMock(return_value="My Stream")
        with (
            patch("livesubs.pipeline.get_stream_info", new_callable=AsyncMock, return_value=info),
            patch("livesubs.pipeline.translate_text", mock_translate),
        ):
            result = await _get_stream_context("https://example.com", client, config)
            live, context, video_id = result

        assert live is True
        assert "Channel: TestChannel" in context
        assert "Mi Stream (My Stream)" in context
        assert video_id == "abc123"

    async def test_returns_vod_on_non_live(self):
        client, config = self._make_client_config()
        info = {"is_live": False, "id": "vid456", "title": "Video", "channel": "Canal"}
        mock_translate = AsyncMock(return_value="Video translated")
        with (
            patch("livesubs.pipeline.get_stream_info", new_callable=AsyncMock, return_value=info),
            patch("livesubs.pipeline.translate_text", mock_translate),
        ):
            result = await _get_stream_context("https://example.com", client, config)
            live, context, video_id = result

        assert live is False
        assert "Channel: Canal" in context
        assert video_id == "vid456"

    async def test_falls_back_on_error(self):
        client, config = self._make_client_config()
        mock_info = AsyncMock(side_effect=RuntimeError("fail"))
        with patch("livesubs.pipeline.get_stream_info", mock_info):
            result = await _get_stream_context("https://example.com", client, config)
            live, context, video_id = result

        assert live is False
        assert context == ""
        assert video_id == ""

    async def test_uses_uploader_fallback(self):
        client, config = self._make_client_config()
        info = {"is_live": True, "title": "Stream", "uploader": "Uploader1"}
        mock_translate = AsyncMock(return_value="Stream translated")
        with (
            patch("livesubs.pipeline.get_stream_info", new_callable=AsyncMock, return_value=info),
            patch("livesubs.pipeline.translate_text", mock_translate),
        ):
            _live, context, _ = await _get_stream_context("https://example.com", client, config)

        assert "Channel: Uploader1" in context

    async def test_translates_title(self):
        client, config = self._make_client_config()
        info = {"is_live": True, "title": "配信テスト", "channel": "TestCh"}
        mock_translate = AsyncMock(return_value="Stream Test")
        with (
            patch("livesubs.pipeline.get_stream_info", new_callable=AsyncMock, return_value=info),
            patch("livesubs.pipeline.translate_text", mock_translate),
        ):
            _, context, _ = await _get_stream_context("https://example.com", client, config)

        mock_translate.assert_called_once_with(
            client, "配信テスト", config.source_lang, config.target_lang,
            config.gpt_model, tracker=None,
        )
        assert "配信テスト (Stream Test)" in context

    async def test_title_unchanged_when_translation_fails(self):
        client, config = self._make_client_config()
        info = {"is_live": False, "title": "配信テスト", "channel": "Ch"}
        with (
            patch("livesubs.pipeline.get_stream_info", new_callable=AsyncMock, return_value=info),
            patch("livesubs.pipeline.translate_text", new_callable=AsyncMock, return_value=None),
        ):
            _, context, _ = await _get_stream_context("https://example.com", client, config)

        assert "Title: 配信テスト" in context
        assert "(" not in context

    async def test_no_translation_when_title_empty(self):
        client, config = self._make_client_config()
        info = {"is_live": True, "title": "", "channel": "SomeChannel"}
        with (
            patch("livesubs.pipeline.get_stream_info", new_callable=AsyncMock, return_value=info),
            patch("livesubs.pipeline.translate_text", new_callable=AsyncMock) as mock_translate,
        ):
            _, context, _ = await _get_stream_context("https://example.com", client, config)

        mock_translate.assert_not_called()
        assert "Titulo" not in context


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
        await translated_queue.put((0.0, 4.5, "Hola mundo"))
        await translated_queue.put(None)

        await _subtitle_writer(config, translated_queue, sub_manager, player, live=False)

        content = srt_path.read_text(encoding="utf-8")
        assert "Hola mundo" in content
        assert "00:00:00,000 --> 00:00:04,500" in content
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
        await translated_queue.put((0.0, 4.5, "Primero"))
        await translated_queue.put((4.5, 5.5, "Segundo"))
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
        player.get_time_pos = AsyncMock(return_value=None)

        translated_queue: asyncio.Queue = asyncio.Queue()
        await translated_queue.put((0.0, 4.5, "En vivo"))
        await translated_queue.put(None)

        await _subtitle_writer(config, translated_queue, sub_manager, player, live=True)

        player.show_text.assert_called_once_with("En vivo", 4500)
        content = srt_path.read_text(encoding="utf-8")
        assert content == ""

    async def test_live_syncs_with_mpv_position(self, tmp_path):
        config = _make_config()
        srt_path = tmp_path / "test.srt"

        from livesubs.subtitles import SubtitleManager
        sub_manager = SubtitleManager(srt_path)

        player = AsyncMock()
        player.show_text = AsyncMock()
        # MPV esta en segundo 3, subtitulo termina en segundo 5 -> espera ~2s
        player.get_time_pos = AsyncMock(return_value=3.0)

        translated_queue: asyncio.Queue = asyncio.Queue()
        await translated_queue.put((0, 5.0, "Sincronizado"))
        await translated_queue.put(None)

        with patch("livesubs.pipeline.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await _subtitle_writer(config, translated_queue, sub_manager, player, live=True)

        # Debe haber esperado ~2s (end_time=5 - mpv_pos=3)
        mock_sleep.assert_called_once()
        delay = mock_sleep.call_args[0][0]
        assert 1.9 < delay < 2.1
        player.show_text.assert_called_once_with("Sincronizado", 5000)

    async def test_live_no_sync_when_mpv_ahead(self, tmp_path):
        config = _make_config()
        srt_path = tmp_path / "test.srt"

        from livesubs.subtitles import SubtitleManager
        sub_manager = SubtitleManager(srt_path)

        player = AsyncMock()
        player.show_text = AsyncMock()
        # MPV ya paso el punto del subtitulo -> no espera
        player.get_time_pos = AsyncMock(return_value=10.0)

        translated_queue: asyncio.Queue = asyncio.Queue()
        await translated_queue.put((0, 5.0, "Ya paso"))
        await translated_queue.put(None)

        with patch("livesubs.pipeline.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await _subtitle_writer(config, translated_queue, sub_manager, player, live=True)

        mock_sleep.assert_not_called()
        player.show_text.assert_called_once_with("Ya paso", 5000)


class TestOverlaySubtitleWriter:
    async def test_updates_overlay_text(self):
        config = _make_config()
        overlay = MagicMock()

        translated_queue: asyncio.Queue = asyncio.Queue()
        await translated_queue.put((0.0, 5.0, "Subtitulo overlay"))
        await translated_queue.put(None)

        await _overlay_subtitle_writer(config, translated_queue, overlay)

        overlay.update_text.assert_called_once_with("Subtitulo overlay")

    async def test_multiple_updates(self):
        config = _make_config()
        overlay = MagicMock()

        translated_queue: asyncio.Queue = asyncio.Queue()
        await translated_queue.put((0.0, 4.5, "Primero"))
        await translated_queue.put((4.5, 5.5, "Segundo"))
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
        await translated_queue.put((0.0, 5.0, "texto"))
        await translated_queue.put(None)

        # No deberia lanzar excepcion
        await _overlay_subtitle_writer(config, translated_queue, overlay)


class TestTranscribeWorkerParallel:
    async def test_processes_chunks_and_sends_sentinel(self):
        config = _make_config()
        client = AsyncMock()

        audio_queue: asyncio.Queue = asyncio.Queue()
        text_queue: asyncio.Queue = asyncio.Queue()

        chunk = _make_chunk(duration=4.5)

        with patch("livesubs.pipeline.transcribe_audio", new_callable=AsyncMock) as mock_transcribe:
            mock_transcribe.return_value = "texto paralelo"
            with patch.object(Path, "unlink"):
                await audio_queue.put(chunk)
                await audio_queue.put(None)

                await _transcribe_worker_parallel(
                    client, config, audio_queue, text_queue, max_concurrent=2,
                )

        item = await text_queue.get()
        assert item == (0, 4.5, "texto paralelo")
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
                await audio_queue.put(_make_chunk(duration=4.2))
                await audio_queue.put(_make_chunk(duration=5.8))
                await audio_queue.put(_make_chunk(duration=3.1))
                await audio_queue.put(None)

                await _transcribe_worker_parallel(
                    client, config, audio_queue, text_queue, max_concurrent=3,
                )

        items = []
        while not text_queue.empty():
            items.append(await text_queue.get())

        assert items[0] == (0, 4.2, "texto_0")
        assert items[1][0] == pytest.approx(4.2)
        assert items[1][1] == 5.8
        assert items[1][2] == "texto_1"
        assert items[2][0] == pytest.approx(10.0)
        assert items[2][1] == 3.1
        assert items[2][2] == "texto_2"
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
                await audio_queue.put(_make_chunk(duration=4.0))
                await audio_queue.put(_make_chunk(duration=5.0))
                await audio_queue.put(None)

                await _transcribe_worker_parallel(
                    client, config, audio_queue, text_queue, max_concurrent=2,
                )

        # Solo chunk 1 deberia estar (chunk 0 fallo)
        item = await text_queue.get()
        assert item[0] == pytest.approx(4.0)
        assert item[1] == 5.0
        assert item[2] == "texto_1"
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
                await audio_queue.put(_make_chunk())
                await audio_queue.put(None)

                await _transcribe_worker_parallel(
                    client, config, audio_queue, text_queue, max_concurrent=2,
                )

        sentinel = await text_queue.get()
        assert sentinel is None

    async def test_tracker_records_whisper_duration(self):
        config = _make_config()
        client = AsyncMock()
        tracker = CostTracker(whisper_model="gpt-4o-transcribe")

        audio_queue: asyncio.Queue = asyncio.Queue()
        text_queue: asyncio.Queue = asyncio.Queue()

        with patch("livesubs.pipeline.transcribe_audio", new_callable=AsyncMock) as mock_transcribe:
            mock_transcribe.side_effect = ["texto_0", None, "texto_2"]
            with patch.object(Path, "unlink"):
                await audio_queue.put(_make_chunk(duration=4.0))
                await audio_queue.put(_make_chunk(duration=5.0))
                await audio_queue.put(_make_chunk(duration=3.0))
                await audio_queue.put(None)

                await _transcribe_worker_parallel(
                    client, config, audio_queue, text_queue,
                    max_concurrent=2, tracker=tracker,
                )

        # Solo chunks exitosos (0 y 2) deben registrarse
        assert tracker.whisper_seconds == pytest.approx(7.0)


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
