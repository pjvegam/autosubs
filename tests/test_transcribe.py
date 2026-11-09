from unittest.mock import AsyncMock, MagicMock, patch

from autosubs.transcribe import transcribe_audio


class TestResponseFormatPorProveedor:
    """OpenRouter rechaza response_format="text"."""

    async def test_openrouter_pide_json(self, tmp_path):
        audio_file = tmp_path / "chunk_0000.wav"
        audio_file.write_bytes(b"fake audio data")
        client = AsyncMock()
        client.audio.transcriptions.create.return_value = MagicMock(text="texto")

        await transcribe_audio(
            client, audio_file, "ja", "microsoft/mai-transcribe-2",
        )

        kwargs = client.audio.transcriptions.create.call_args.kwargs
        assert kwargs["response_format"] == "json"

    async def test_openai_pide_text(self, tmp_path):
        audio_file = tmp_path / "chunk_0000.wav"
        audio_file.write_bytes(b"fake audio data")
        client = AsyncMock()
        client.audio.transcriptions.create.return_value = "texto"

        await transcribe_audio(client, audio_file, "ja", "gpt-4o-transcribe")

        kwargs = client.audio.transcriptions.create.call_args.kwargs
        assert kwargs["response_format"] == "text"

    async def test_explicito_gana(self, tmp_path):
        audio_file = tmp_path / "chunk_0000.wav"
        audio_file.write_bytes(b"fake audio data")
        client = AsyncMock()
        client.audio.transcriptions.create.return_value = MagicMock(text="texto")

        await transcribe_audio(
            client, audio_file, "ja", "gpt-4o-transcribe", response_format="json",
        )

        kwargs = client.audio.transcriptions.create.call_args.kwargs
        assert kwargs["response_format"] == "json"


class TestTranscribeAudio:
    async def test_returns_transcribed_text(self, tmp_path):
        audio_file = tmp_path / "chunk_0000.wav"
        audio_file.write_bytes(b"fake audio data")

        client = AsyncMock()
        client.audio.transcriptions.create.return_value = "  Hola mundo  "

        result = await transcribe_audio(client, audio_file, "ja")
        assert result == "Hola mundo"

    async def test_returns_none_on_empty_response(self, tmp_path):
        audio_file = tmp_path / "chunk_0000.wav"
        audio_file.write_bytes(b"fake audio data")

        client = AsyncMock()
        client.audio.transcriptions.create.return_value = "   "

        result = await transcribe_audio(client, audio_file, "ja")
        assert result is None

    async def test_passes_correct_params(self, tmp_path):
        audio_file = tmp_path / "chunk_0000.wav"
        audio_file.write_bytes(b"fake audio data")

        client = AsyncMock()
        client.audio.transcriptions.create.return_value = "texto"

        await transcribe_audio(client, audio_file, "en", model="gpt-4o-transcribe")

        call_kwargs = client.audio.transcriptions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o-transcribe"
        assert call_kwargs["language"] == "en"
        assert call_kwargs["response_format"] == "text"

    async def test_retries_on_api_error(self, tmp_path):
        audio_file = tmp_path / "chunk_0000.wav"
        audio_file.write_bytes(b"fake audio data")

        client = AsyncMock()
        client.audio.transcriptions.create.side_effect = [
            RuntimeError("API error"),
            RuntimeError("API error"),
            "texto recuperado",
        ]

        with patch("autosubs.retry.BASE_DELAY", 0.01):
            result = await transcribe_audio(client, audio_file, "ja")
        assert result == "texto recuperado"

    async def test_returns_none_after_all_retries_fail(self, tmp_path):
        audio_file = tmp_path / "chunk_0000.wav"
        audio_file.write_bytes(b"fake audio data")

        client = AsyncMock()
        client.audio.transcriptions.create.side_effect = RuntimeError("API error")

        with patch("autosubs.retry.BASE_DELAY", 0.01):
            result = await transcribe_audio(client, audio_file, "ja")
        assert result is None
