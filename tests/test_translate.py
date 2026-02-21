from unittest.mock import AsyncMock, MagicMock, patch

from livesubs.translate import SYSTEM_PROMPT, translate_text


def _mock_completion(content: str) -> MagicMock:
    """Crea un mock de ChatCompletion response."""
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    return response


class TestTranslateText:
    async def test_returns_translated_text(self):
        client = AsyncMock()
        client.chat.completions.create.return_value = _mock_completion("  Hola mundo  ")

        result = await translate_text(client, "こんにちは世界", "ja", "es")
        assert result == "Hola mundo"

    async def test_returns_none_on_empty_response(self):
        client = AsyncMock()
        client.chat.completions.create.return_value = _mock_completion("   ")

        result = await translate_text(client, "テスト", "ja", "es")
        assert result is None

    async def test_passes_correct_params(self):
        client = AsyncMock()
        client.chat.completions.create.return_value = _mock_completion("Hello")

        await translate_text(client, "Hola", "es", "en", model="gpt-4o")

        call_kwargs = client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o"
        assert call_kwargs["temperature"] == 0.3
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == SYSTEM_PROMPT
        assert messages[1]["role"] == "user"
        assert "es" in messages[1]["content"]
        assert "en" in messages[1]["content"]
        assert "Hola" in messages[1]["content"]

    async def test_retries_on_api_error(self):
        client = AsyncMock()
        client.chat.completions.create.side_effect = [
            RuntimeError("API error"),
            _mock_completion("texto recuperado"),
        ]

        with patch("livesubs.retry.BASE_DELAY", 0.01):
            result = await translate_text(client, "テスト", "ja", "es")
        assert result == "texto recuperado"

    async def test_returns_none_after_all_retries_fail(self):
        client = AsyncMock()
        client.chat.completions.create.side_effect = RuntimeError("API error")

        with patch("livesubs.retry.BASE_DELAY", 0.01):
            result = await translate_text(client, "テスト", "ja", "es")
        assert result is None
