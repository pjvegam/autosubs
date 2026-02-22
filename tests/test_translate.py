from unittest.mock import AsyncMock, MagicMock, patch

from livesubs.costs import CostTracker
from livesubs.translate import (
    _ROLE,
    _RULES,
    SYSTEM_PROMPT,
    _build_system_prompt,
    translate_text,
)


def _mock_completion(
    content: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> MagicMock:
    """Crea un mock de ChatCompletion response."""
    choice = MagicMock()
    choice.message.content = content
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
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

    async def test_context_appears_in_system_prompt(self):
        client = AsyncMock()
        client.chat.completions.create.return_value = _mock_completion("Hello")

        await translate_text(
            client, "Hola", "es", "en", context="Canal: TestChannel\nTitulo: Test Stream",
        )

        call_kwargs = client.chat.completions.create.call_args.kwargs
        system_content = call_kwargs["messages"][0]["content"]
        assert "Canal: TestChannel" in system_content
        assert "Titulo: Test Stream" in system_content
        assert system_content.startswith(_ROLE)
        assert system_content.endswith(_RULES)

    async def test_no_context_uses_default_prompt(self):
        client = AsyncMock()
        client.chat.completions.create.return_value = _mock_completion("Hello")

        await translate_text(client, "Hola", "es", "en")

        call_kwargs = client.chat.completions.create.call_args.kwargs
        assert call_kwargs["messages"][0]["content"] == SYSTEM_PROMPT


    async def test_tracker_records_usage(self):
        client = AsyncMock()
        client.chat.completions.create.return_value = _mock_completion(
            "Hola", prompt_tokens=120, completion_tokens=30,
        )
        tracker = CostTracker(translation_model="gpt-4o-mini")

        await translate_text(client, "Hello", "en", "es", tracker=tracker)

        assert tracker.translation_prompt_tokens == 120
        assert tracker.translation_completion_tokens == 30

    async def test_tracker_accumulates_across_calls(self):
        client = AsyncMock()
        client.chat.completions.create.side_effect = [
            _mock_completion("Hola", prompt_tokens=100, completion_tokens=20),
            _mock_completion("Mundo", prompt_tokens=110, completion_tokens=25),
        ]
        tracker = CostTracker(translation_model="gpt-4o-mini")

        await translate_text(client, "Hello", "en", "es", tracker=tracker)
        await translate_text(client, "World", "en", "es", tracker=tracker)

        assert tracker.translation_prompt_tokens == 210
        assert tracker.translation_completion_tokens == 45

    async def test_no_tracker_does_not_fail(self):
        client = AsyncMock()
        client.chat.completions.create.return_value = _mock_completion(
            "Hola", prompt_tokens=100, completion_tokens=20,
        )

        result = await translate_text(client, "Hello", "en", "es")
        assert result == "Hola"


class TestBuildSystemPrompt:
    def test_empty_context_returns_default(self):
        assert _build_system_prompt("") == SYSTEM_PROMPT

    def test_context_inserted_between_role_and_rules(self):
        result = _build_system_prompt("Canal: Test")
        assert result.startswith(_ROLE)
        assert result.endswith(_RULES)
        assert "Canal: Test" in result
