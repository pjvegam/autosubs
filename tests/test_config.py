from unittest.mock import patch

import pytest

from autosubs.config import Config


class TestConfig:
    def test_create_with_explicit_api_key(self, monkeypatch):
        # Patch load_dotenv para que no re-cargue .env del proyecto
        monkeypatch.setattr("autosubs.config.load_dotenv", lambda: None)
        # Forzar defaults basicos (sin Gemini, sin OpenRouter, sin faster-whisper)
        monkeypatch.setattr(
            "autosubs.config.importlib.util.find_spec", lambda name: None,
        )
        for var in (
            "AUTOSUBS_SOURCE_LANG", "AUTOSUBS_TARGET_LANG",
            "GEMINI_API_KEY", "OPENROUTER_API_KEY",
            "AUTOSUBS_WHISPER_MODEL", "AUTOSUBS_GPT_MODEL",
            "AUTOSUBS_SEGMENT_DURATION",
        ):
            monkeypatch.delenv(var, raising=False)
        config = Config(url="https://example.com", api_key="sk-test123")
        assert config.url == "https://example.com"
        assert config.api_key == "sk-test123"
        assert config.source_lang == "ja"
        assert config.target_lang == "es"
        assert config.segment_duration == 5
        assert config.whisper_model == "gpt-4o-transcribe"
        assert config.gpt_model == "gpt-4.1"
        assert config.verbose is False

    def test_langs_from_env_vars(self, monkeypatch):
        monkeypatch.setenv("AUTOSUBS_SOURCE_LANG", "en")
        monkeypatch.setenv("AUTOSUBS_TARGET_LANG", "fr")
        config = Config(url="https://example.com", api_key="sk-test")
        assert config.source_lang == "en"
        assert config.target_lang == "fr"

    def test_explicit_langs_override_env(self, monkeypatch):
        monkeypatch.setenv("AUTOSUBS_SOURCE_LANG", "en")
        monkeypatch.setenv("AUTOSUBS_TARGET_LANG", "fr")
        config = Config(
            url="https://example.com", api_key="sk-test",
            source_lang="ko", target_lang="de",
        )
        assert config.source_lang == "ko"
        assert config.target_lang == "de"

    def test_create_with_env_var(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        config = Config(url="https://example.com")
        assert config.api_key == "sk-from-env"

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        # Mockear load_dotenv para evitar que recargue el .env real
        with (
            patch("autosubs.config.load_dotenv"),
            pytest.raises(ValueError, match="OPENAI_API_KEY no encontrada"),
        ):
            Config(url="https://example.com")

    def test_explicit_key_overrides_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        config = Config(url="https://example.com", api_key="sk-explicit")
        assert config.api_key == "sk-explicit"

    def test_custom_params(self):
        config = Config(
            url="https://example.com",
            api_key="sk-test",
            source_lang="en",
            target_lang="fr",
            segment_duration=10,
            whisper_model="whisper-1",
            gpt_model="gpt-4o",
            verbose=True,
        )
        assert config.source_lang == "en"
        assert config.target_lang == "fr"
        assert config.segment_duration == 10
        assert config.whisper_model == "whisper-1"
        assert config.gpt_model == "gpt-4o"
        assert config.verbose is True

    def test_segment_duration_from_env(self, monkeypatch):
        monkeypatch.setenv("AUTOSUBS_SEGMENT_DURATION", "10")
        config = Config(url="https://example.com", api_key="sk-test")
        assert config.segment_duration == 10

    def test_whisper_model_from_env(self, monkeypatch):
        monkeypatch.setenv("AUTOSUBS_WHISPER_MODEL", "whisper-1")
        config = Config(url="https://example.com", api_key="sk-test")
        assert config.whisper_model == "whisper-1"

    def test_gpt_model_from_env(self, monkeypatch):
        monkeypatch.setenv("AUTOSUBS_GPT_MODEL", "gpt-4o")
        config = Config(url="https://example.com", api_key="sk-test")
        assert config.gpt_model == "gpt-4o"

    def test_explicit_params_override_env(self, monkeypatch):
        monkeypatch.setenv("AUTOSUBS_SEGMENT_DURATION", "10")
        monkeypatch.setenv("AUTOSUBS_WHISPER_MODEL", "whisper-1")
        monkeypatch.setenv("AUTOSUBS_GPT_MODEL", "gpt-4o")
        config = Config(
            url="https://example.com", api_key="sk-test",
            segment_duration=3, whisper_model="custom", gpt_model="custom-gpt",
        )
        assert config.segment_duration == 3
        assert config.whisper_model == "custom"
        assert config.gpt_model == "custom-gpt"

    def test_ipc_pipe_is_windows_named_pipe(self):
        config = Config(url="https://example.com", api_key="sk-test")
        assert config.ipc_pipe == r"\\.\pipe\mpvsocket"

    def test_api_key_not_in_repr(self):
        config = Config(url="https://example.com", api_key="sk-secret-key")
        repr_str = repr(config)
        assert "sk-secret-key" not in repr_str
