
import pytest

from autosubs.asr_eval import (
    LOCAL_ENGINES,
    EngineReport,
    client_for,
    is_openrouter,
)


class TestIsOpenrouter:
    def test_slug_with_slash_goes_to_openrouter(self):
        assert is_openrouter("microsoft/mai-transcribe-1.5")

    def test_plain_model_name_goes_to_openai(self):
        assert not is_openrouter("gpt-4o-transcribe")

    def test_local_engines_are_not_openrouter(self):
        assert all(not is_openrouter(e) for e in LOCAL_ENGINES)


class TestEngineReport:
    def test_counts_empty_hypotheses(self):
        report = EngineReport(engine="x", hypotheses={"a": "texto", "b": "", "c": ""})
        assert report.empty == 2

    def test_no_empties(self):
        assert EngineReport(engine="x", hypotheses={"a": "hola"}).empty == 0

    def test_empty_report(self):
        assert EngineReport(engine="x").empty == 0


class TestClientFor:
    def test_openrouter_needs_key(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
            client_for("microsoft/mai-transcribe-1.5")

    def test_openrouter_uses_its_base_url(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
        client = client_for("microsoft/mai-transcribe-1.5")
        assert "openrouter.ai" in str(client.base_url)

    def test_openai_engine_uses_default_base_url(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        client = client_for("gpt-4o-transcribe")
        assert "openrouter.ai" not in str(client.base_url)
