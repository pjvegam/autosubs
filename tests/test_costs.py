from livesubs.costs import PRICING, CostTracker, escape_md_table


class TestCostTracker:
    def test_initial_state(self):
        tracker = CostTracker(whisper_model="gpt-4o-transcribe", translation_model="gpt-4.1")
        assert tracker.whisper_seconds == 0.0
        assert tracker.translation_prompt_tokens == 0
        assert tracker.translation_completion_tokens == 0

    def test_add_whisper_accumulates(self):
        tracker = CostTracker(whisper_model="gpt-4o-transcribe")
        tracker.add_whisper(30.0)
        tracker.add_whisper(15.0)
        assert tracker.whisper_seconds == 45.0

    def test_add_translation_accumulates(self):
        tracker = CostTracker(translation_model="gpt-4.1")
        tracker.add_translation(100, 50)
        tracker.add_translation(200, 80)
        assert tracker.translation_prompt_tokens == 300
        assert tracker.translation_completion_tokens == 130

    def test_whisper_cost_calculation(self):
        tracker = CostTracker(whisper_model="gpt-4o-transcribe")
        tracker.add_whisper(60.0)  # 1 minuto
        cost = tracker.whisper_cost()
        assert cost == PRICING["gpt-4o-transcribe"]["per_minute"]

    def test_whisper_cost_unknown_model(self):
        tracker = CostTracker(whisper_model="modelo-inventado")
        tracker.add_whisper(120.0)
        assert tracker.whisper_cost() == 0.0

    def test_translation_cost_calculation(self):
        tracker = CostTracker(translation_model="gpt-4.1")
        tracker.add_translation(1_000_000, 500_000)
        input_cost, output_cost = tracker.translation_cost()
        assert input_cost == PRICING["gpt-4.1"]["input_mtok"]
        assert output_cost == PRICING["gpt-4.1"]["output_mtok"] * 0.5

    def test_translation_cost_unknown_model(self):
        tracker = CostTracker(translation_model="modelo-inventado")
        tracker.add_translation(1000, 500)
        assert tracker.translation_cost() == (0.0, 0.0)

    def test_summary_format(self):
        tracker = CostTracker(
            whisper_model="gpt-4o-transcribe",
            translation_model="gpt-4.1",
        )
        tracker.add_whisper(120.0)
        tracker.add_translation(5000, 1000)

        summary = tracker.summary()
        assert "--- Resumen de costos ---" in summary
        assert "Whisper (gpt-4o-transcribe)" in summary
        assert "2.0 min" in summary
        assert "Traduccion (gpt-4.1)" in summary
        assert "5,000 in" in summary
        assert "1,000 out tokens" in summary
        assert "Total:" in summary
        assert "USD" in summary

    def test_summary_with_zero_usage(self):
        tracker = CostTracker(
            whisper_model="gpt-4o-transcribe",
            translation_model="gpt-4.1",
        )
        summary = tracker.summary()
        assert "0.0 min" in summary
        assert "0 in" in summary
        assert "$0.0000" in summary

    def test_total_combines_both_costs(self):
        tracker = CostTracker(
            whisper_model="gpt-4o-transcribe",
            translation_model="gpt-4.1",
        )
        tracker.add_whisper(60.0)       # 1 min -> $0.006
        tracker.add_translation(1_000_000, 0)  # 1M in tokens -> $2.00

        summary = tracker.summary()
        # Total deberia ser $2.006
        assert "$2.0060 USD" in summary


class TestSessionLog:
    def test_add_entry(self):
        tracker = CostTracker()
        tracker.add_entry(0.0, 5.0, "hello", "hola")
        tracker.add_entry(5.0, 4.0, "world", "mundo")
        assert len(tracker.entries) == 2
        assert tracker.entries[0].transcription == "hello"
        assert tracker.entries[1].translation == "mundo"

    def test_save_markdown_creates_file(self, tmp_path):
        tracker = CostTracker(
            whisper_model="gpt-4o-transcribe",
            translation_model="gpt-4.1",
            source_lang="ja",
            target_lang="es",
        )
        tracker.add_entry(0.0, 5.0, "こんにちは", "Hola")
        tracker.add_entry(5.0, 4.0, "世界", "Mundo")

        path = tracker.save_markdown(tmp_path / "logs")
        assert path is not None
        assert path.exists()

        content = path.read_text(encoding="utf-8")
        assert "# LiveSubs" in content
        assert "ja -> es" in content
        assert "gpt-4o-transcribe" in content
        assert "gpt-4.1" in content
        assert "こんにちは" in content
        assert "Hola" in content
        assert "世界" in content
        assert "Mundo" in content
        assert "--- Resumen de costos ---" in content

    def test_save_markdown_returns_none_when_empty(self, tmp_path):
        tracker = CostTracker()
        assert tracker.save_markdown(tmp_path / "logs") is None

    def test_save_markdown_creates_logs_dir(self, tmp_path):
        logs_dir = tmp_path / "nested" / "logs"
        tracker = CostTracker()
        tracker.add_entry(0.0, 5.0, "text", "texto")

        path = tracker.save_markdown(logs_dir)
        assert logs_dir.exists()
        assert path.parent == logs_dir

    def test_save_markdown_escapes_pipe_in_content(self, tmp_path):
        tracker = CostTracker()
        tracker.add_entry(0.0, 5.0, "A | B", "A | B")

        path = tracker.save_markdown(tmp_path / "logs")
        content = path.read_text(encoding="utf-8")
        assert "A \\| B" in content

    def test_save_markdown_table_has_timestamps(self, tmp_path):
        tracker = CostTracker()
        tracker.add_entry(65.0, 5.0, "text", "texto")

        path = tracker.save_markdown(tmp_path / "logs")
        content = path.read_text(encoding="utf-8")
        assert "| 01:05 |" in content


class TestEscapeMdTable:
    def test_escapes_pipe(self):
        assert escape_md_table("A | B") == "A \\| B"

    def test_replaces_newline(self):
        assert escape_md_table("line1\nline2") == "line1 line2"

    def test_plain_text_unchanged(self):
        assert escape_md_table("hello world") == "hello world"
