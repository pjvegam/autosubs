
from autosubs.subtitles import SubtitleManager, _format_timestamp


class TestFormatTimestamp:
    def test_zero(self):
        assert _format_timestamp(0.0) == "00:00:00,000"

    def test_seconds_only(self):
        assert _format_timestamp(5.0) == "00:00:05,000"

    def test_with_millis(self):
        assert _format_timestamp(5.5) == "00:00:05,500"

    def test_minutes(self):
        assert _format_timestamp(65.0) == "00:01:05,000"

    def test_hours(self):
        assert _format_timestamp(3661.123) == "01:01:01,123"

    def test_large_value(self):
        assert _format_timestamp(36000.0) == "10:00:00,000"

    def test_fractional_millis(self):
        # 1.9999 -> 1999ms -> deberia truncar a 999
        assert _format_timestamp(1.999) == "00:00:01,999"


class TestSubtitleManager:
    def test_creates_empty_file(self, tmp_path):
        srt = tmp_path / "test.srt"
        SubtitleManager(srt)
        assert srt.exists()
        assert srt.read_text(encoding="utf-8") == ""

    def test_add_single_entry(self, tmp_path):
        srt = tmp_path / "test.srt"
        manager = SubtitleManager(srt)
        manager.add_entry(0.0, 10.0, "Hola mundo")

        content = srt.read_text(encoding="utf-8")
        assert "1\n" in content
        assert "00:00:00,000 --> 00:00:10,000\n" in content
        assert "Hola mundo\n" in content

    def test_add_multiple_entries(self, tmp_path):
        srt = tmp_path / "test.srt"
        manager = SubtitleManager(srt)
        manager.add_entry(0.0, 10.0, "Primera linea")
        manager.add_entry(10.0, 20.0, "Segunda linea")

        content = srt.read_text(encoding="utf-8")
        lines = content.strip().split("\n\n")
        assert len(lines) == 2
        assert lines[0].startswith("1\n")
        assert lines[1].startswith("2\n")

    def test_entry_format_is_valid_srt(self, tmp_path):
        srt = tmp_path / "test.srt"
        manager = SubtitleManager(srt)
        manager.add_entry(61.5, 71.5, "Texto de prueba")

        content = srt.read_text(encoding="utf-8")
        expected = (
            "1\n"
            "00:01:01,500 --> 00:01:11,500\n"
            "Texto de prueba\n"
            "\n"
        )
        assert content == expected

    def test_append_does_not_overwrite(self, tmp_path):
        srt = tmp_path / "test.srt"
        manager = SubtitleManager(srt)
        manager.add_entry(0.0, 10.0, "Primero")
        manager.add_entry(10.0, 20.0, "Segundo")

        content = srt.read_text(encoding="utf-8")
        assert "Primero" in content
        assert "Segundo" in content
