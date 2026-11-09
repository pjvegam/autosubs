
import json

import pytest

from autosubs.tedx import (
    apply_patch,
    build_utterances,
    clean_text,
    corpus_cer,
    edit_distance,
    load_vtt,
    normalize_for_scoring,
    utt_hash,
)

# Utterance real de TEDxJP-10K v1.1. Si clean_text o utt_hash cambian, este md5
# deja de coincidir y el dataset se reconstruye vacio sin avisar.
REAL_VIDEO_ID = "-6K2nN9aWsg"
REAL_TEXT = "今大学教員をやってるんですけど"
REAL_MD5 = "e85ef8accb029d8d890fdc5b76ef85b0"


class TestUttHash:
    def test_matches_official_dataset(self):
        assert utt_hash(REAL_VIDEO_ID, REAL_TEXT) == REAL_MD5

    def test_depends_on_video_id(self):
        assert utt_hash("otroVideoId", REAL_TEXT) != REAL_MD5


class TestCleanText:
    def test_strips_subtitle_credits(self):
        assert clean_text("字幕： 山田太郎") == ""

    def test_strips_translation_credits(self):
        assert clean_text("翻訳：田中") == ""

    def test_removes_subtitle_only_annotations(self):
        assert clean_text("そうですね（笑）") == "そうですね"

    def test_removes_bracket_insertions(self):
        assert clean_text("私［は］行く") == "私行く"

    def test_removes_punctuation(self):
        assert clean_text("これは、テストです。") == "これは テストです"

    def test_normalizes_wave_dash(self):
        assert clean_text("あ〜") == "あー"

    def test_halfwidth_to_fullwidth(self):
        assert clean_text("OK 123") == "ＯＫ １２３"

    def test_collapses_whitespace(self):
        assert clean_text("  あ   い  ") == "あ い"

    def test_drops_foreign_language_segments(self):
        assert clean_text("（英語）hello") == ""


class TestNormalizeForScoring:
    def test_removes_all_spaces(self):
        assert normalize_for_scoring("これは、テストです。 OK") == "これはテストですＯＫ"

    def test_reference_and_hypothesis_converge(self):
        # Mismo contenido, distinta puntuacion/espaciado del ASR
        assert normalize_for_scoring("今日はいい天気ですね") == normalize_for_scoring(
            "今日は、いい天気ですね。",
        )


class TestApplyPatch:
    def test_keeps_unchanged_text(self):
        assert apply_patch("あいう", ["", "", ""]) == "あいう"

    def test_deletes_character(self):
        assert apply_patch("あいう", ["", "- い", ""]) == "あう"

    def test_inserts_character(self):
        assert apply_patch("あう", ["", "+ い", ""]) == "あいう"

    def test_raises_when_patch_overruns(self):
        with pytest.raises(ValueError, match="excede"):
            apply_patch("あ", ["", ""])


class TestEditDistance:
    def test_identical(self):
        assert edit_distance("あいう", "あいう") == 0

    def test_substitution(self):
        assert edit_distance("あいう", "あXう") == 1

    def test_insertion(self):
        assert edit_distance("あう", "あいう") == 1

    def test_deletion(self):
        assert edit_distance("あいう", "あう") == 1

    def test_empty_hypothesis(self):
        assert edit_distance("あいう", "") == 3

    def test_empty_reference(self):
        assert edit_distance("", "あいう") == 3


class TestCorpusCer:
    def test_perfect_transcription(self):
        assert corpus_cer([("あいうえお", "あいうえお")]).cer == 0.0

    def test_single_substitution(self):
        assert corpus_cer([("あいうえお", "あいうXお")]).cer == pytest.approx(0.2)

    def test_aggregates_not_averages(self):
        # Promedio de CER por utterance seria (1.0 + 0.1)/2 = 0.55.
        # El agregado pondera por largo: (1 + 1) / (1 + 10) = 0.1818...
        pairs = [("あ", "X"), ("あいうえおかきくけこ", "あいうえおかきくけX")]
        result = corpus_cer(pairs)
        assert result.cer == pytest.approx(2 / 11)
        assert result.utterances == 2
        assert result.ref_chars == 11

    def test_empty_reference_does_not_divide_by_zero(self):
        assert corpus_cer([]).cer == 0.0


class TestLoadVtt:
    def _write(self, tmp_path, body):
        path = tmp_path / "vid.ja.vtt"
        path.write_text(body, encoding="utf8")
        return path

    def test_parses_cues(self, tmp_path):
        path = self._write(tmp_path, (
            "WEBVTT\n\n"
            "00:00:08.948 --> 00:00:13.655\n"
            "私が日本に来たのは\n\n"
        ))
        records = load_vtt(path)
        assert records == [(8.948, 13.655, "私が日本に来たのは")]

    def test_joins_multiline_cue(self, tmp_path):
        path = self._write(tmp_path, (
            "WEBVTT\n\n"
            "00:00:01.000 --> 00:00:02.000\n"
            "前半\n後半\n\n"
        ))
        assert load_vtt(path)[0][2] == "前半 後半"

    def test_parses_final_cue_without_trailing_blank_line(self, tmp_path):
        path = self._write(tmp_path, (
            "WEBVTT\n\n"
            "00:00:01.000 --> 00:00:02.000\n"
            "最後"
        ))
        assert len(load_vtt(path)) == 1

    def test_handles_hours(self, tmp_path):
        path = self._write(tmp_path, (
            "WEBVTT\n\n"
            "01:02:03.500 --> 01:02:04.000\n"
            "テスト\n\n"
        ))
        assert load_vtt(path)[0][0] == pytest.approx(3723.5)


class TestBuildUtterances:
    def _dataset(self, tmp_path, *, with_diff: bool) -> tuple:
        vtt_dir = tmp_path / "vtt"
        data_dir = tmp_path / "data"
        (data_dir / "diffs").mkdir(parents=True)
        vtt_dir.mkdir()

        (vtt_dir / "-6K2nN9aWsg.ja.vtt").write_text(
            "WEBVTT\n\n00:00:26.860 --> 00:00:29.400\n" + REAL_TEXT + "\n\n",
            encoding="utf8",
        )
        utt_id = f"{REAL_VIDEO_ID}-00002686-00002940"
        (data_dir / "utt_id_table.csv").write_text(
            f"{REAL_MD5} {utt_id}\n", encoding="utf8",
        )
        if with_diff:
            # Borra el primer caracter (今)
            patch = ["- 今"] + [""] * (len(REAL_TEXT) - 1)
            (data_dir / "diffs" / f"{utt_id}.diff").write_text(
                json.dumps(patch), encoding="utf8",
            )
        return vtt_dir, data_dir

    def test_reconstructs_utterance(self, tmp_path):
        vtt_dir, data_dir = self._dataset(tmp_path, with_diff=False)
        utterances, missing = build_utterances(vtt_dir, data_dir)

        assert missing == 0
        assert len(utterances) == 1
        utt = utterances[0]
        assert utt.text == REAL_TEXT
        assert utt.video_id == REAL_VIDEO_ID
        assert utt.start == pytest.approx(26.86)
        assert utt.end == pytest.approx(29.40)
        assert utt.duration == pytest.approx(2.54)

    def test_applies_human_correction(self, tmp_path):
        vtt_dir, data_dir = self._dataset(tmp_path, with_diff=True)
        utterances, _ = build_utterances(vtt_dir, data_dir)
        assert utterances[0].text == REAL_TEXT[1:]

    def test_reports_edited_subtitles_as_missing(self, tmp_path):
        vtt_dir, data_dir = self._dataset(tmp_path, with_diff=False)
        # Simula que YouTube edito el subtitulo despues de publicado el dataset
        (vtt_dir / "-6K2nN9aWsg.ja.vtt").write_text(
            "WEBVTT\n\n00:00:26.860 --> 00:00:29.400\nテキストが変わった\n\n",
            encoding="utf8",
        )
        utterances, missing = build_utterances(vtt_dir, data_dir)
        assert utterances == []
        assert missing == 1

    def test_video_ids_filter_excludes_video(self, tmp_path):
        # Un video con subtitulo bajado pero sin audio no debe generar utterances
        vtt_dir, data_dir = self._dataset(tmp_path, with_diff=False)
        utterances, missing = build_utterances(vtt_dir, data_dir, video_ids=set())
        assert utterances == []
        assert missing == 0

    def test_video_ids_filter_keeps_requested_video(self, tmp_path):
        vtt_dir, data_dir = self._dataset(tmp_path, with_diff=False)
        utterances, _ = build_utterances(vtt_dir, data_dir, video_ids={REAL_VIDEO_ID})
        assert len(utterances) == 1

    def test_ignores_videos_not_downloaded(self, tmp_path):
        vtt_dir, data_dir = self._dataset(tmp_path, with_diff=False)
        with (data_dir / "utt_id_table.csv").open("a", encoding="utf8") as f:
            f.write("00000000000000000000000000000000 otroVideoId-00000100-00000200\n")

        utterances, missing = build_utterances(vtt_dir, data_dir)
        assert len(utterances) == 1
        assert missing == 0  # no descargado != subtitulo cambiado
