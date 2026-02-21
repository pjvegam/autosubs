from livesubs.diarize import SpeakerSegment, _StreamingState


class TestSpeakerSegment:
    def test_creation(self):
        seg = SpeakerSegment(speaker=0, start=0.0, end=1.5)
        assert seg.speaker == 0
        assert seg.start == 0.0
        assert seg.end == 1.5


class TestGetSpeaker:
    """Tests para la logica de get_speaker sin instanciar DiarizationEngine."""

    def _get_speaker(
        self, annotations: list[SpeakerSegment], start: float, end: float,
    ) -> int | None:
        """Replica la logica de DiarizationEngine.get_speaker para testing."""
        if not annotations:
            return None

        speaker_overlap: dict[int, float] = {}
        for seg in annotations:
            overlap_start = max(seg.start, start)
            overlap_end = min(seg.end, end)
            if overlap_start < overlap_end:
                overlap = overlap_end - overlap_start
                speaker_overlap[seg.speaker] = speaker_overlap.get(seg.speaker, 0.0) + overlap

        if not speaker_overlap:
            return None

        return max(speaker_overlap, key=speaker_overlap.get)

    def test_returns_none_without_annotations(self):
        assert self._get_speaker([], 0.0, 5.0) is None

    def test_returns_dominant_speaker(self):
        annotations = [
            SpeakerSegment(speaker=0, start=0.0, end=1.5),
            SpeakerSegment(speaker=2, start=1.5, end=4.0),
            SpeakerSegment(speaker=1, start=4.0, end=5.0),
        ]
        # En rango [1.0, 5.0]: speaker 0 = 0.5s, speaker 2 = 2.5s, speaker 1 = 1.0s
        assert self._get_speaker(annotations, 1.0, 5.0) == 2

    def test_returns_none_no_overlap(self):
        annotations = [
            SpeakerSegment(speaker=0, start=0.0, end=1.0),
        ]
        assert self._get_speaker(annotations, 5.0, 10.0) is None

    def test_single_speaker_full_overlap(self):
        annotations = [
            SpeakerSegment(speaker=3, start=0.0, end=10.0),
        ]
        assert self._get_speaker(annotations, 2.0, 8.0) == 3

    def test_partial_overlap_at_boundaries(self):
        annotations = [
            SpeakerSegment(speaker=0, start=0.0, end=3.0),
            SpeakerSegment(speaker=1, start=3.0, end=6.0),
        ]
        # En rango [2.0, 4.0]: speaker 0 = 1.0s, speaker 1 = 1.0s -> empate, max() retorna 0 o 1
        result = self._get_speaker(annotations, 2.0, 4.0)
        assert result in (0, 1)

    def test_multiple_segments_same_speaker(self):
        annotations = [
            SpeakerSegment(speaker=0, start=0.0, end=1.0),
            SpeakerSegment(speaker=1, start=1.0, end=2.0),
            SpeakerSegment(speaker=0, start=2.0, end=4.0),
        ]
        # En rango [0.0, 4.0]: speaker 0 = 3.0s, speaker 1 = 1.0s
        assert self._get_speaker(annotations, 0.0, 4.0) == 0


class TestStreamingState:
    def test_default_values(self):
        state = _StreamingState()
        assert state.previous_chunk_features is None
        assert state.chunk_index == 0
        assert state.len_prediction is None
        assert state.annotations == []

    def test_annotations_mutable(self):
        state = _StreamingState()
        state.annotations.append(SpeakerSegment(speaker=0, start=0.0, end=1.0))
        assert len(state.annotations) == 1
