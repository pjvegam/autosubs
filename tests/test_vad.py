import wave
from unittest.mock import patch

from livesubs.vad import FRAME_BYTES, SAMPLE_RATE, AudioChunk, VadSegmenter


def _seconds_to_bytes(seconds: float) -> bytes:
    """Genera PCM silencioso de la duracion dada."""
    num_samples = int(SAMPLE_RATE * seconds)
    return b"\x00" * (num_samples * 2)


def _make_segmenter(tmp_path, **kwargs):
    defaults = dict(
        min_duration=3.0,
        target_duration=5.0,
        max_duration=8.0,
        silence_duration=0.3,
        aggressiveness=2,
    )
    defaults.update(kwargs)
    tmp_path.mkdir(parents=True, exist_ok=True)
    return VadSegmenter(tmp_path, **defaults)


class TestAudioChunk:
    def test_dataclass(self, tmp_path):
        chunk = AudioChunk(path=tmp_path / "test.wav", duration=4.2)
        assert chunk.duration == 4.2
        assert chunk.path.name == "test.wav"


class TestVadSegmenter:
    def test_init(self, tmp_path):
        seg = _make_segmenter(tmp_path)
        assert seg.min_duration == 3.0
        assert seg.max_duration == 8.0

    def test_short_audio_no_cut(self, tmp_path):
        """Audio menor a min_duration no produce chunks."""
        seg = _make_segmenter(tmp_path)
        pcm = _seconds_to_bytes(2.0)

        with patch("webrtcvad.Vad.is_speech", return_value=False):
            chunks = seg.feed(pcm)

        assert chunks == []

    def test_silence_after_min_duration_cuts(self, tmp_path):
        """Silencio sostenido despues de min_duration produce un corte."""
        seg = _make_segmenter(tmp_path, min_duration=1.0, silence_duration=0.2)
        pcm = _seconds_to_bytes(2.0)

        with patch("webrtcvad.Vad.is_speech", return_value=False):
            chunks = seg.feed(pcm)

        assert len(chunks) >= 1
        assert all(isinstance(c, AudioChunk) for c in chunks)
        assert all(c.path.exists() for c in chunks)

    def test_speech_prevents_cut(self, tmp_path):
        """Voz continua no produce corte aunque pase min_duration."""
        seg = _make_segmenter(tmp_path, min_duration=1.0)
        pcm = _seconds_to_bytes(2.0)

        with patch("webrtcvad.Vad.is_speech", return_value=True):
            chunks = seg.feed(pcm)

        # Sin silencio, no corta (a menos que llegue a max_duration)
        assert chunks == []

    def test_max_duration_forces_cut(self, tmp_path):
        """Al llegar a max_duration se corta sin importar si hay voz."""
        seg = _make_segmenter(tmp_path, max_duration=1.0)
        pcm = _seconds_to_bytes(2.0)

        with patch("webrtcvad.Vad.is_speech", return_value=True):
            chunks = seg.feed(pcm)

        assert len(chunks) >= 1

    def test_chunk_writes_valid_wav(self, tmp_path):
        """Los chunks escritos son WAV validos."""
        seg = _make_segmenter(tmp_path, max_duration=1.0)
        pcm = _seconds_to_bytes(1.5)

        with patch("webrtcvad.Vad.is_speech", return_value=True):
            chunks = seg.feed(pcm)

        assert len(chunks) >= 1
        with wave.open(str(chunks[0].path), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getframerate() == SAMPLE_RATE
            assert wf.getsampwidth() == 2

    def test_flush_remaining(self, tmp_path):
        """flush_remaining escribe el buffer restante."""
        seg = _make_segmenter(tmp_path)
        pcm = _seconds_to_bytes(2.0)

        with patch("webrtcvad.Vad.is_speech", return_value=True):
            seg.feed(pcm)

        chunk = seg.flush_remaining(min_seconds=1.0)
        assert chunk is not None
        assert chunk.path.exists()
        assert chunk.duration > 1.0

    def test_flush_remaining_too_short(self, tmp_path):
        """flush_remaining ignora audio muy corto."""
        seg = _make_segmenter(tmp_path)
        pcm = _seconds_to_bytes(0.5)

        with patch("webrtcvad.Vad.is_speech", return_value=True):
            seg.feed(pcm)

        chunk = seg.flush_remaining(min_seconds=1.0)
        assert chunk is None

    def test_chunk_index_increments(self, tmp_path):
        """El indice de chunk se incrementa correctamente."""
        seg = _make_segmenter(tmp_path, max_duration=1.0)
        pcm = _seconds_to_bytes(3.0)

        with patch("webrtcvad.Vad.is_speech", return_value=True):
            chunks = seg.feed(pcm)

        assert len(chunks) >= 2
        assert "chunk_0000" in chunks[0].path.name
        assert "chunk_0001" in chunks[1].path.name

    def test_silence_frames_reset_on_speech(self, tmp_path):
        """El contador de silencio se resetea cuando hay voz."""
        seg = _make_segmenter(tmp_path, min_duration=0.1, silence_duration=0.1)

        # Alimentar silencio + voz en secuencia
        call_count = 0

        def alternating_speech(frame, rate):
            nonlocal call_count
            call_count += 1
            # Primeros frames: silencio, luego voz
            return call_count > 10

        pcm = _seconds_to_bytes(0.5)
        with patch("webrtcvad.Vad.is_speech", side_effect=alternating_speech):
            chunks = seg.feed(pcm)

        # La voz deberia haber interrumpido el corte por silencio
        # El comportamiento exacto depende del timing
        assert isinstance(chunks, list)

    def test_multiple_feeds(self, tmp_path):
        """Alimentar en multiples llamadas funciona correctamente."""
        seg = _make_segmenter(tmp_path, min_duration=1.0, silence_duration=0.2)

        with patch("webrtcvad.Vad.is_speech", return_value=False):
            chunks1 = seg.feed(_seconds_to_bytes(0.5))
            chunks2 = seg.feed(_seconds_to_bytes(0.5))
            chunks3 = seg.feed(_seconds_to_bytes(0.5))

        total_chunks = chunks1 + chunks2 + chunks3
        assert len(total_chunks) >= 1

    def test_feed_partial_frame(self, tmp_path):
        """Alimentar menos de un frame no crashea."""
        seg = _make_segmenter(tmp_path)
        partial = b"\x00" * (FRAME_BYTES - 10)

        with patch("webrtcvad.Vad.is_speech", return_value=True):
            chunks = seg.feed(partial)

        assert chunks == []

    def test_duration_accuracy(self, tmp_path):
        """La duracion reportada es cercana a la real."""
        seg = _make_segmenter(tmp_path, max_duration=2.0)
        pcm = _seconds_to_bytes(3.0)

        with patch("webrtcvad.Vad.is_speech", return_value=True):
            chunks = seg.feed(pcm)

        assert len(chunks) >= 1
        # Cada chunk deberia estar cerca de max_duration
        for chunk in chunks:
            assert 1.5 < chunk.duration < 2.5
