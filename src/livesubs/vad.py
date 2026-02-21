"""Segmentacion inteligente de audio usando WebRTC VAD."""

import logging
import wave
from dataclasses import dataclass
from pathlib import Path

import webrtcvad

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
FRAME_DURATION_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_DURATION_MS // 1000  # 320
FRAME_BYTES = FRAME_SAMPLES * 2  # 640 (16-bit PCM)


@dataclass
class AudioChunk:
    """Chunk de audio con su duracion real y ratio de voz."""

    path: Path
    duration: float  # segundos
    speech_ratio: float = 1.0  # 0.0=silencio, 1.0=todo voz


class VadSegmenter:
    """Segmenta audio PCM en chunks basados en deteccion de voz/silencio.

    Alimentar con feed() que retorna chunks listos cuando detecta silencios
    naturales entre frases. Rango flexible: min_duration → max_duration,
    cortando en silencios >= silence_duration cuando esta dentro del rango.
    """

    def __init__(
        self,
        output_dir: Path,
        *,
        min_duration: float = 3.0,
        target_duration: float = 5.0,
        max_duration: float = 8.0,
        silence_duration: float = 0.3,
        aggressiveness: int = 2,
    ) -> None:
        self.output_dir = output_dir
        self.min_duration = min_duration
        self.target_duration = target_duration
        self.max_duration = max_duration
        self.silence_duration = silence_duration

        self._vad = webrtcvad.Vad(aggressiveness)
        self._buffer = bytearray()
        self._processed = 0  # bytes analizados frame-por-frame dentro de _buffer
        self._silence_frames = 0
        self._speech_frames = 0
        self._total_frames = 0
        self._chunk_index = 0

        self._silence_frames_threshold = int(silence_duration / (FRAME_DURATION_MS / 1000))

    def feed(self, pcm_data: bytes) -> list[AudioChunk]:
        """Alimenta audio PCM 16kHz mono 16-bit. Retorna chunks listos."""
        self._buffer.extend(pcm_data)
        chunks: list[AudioChunk] = []

        while self._processed + FRAME_BYTES <= len(self._buffer):
            frame = bytes(self._buffer[self._processed : self._processed + FRAME_BYTES])
            is_speech = self._vad.is_speech(frame, SAMPLE_RATE)

            self._total_frames += 1
            if is_speech:
                self._silence_frames = 0
                self._speech_frames += 1
            else:
                self._silence_frames += 1

            self._processed += FRAME_BYTES
            duration = self._processed / (SAMPLE_RATE * 2)

            should_cut = duration >= self.max_duration or (
                duration >= self.min_duration
                and self._silence_frames >= self._silence_frames_threshold
            )

            if should_cut:
                chunk = self._flush_processed()
                if chunk is not None:
                    chunks.append(chunk)

        return chunks

    def flush_remaining(self, min_seconds: float = 1.0) -> AudioChunk | None:
        """Escribe el buffer restante si tiene suficiente duracion."""
        total_bytes = len(self._buffer)
        if total_bytes / (SAMPLE_RATE * 2) >= min_seconds:
            ratio = self._speech_frames / self._total_frames if self._total_frames else 0.0
            self._speech_frames = 0
            self._total_frames = 0
            return self._write_chunk_data(bytes(self._buffer), total_bytes, ratio)
        self._buffer.clear()
        self._processed = 0
        self._speech_frames = 0
        self._total_frames = 0
        return None

    def _flush_processed(self) -> AudioChunk | None:
        """Escribe los bytes procesados como WAV, conserva el resto."""
        if self._processed == 0:
            return None

        speech_ratio = self._speech_frames / self._total_frames if self._total_frames else 0.0
        chunk_data = bytes(self._buffer[: self._processed])
        self._buffer = self._buffer[self._processed :]
        self._processed = 0
        self._silence_frames = 0
        self._speech_frames = 0
        self._total_frames = 0

        return self._write_chunk_data(chunk_data, len(chunk_data), speech_ratio)

    def _write_chunk_data(
        self, data: bytes, num_bytes: int, speech_ratio: float = 1.0,
    ) -> AudioChunk | None:
        """Escribe datos PCM como WAV."""
        if not data:
            return None

        duration = num_bytes / (SAMPLE_RATE * 2)
        chunk_path = self.output_dir / f"chunk_{self._chunk_index:04d}.wav"

        with wave.open(str(chunk_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(data)

        logger.debug(
            "VAD chunk %s: %.2fs (speech: %.0f%%)", chunk_path.name, duration, speech_ratio * 100,
        )
        self._chunk_index += 1

        return AudioChunk(path=chunk_path, duration=duration, speech_ratio=speech_ratio)
