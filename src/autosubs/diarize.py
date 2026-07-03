"""Diarizacion de speakers en streaming con NVIDIA Sortformer.

Requiere instalacion opcional: pip install autosubs[diarize]
Basado en WhisperLiveKit sortformer_backend.py.
"""

from __future__ import annotations

import logging
import os
import queue
import struct
import threading
import warnings
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
FRAME_DURATION_S = 0.08  # 80ms por frame de Sortformer


@dataclass
class SpeakerSegment:
    """Segmento temporal con speaker asignado."""

    speaker: int
    start: float
    end: float


def _suppress_nemo_noise() -> None:
    """Silencia warnings y logs ruidosos de NeMo antes de importarlo."""
    # NeMo usa su propio handler via nemo.utils.logging (nemo_logging)
    # que escribe directo a stderr. Subimos el nivel ANTES del import.
    os.environ.setdefault("NEMO_LOG_LEVEL", "ERROR")

    # Silenciar SyntaxWarnings de lhotse/pydub (invalid escape sequences)
    warnings.filterwarnings("ignore", category=SyntaxWarning)

    # Loggers de terceros ruidosos
    for name in (
        "nemo", "nemo.utils", "nemo.utils.nemo_logging",
        "nv_one_logger", "nv_one_logger.api.config",
        "nv_one_logger.exporter.export_config_manager",
        "nv_one_logger.training_telemetry.api.training_telemetry_provider",
        "numexpr", "numexpr.utils",
        "torch.distributed", "torch.distributed.elastic",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)


class SortformerModel:
    """Carga y configura Sortformer una sola vez (singleton compartido)."""

    def __init__(self, *, vod: bool = False, device: str | None = None) -> None:
        # Silenciar ANTES de importar NeMo
        _suppress_nemo_noise()

        import torch
        from nemo.collections.asr.models import SortformerEncLabelModel

        logger.info("Cargando modelo Sortformer...")
        self.model = SortformerEncLabelModel.from_pretrained(
            "nvidia/diar_streaming_sortformer_4spk-v2",
        )
        self.model.eval()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)

        sm = self.model.sortformer_modules
        sm.subsampling_factor = 10
        if vod:
            # Alta latencia, mas contexto → mejor precision
            sm.chunk_len = 340
            sm.chunk_right_context = 40
            sm.chunk_left_context = 340
        else:
            # Baja latencia para lives
            sm.chunk_len = 10
            sm.chunk_right_context = 0
            sm.chunk_left_context = 10
        sm.spkcache_len = 188
        sm.fifo_len = 188
        sm.spkcache_update_period = 144
        sm.log = False
        sm._check_streaming_parameters()

        logger.info("Sortformer cargado en %s (vod=%s)", self.device, vod)


@dataclass
class _StreamingState:
    """Estado mutable del streaming por sesion."""

    previous_chunk_features: Any = None  # torch.Tensor | None
    chunk_index: int = 0
    len_prediction: int | None = None
    annotations: list[SpeakerSegment] = field(default_factory=list)


class DiarizationEngine:
    """Motor de diarizacion streaming con Sortformer."""

    def __init__(self, shared_model: SortformerModel, *, wait_timeout: float = 5.0) -> None:
        import torch
        from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor
        from nemo.collections.asr.modules.sortformer_modules import StreamingSortformerState

        self._model = shared_model.model
        self._device = shared_model.device
        self._wait_timeout = wait_timeout
        self._pcm_queue: queue.Queue[bytes | None] = queue.Queue()
        self._lock = threading.Lock()
        self._state = _StreamingState()
        self._thread: threading.Thread | None = None
        self._progress_event = threading.Event()

        # Preprocesador mel independiente por sesion
        self._audio2mel = AudioToMelSpectrogramPreprocessor(
            window_size=0.025,
            normalize="NA",
            n_fft=512,
            features=128,
            pad_to=0,
        )
        self._audio2mel.to(self._device)

        # Duracion del chunk en segundos y samples
        self._chunk_duration_s = (
            self._model.sortformer_modules.chunk_len
            * self._model.sortformer_modules.subsampling_factor
            * self._model.preprocessor._cfg.window_stride
        )
        self._chunk_samples = int(self._chunk_duration_s * SAMPLE_RATE)

        # Estado de streaming de NeMo
        sm = self._model.sortformer_modules
        state = StreamingSortformerState()
        state.spkcache = torch.zeros(
            (1, sm.spkcache_len, sm.fc_d_model), device=self._device,
        )
        state.spkcache_preds = torch.zeros(
            (1, sm.spkcache_len, sm.n_spk), device=self._device,
        )
        state.spkcache_lengths = torch.zeros((1,), dtype=torch.long, device=self._device)
        state.fifo = torch.zeros(
            (1, sm.fifo_len, sm.fc_d_model), device=self._device,
        )
        state.fifo_lengths = torch.zeros((1,), dtype=torch.long, device=self._device)
        state.mean_sil_emb = torch.zeros(
            (1, sm.fc_d_model), device=self._device,
        )
        state.n_sil_frames = torch.zeros((1,), dtype=torch.long, device=self._device)
        self._streaming_state = state
        self._total_preds = torch.zeros(
            (1, 0, sm.n_spk), device=self._device,
        )

    def start(self) -> None:
        """Lanza thread de procesamiento."""
        self._thread = threading.Thread(target=self._processing_loop, daemon=True)
        self._thread.start()
        logger.info("DiarizationEngine iniciado")

    def push_pcm(self, data: bytes) -> None:
        """Thread-safe: encola PCM int16 16kHz mono."""
        self._pcm_queue.put(data)

    def get_speaker(self, start: float, end: float) -> int | None:
        """Thread-safe: retorna speaker dominante (0-3) para el rango temporal.

        Si las anotaciones aun no cubren el rango, espera hasta ``wait_timeout``
        para que el thread de diarizacion alcance.
        """
        # Esperar a que la diarizacion cubra al menos el inicio del rango
        deadline = start
        waited = 0.0
        step = 0.1
        while waited < self._wait_timeout:
            with self._lock:
                annotations = self._state.annotations
                if annotations and annotations[-1].end >= deadline:
                    break
            self._progress_event.wait(timeout=step)
            self._progress_event.clear()
            waited += step

        with self._lock:
            annotations = list(self._state.annotations)

        if not annotations:
            return None

        # Calcular overlap de cada speaker con el rango [start, end]
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

    async def stop(self) -> None:
        """Senaliza fin y espera al thread."""
        import asyncio

        self._pcm_queue.put(None)
        if self._thread is not None and self._thread.is_alive():
            await asyncio.to_thread(self._thread.join, timeout=10)
        logger.info("DiarizationEngine detenido")

    def _processing_loop(self) -> None:
        """Thread principal: lee PCM, procesa chunks con Sortformer."""
        pcm_buffer = bytearray()

        try:
            while True:
                data = self._pcm_queue.get()
                if data is None:
                    break

                pcm_buffer.extend(data)
                threshold_bytes = self._chunk_samples * 2  # int16 = 2 bytes

                while len(pcm_buffer) >= threshold_bytes:
                    chunk_bytes = bytes(pcm_buffer[:threshold_bytes])
                    del pcm_buffer[:threshold_bytes]
                    self._process_chunk(chunk_bytes)

        except Exception as e:
            logger.error("Error en DiarizationEngine: %s", e)

    def _process_chunk(self, pcm_bytes: bytes) -> None:
        """Procesa un chunk de audio con Sortformer streaming."""
        import numpy as np
        import torch

        # int16 -> float32 normalizado
        n_samples = len(pcm_bytes) // 2
        samples = struct.unpack(f"<{n_samples}h", pcm_bytes)
        audio_np = np.array(samples, dtype=np.float32) / 32768.0

        audio_tensor = torch.tensor(audio_np, device=self._device).unsqueeze(0)
        audio_length = torch.tensor([audio_tensor.shape[1]], device=self._device)

        # Mel spectrogram
        mel_features, _mel_lengths = self._audio2mel.get_features(audio_tensor, audio_length)

        # Left context: ultimos 99 frames del chunk anterior
        state = self._state
        if state.previous_chunk_features is not None:
            context = state.previous_chunk_features[:, :, -99:]
            total_features = torch.cat([context, mel_features], dim=2)
        else:
            total_features = mel_features

        state.previous_chunk_features = mel_features

        # Transponer a (batch, time, features)
        chunk_feat = torch.transpose(total_features, 1, 2)

        # Forward streaming
        with torch.inference_mode():
            left_offset = 8 if state.chunk_index > 0 else 0
            right_offset = 8

            self._streaming_state, self._total_preds = self._model.forward_streaming_step(
                processed_signal=chunk_feat,
                processed_signal_length=torch.tensor(
                    [chunk_feat.shape[1]], device=self._device,
                ),
                streaming_state=self._streaming_state,
                total_preds=self._total_preds,
                left_offset=left_offset,
                right_offset=right_offset,
            )

        # Post-procesar predicciones
        self._post_process_predictions()
        state.chunk_index += 1

        # Notificar a get_speaker que hay nuevas anotaciones
        self._progress_event.set()

    def _post_process_predictions(self) -> None:
        """Convierte predicciones del modelo en segmentos de speaker."""
        import numpy as np

        preds_np = self._total_preds[0].cpu().numpy()
        active_speakers = np.argmax(preds_np, axis=1)

        state = self._state
        if state.len_prediction is None:
            state.len_prediction = len(active_speakers)
            if state.len_prediction == 0:
                return

        frame_duration = self._chunk_duration_s / state.len_prediction
        current_preds = active_speakers[-state.len_prediction:]

        base_time = state.chunk_index * self._chunk_duration_s
        new_segments: list[SpeakerSegment] = []

        current_spk = int(current_preds[0])
        start_time = round(base_time, 3)

        for idx, spk in enumerate(current_preds):
            spk = int(spk)
            current_time = round(base_time + idx * frame_duration, 3)
            if spk != current_spk:
                new_segments.append(SpeakerSegment(
                    speaker=current_spk, start=start_time, end=current_time,
                ))
                start_time = current_time
                current_spk = spk

        # Segmento final
        end_time = round(base_time + len(current_preds) * frame_duration, 3)
        new_segments.append(SpeakerSegment(
            speaker=current_spk, start=start_time, end=end_time,
        ))

        with self._lock:
            state.annotations.extend(new_segments)
