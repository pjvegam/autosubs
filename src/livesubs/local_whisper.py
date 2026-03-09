"""Wrapper de faster-whisper para transcripcion local con GPU."""

from __future__ import annotations

import ctypes
import logging
import sys
from pathlib import Path

from livesubs.transcribe import TranscriptSegment, _is_hallucination

logger = logging.getLogger(__name__)

_CUDA_DLLS = ["cublas64_12.dll", "cublasLt64_12.dll"]


def _find_site_packages() -> Path | None:
    """Encuentra el directorio site-packages del venv actual."""
    for p in sys.path:
        sp = Path(p)
        if sp.name == "site-packages" and sp.is_dir():
            return sp
    return None


def _preload_cuda_dlls() -> None:
    """Pre-carga DLLs CUDA desde nvidia pip packages en Windows."""
    if sys.platform != "win32":
        return
    site_packages = _find_site_packages()
    if site_packages is None:
        return
    nvidia_dir = site_packages / "nvidia"
    if not nvidia_dir.is_dir():
        return
    for dll_name in _CUDA_DLLS:
        for dll_path in nvidia_dir.rglob(dll_name):
            try:
                ctypes.WinDLL(str(dll_path))
                logger.debug("Pre-cargada: %s", dll_path)
            except OSError:
                logger.debug("No se pudo cargar: %s", dll_path)


class LocalWhisperModel:
    """Modelo faster-whisper local. Constructor y transcribe son BLOCKING."""

    def __init__(
        self,
        model_size: str = "turbo",
        device: str = "cuda",
        compute_type: str = "float16",
    ) -> None:
        _preload_cuda_dlls()
        from faster_whisper import WhisperModel

        logger.info(
            "Cargando faster-whisper %s (device=%s, compute=%s)...",
            model_size, device, compute_type,
        )
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        logger.info("faster-whisper cargado")

    def transcribe(self, audio_path: Path, language: str) -> list[TranscriptSegment]:
        """Transcribe audio con faster-whisper. BLOCKING — usar con asyncio.to_thread()."""
        segments_gen, info = self.model.transcribe(
            str(audio_path),
            language=language,
            beam_size=5,
            hallucination_silence_threshold=2.0,
        )

        segments: list[TranscriptSegment] = []
        for seg in segments_gen:
            text = seg.text.strip()
            if _is_hallucination(text):
                continue
            segments.append(TranscriptSegment(text=text, start=seg.start, end=seg.end))

        logger.info(
            "faster-whisper [%s]: %d segments (lang=%s, prob=%.2f)",
            audio_path.name, len(segments), info.language, info.language_probability,
        )
        return segments
