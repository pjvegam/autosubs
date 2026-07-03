"""Captura de audio del sistema via WASAPI loopback (Windows)."""

import asyncio
import audioop
import logging
import queue
import struct
import threading
from collections.abc import Callable
from pathlib import Path

import pyaudiowpatch as pyaudio

from autosubs.vad import AudioChunk, VadSegmenter

logger = logging.getLogger(__name__)


def _convert_to_16k_mono(
    data: bytes,
    channels: int,
    source_rate: int,
    target_rate: int = 16000,
) -> bytes:
    """Convierte audio PCM a mono 16kHz."""
    if channels > 1:
        data = audioop.tomono(data, 2, 1, 1)

    if source_rate != target_rate:
        ratio = source_rate // target_rate
        if source_rate % target_rate == 0 and ratio > 1:
            samples = struct.unpack(f"<{len(data) // 2}h", data)
            decimated = samples[::ratio]
            data = struct.pack(f"<{len(decimated)}h", *decimated)
        else:
            data, _ = audioop.ratecv(data, 2, 1, source_rate, target_rate, None)

    return data


class SystemAudioCapture:
    """Captura audio del sistema (loopback WASAPI) y genera chunks WAV via VAD."""

    def __init__(
        self,
        output_dir: Path,
        segment_duration: int = 5,
        *,
        vad_aggressiveness: int = 2,
        pcm_callback: Callable[[bytes], None] | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.segment_duration = segment_duration
        self._vad_aggressiveness = vad_aggressiveness
        self._pcm_callback = pcm_callback
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._chunk_queue: queue.Queue[AudioChunk | None] = queue.Queue()
        self._running = False

    def _find_loopback_device(self, pa: pyaudio.PyAudio) -> dict:
        """Encuentra el dispositivo loopback WASAPI por defecto."""
        wasapi_info = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_speakers = pa.get_device_info_by_index(wasapi_info["defaultOutputDevice"])

        for i in range(pa.get_device_count()):
            device = pa.get_device_info_by_index(i)
            if (
                device.get("isLoopbackDevice")
                and device["name"].startswith(default_speakers["name"].split(" (")[0])
            ):
                return device

        raise RuntimeError(
            "No se encontro dispositivo loopback WASAPI. "
            "Asegurate de tener un dispositivo de audio activo."
        )

    def _capture_loop(self) -> None:
        """Loop de captura que corre en un hilo separado."""
        pa = pyaudio.PyAudio()
        stream = None
        try:
            device = self._find_loopback_device(pa)
            device_rate = int(device["defaultSampleRate"])
            device_channels = device["maxInputChannels"]

            logger.info(
                "Capturando audio: %s (%dHz, %dch)",
                device["name"], device_rate, device_channels,
            )

            min_dur = self.segment_duration * 0.6
            max_dur = self.segment_duration * 1.6
            segmenter = VadSegmenter(
                self.output_dir,
                min_duration=min_dur,
                target_duration=float(self.segment_duration),
                max_duration=max_dur,
                aggressiveness=self._vad_aggressiveness,
            )

            stream = pa.open(
                format=pyaudio.paInt16,
                channels=device_channels,
                rate=device_rate,
                input=True,
                input_device_index=device["index"],
                frames_per_buffer=1024,
            )

            while not self._stop_event.is_set():
                try:
                    data = stream.read(1024, exception_on_overflow=False)
                except OSError as e:
                    logger.warning("Error leyendo audio: %s", e)
                    continue

                pcm = _convert_to_16k_mono(data, device_channels, device_rate)
                if self._pcm_callback is not None:
                    self._pcm_callback(pcm)
                for chunk in segmenter.feed(pcm):
                    self._chunk_queue.put(chunk)

            if remaining := segmenter.flush_remaining():
                self._chunk_queue.put(remaining)

        except Exception as e:
            logger.error("Error en captura de audio: %s", e)
        finally:
            if stream is not None:
                stream.stop_stream()
                stream.close()
            pa.terminate()
            self._chunk_queue.put(None)
            self._running = False

    async def start(self) -> None:
        """Inicia captura de loopback WASAPI en thread separado."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("Captura de audio del sistema iniciada")

    async def poll_chunks(self, async_queue: asyncio.Queue) -> None:
        """Drena chunks del thread de captura hacia la queue async."""
        while True:
            chunk = await asyncio.to_thread(self._chunk_queue.get)
            if chunk is None:
                break
            logger.debug("Chunk listo: %s (%.2fs)", chunk.path.name, chunk.duration)
            await async_queue.put(chunk)

        await async_queue.put(None)
        logger.info("Polling finalizado")

    def is_running(self) -> bool:
        return self._running

    async def stop(self) -> None:
        """Detiene la captura de audio."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            await asyncio.to_thread(self._thread.join, timeout=5)
        self._running = False
        logger.info("Captura de audio detenida")
