"""Captura de audio del sistema via WASAPI loopback (Windows)."""

import asyncio
import logging
import struct
import threading
import wave
from pathlib import Path

import audioop
import pyaudiowpatch as pyaudio

logger = logging.getLogger(__name__)


class SystemAudioCapture:
    """Captura audio del sistema (loopback WASAPI) y genera chunks WAV."""

    def __init__(self, output_dir: Path, segment_duration: int = 5) -> None:
        self.output_dir = output_dir
        self.segment_duration = segment_duration
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._chunk_index = 0
        self._poll_index = 0
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

            frames_per_segment = device_rate * self.segment_duration
            target_rate = 16000

            stream = pa.open(
                format=pyaudio.paInt16,
                channels=device_channels,
                rate=device_rate,
                input=True,
                input_device_index=device["index"],
                frames_per_buffer=1024,
            )

            buffer = b""
            bytes_per_frame = device_channels * 2  # 16-bit = 2 bytes per sample
            bytes_per_segment = frames_per_segment * bytes_per_frame

            while not self._stop_event.is_set():
                try:
                    data = stream.read(1024, exception_on_overflow=False)
                    buffer += data
                except OSError as e:
                    logger.warning("Error leyendo audio: %s", e)
                    continue

                if len(buffer) >= bytes_per_segment:
                    segment_data = buffer[:bytes_per_segment]
                    buffer = buffer[bytes_per_segment:]
                    self._write_chunk(segment_data, device_channels, device_rate, target_rate)

            # Escribir lo que quede en el buffer si es suficiente (>1s)
            min_bytes = device_rate * bytes_per_frame  # 1 segundo minimo
            if len(buffer) >= min_bytes:
                self._write_chunk(buffer, device_channels, device_rate, target_rate)

        except Exception as e:
            logger.error("Error en captura de audio: %s", e)
        finally:
            if stream is not None:
                stream.stop_stream()
                stream.close()
            pa.terminate()
            self._running = False

    def _write_chunk(
        self,
        data: bytes,
        channels: int,
        source_rate: int,
        target_rate: int,
    ) -> None:
        """Convierte a mono 16kHz y escribe un chunk WAV."""
        # Estereo a mono
        if channels > 1:
            data = audioop.tomono(data, 2, 1, 1)

        # Resamplear a 16kHz
        if source_rate != target_rate:
            ratio = source_rate // target_rate
            if source_rate % target_rate == 0 and ratio > 1:
                # Decimation simple: tomar cada N-esima muestra
                samples = struct.unpack(f"<{len(data) // 2}h", data)
                decimated = samples[::ratio]
                data = struct.pack(f"<{len(decimated)}h", *decimated)
            else:
                data, _ = audioop.ratecv(data, 2, 1, source_rate, target_rate, None)

        chunk_path = self.output_dir / f"chunk_{self._chunk_index:04d}.wav"
        with wave.open(str(chunk_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(target_rate)
            wf.writeframes(data)

        logger.debug("Chunk escrito: %s (%d bytes)", chunk_path.name, len(data))
        self._chunk_index += 1

    async def start(self) -> None:
        """Inicia captura de loopback WASAPI en thread separado."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("Captura de audio del sistema iniciada")

    async def poll_chunks(self, queue: asyncio.Queue) -> None:
        """Detecta nuevos chunks WAV y los encola."""
        stall_seconds = 0
        while True:
            expected = self.output_dir / f"chunk_{self._poll_index:04d}.wav"
            next_chunk = self.output_dir / f"chunk_{self._poll_index + 1:04d}.wav"

            if next_chunk.exists():
                logger.debug("Chunk listo: %s", expected.name)
                await queue.put(expected)
                self._poll_index += 1
                stall_seconds = 0
            elif not self.is_running() and expected.exists():
                logger.debug("Ultimo chunk: %s", expected.name)
                await queue.put(expected)
                self._poll_index += 1
                break
            elif not self.is_running():
                logger.warning("Captura termino sin producir mas chunks")
                break
            else:
                stall_seconds += 1
                if stall_seconds == 30:
                    logger.warning("Sin chunks nuevos por 30s")
                await asyncio.sleep(1)

        await queue.put(None)
        logger.info("Polling finalizado (%d chunks procesados)", self._poll_index)

    def is_running(self) -> bool:
        return self._running

    async def stop(self) -> None:
        """Detiene la captura de audio."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            await asyncio.to_thread(self._thread.join, timeout=5)
        self._running = False
        logger.info("Captura de audio detenida")
