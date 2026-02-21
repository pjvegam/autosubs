import asyncio
import struct
import wave
from unittest.mock import MagicMock

import pytest

from livesubs.capture import SystemAudioCapture


class TestSystemAudioCapture:
    def test_init(self, tmp_path):
        capture = SystemAudioCapture(tmp_path / "audio", segment_duration=5)
        assert capture.segment_duration == 5
        assert not capture.is_running()

    def test_write_chunk_creates_wav(self, tmp_path):
        capture = SystemAudioCapture(tmp_path, segment_duration=5)
        capture.output_dir.mkdir(parents=True, exist_ok=True)

        # Generar 1 segundo de silencio a 48kHz estereo (16-bit)
        num_samples = 48000
        stereo_data = struct.pack(f"<{num_samples * 2}h", *([0] * num_samples * 2))

        capture._write_chunk(stereo_data, channels=2, source_rate=48000, target_rate=16000)

        chunk_path = tmp_path / "chunk_0000.wav"
        assert chunk_path.exists()

        with wave.open(str(chunk_path), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getframerate() == 16000
            assert wf.getsampwidth() == 2

    def test_write_chunk_increments_index(self, tmp_path):
        capture = SystemAudioCapture(tmp_path, segment_duration=5)
        capture.output_dir.mkdir(parents=True, exist_ok=True)

        # Mono a 16kHz, sin conversion necesaria
        mono_data = struct.pack("<16000h", *([0] * 16000))

        capture._write_chunk(mono_data, channels=1, source_rate=16000, target_rate=16000)
        assert capture._chunk_index == 1

        capture._write_chunk(mono_data, channels=1, source_rate=16000, target_rate=16000)
        assert capture._chunk_index == 2

        assert (tmp_path / "chunk_0000.wav").exists()
        assert (tmp_path / "chunk_0001.wav").exists()

    def test_write_chunk_resamples_non_integer_ratio(self, tmp_path):
        capture = SystemAudioCapture(tmp_path, segment_duration=5)
        capture.output_dir.mkdir(parents=True, exist_ok=True)

        # 44100Hz mono -> 16000Hz (ratio no entero)
        num_samples = 44100
        mono_data = struct.pack(f"<{num_samples}h", *([0] * num_samples))

        capture._write_chunk(mono_data, channels=1, source_rate=44100, target_rate=16000)

        chunk_path = tmp_path / "chunk_0000.wav"
        assert chunk_path.exists()
        with wave.open(str(chunk_path), "rb") as wf:
            assert wf.getframerate() == 16000

    async def test_poll_chunks_detects_files(self, tmp_path):
        capture = SystemAudioCapture(tmp_path, segment_duration=5)
        capture._running = True

        # Crear 2 chunks y luego parar
        mono_data = struct.pack("<16000h", *([0] * 16000))
        for i in range(2):
            path = tmp_path / f"chunk_{i:04d}.wav"
            with wave.open(str(path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(mono_data)

        # Simular que la captura se detiene despues de crear los chunks
        async def _stop_after_delay():
            await asyncio.sleep(0.5)
            capture._running = False

        queue: asyncio.Queue = asyncio.Queue()
        stop_task = asyncio.create_task(_stop_after_delay())
        await capture.poll_chunks(queue)
        await stop_task

        items = []
        while not queue.empty():
            items.append(await queue.get())

        # chunk_0000.wav, chunk_0001.wav (ultimo), None (sentinel)
        assert len(items) == 3
        assert items[0] == tmp_path / "chunk_0000.wav"
        assert items[1] == tmp_path / "chunk_0001.wav"
        assert items[2] is None

    async def test_poll_chunks_sends_sentinel_when_stopped(self, tmp_path):
        capture = SystemAudioCapture(tmp_path, segment_duration=5)
        capture._running = False  # Ya detenido

        queue: asyncio.Queue = asyncio.Queue()
        await capture.poll_chunks(queue)

        sentinel = await queue.get()
        assert sentinel is None

    async def test_stop_sets_event(self, tmp_path):
        capture = SystemAudioCapture(tmp_path, segment_duration=5)
        # Sin thread activo, solo verifica que no crashea
        await capture.stop()
        assert capture._stop_event.is_set()
        assert not capture.is_running()

    def test_find_loopback_device_raises_on_no_device(self, tmp_path):
        capture = SystemAudioCapture(tmp_path)

        pa_mock = MagicMock()
        pa_mock.get_host_api_info_by_type.return_value = {"defaultOutputDevice": 0}
        pa_mock.get_device_info_by_index.side_effect = [
            {"name": "Speakers (Realtek)", "maxInputChannels": 2},  # default speakers
            {"name": "Mic", "isLoopbackDevice": False, "maxInputChannels": 1},
        ]
        pa_mock.get_device_count.return_value = 1

        with pytest.raises(RuntimeError, match="No se encontro dispositivo loopback"):
            capture._find_loopback_device(pa_mock)

    def test_find_loopback_device_finds_match(self, tmp_path):
        capture = SystemAudioCapture(tmp_path)

        loopback_device = {
            "index": 3,
            "name": "Speakers (Realtek) [Loopback]",
            "isLoopbackDevice": True,
            "maxInputChannels": 2,
            "defaultSampleRate": 48000.0,
        }

        pa_mock = MagicMock()
        pa_mock.get_host_api_info_by_type.return_value = {"defaultOutputDevice": 0}
        pa_mock.get_device_info_by_index.side_effect = [
            {"name": "Speakers (Realtek)", "maxInputChannels": 2},  # default speakers
            {"name": "Mic", "isLoopbackDevice": False},  # device 0
            loopback_device,  # device 1
        ]
        pa_mock.get_device_count.return_value = 2

        result = capture._find_loopback_device(pa_mock)
        assert result["index"] == 3
        assert result["isLoopbackDevice"] is True
