import asyncio
import struct
from unittest.mock import MagicMock, patch

import pytest

from autosubs.capture import SystemAudioCapture, _convert_to_16k_mono
from autosubs.vad import AudioChunk


class TestConvertTo16kMono:
    def test_stereo_to_mono(self):
        # 100 samples estereo
        stereo = struct.pack("<200h", *([1000, -1000] * 100))
        result = _convert_to_16k_mono(stereo, channels=2, source_rate=16000)
        # Mono: promedio de L y R = 0
        samples = struct.unpack(f"<{len(result) // 2}h", result)
        assert len(samples) == 100

    def test_resample_integer_ratio(self):
        # 48000Hz → 16000Hz (ratio 3)
        num_samples = 48000
        mono = struct.pack(f"<{num_samples}h", *([0] * num_samples))
        result = _convert_to_16k_mono(mono, channels=1, source_rate=48000)
        samples = struct.unpack(f"<{len(result) // 2}h", result)
        assert len(samples) == 16000

    def test_resample_non_integer_ratio(self):
        # 44100Hz → 16000Hz (ratio no entero)
        num_samples = 44100
        mono = struct.pack(f"<{num_samples}h", *([0] * num_samples))
        result = _convert_to_16k_mono(mono, channels=1, source_rate=44100)
        # Deberia estar cerca de 16000 samples
        samples = struct.unpack(f"<{len(result) // 2}h", result)
        assert 15000 < len(samples) < 17000

    def test_no_conversion_needed(self):
        mono = struct.pack("<16000h", *([0] * 16000))
        result = _convert_to_16k_mono(mono, channels=1, source_rate=16000)
        assert result == mono


class TestSystemAudioCapture:
    def test_init(self, tmp_path):
        capture = SystemAudioCapture(tmp_path / "audio", segment_duration=5)
        assert capture.segment_duration == 5
        assert not capture.is_running()

    async def test_start_sets_running(self, tmp_path):
        capture = SystemAudioCapture(tmp_path / "audio", segment_duration=5)

        with patch.object(capture, "_capture_loop"):
            capture._running = True
            capture.output_dir.mkdir(parents=True, exist_ok=True)

        # Verificar que start crea el directorio
        await capture.start()
        assert capture.output_dir.exists()
        assert capture.is_running()
        # Cleanup
        capture._stop_event.set()
        capture._running = False

    async def test_poll_chunks_drains_queue(self, tmp_path):
        capture = SystemAudioCapture(tmp_path, segment_duration=5)

        chunk1 = AudioChunk(path=tmp_path / "chunk_0000.wav", duration=4.2)
        chunk2 = AudioChunk(path=tmp_path / "chunk_0001.wav", duration=5.1)

        # Simular que la queue del thread tiene chunks
        capture._chunk_queue.put(chunk1)
        capture._chunk_queue.put(chunk2)
        capture._chunk_queue.put(None)

        async_queue: asyncio.Queue = asyncio.Queue()
        await capture.poll_chunks(async_queue)

        items = []
        while not async_queue.empty():
            items.append(await async_queue.get())

        assert len(items) == 3
        assert items[0] is chunk1
        assert items[1] is chunk2
        assert items[2] is None

    async def test_poll_chunks_sends_sentinel_when_empty(self, tmp_path):
        capture = SystemAudioCapture(tmp_path, segment_duration=5)
        capture._chunk_queue.put(None)

        async_queue: asyncio.Queue = asyncio.Queue()
        await capture.poll_chunks(async_queue)

        sentinel = await async_queue.get()
        assert sentinel is None

    async def test_stop_sets_event(self, tmp_path):
        capture = SystemAudioCapture(tmp_path, segment_duration=5)
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
