import os
import sys
from dataclasses import dataclass, field

from dotenv import load_dotenv


@dataclass
class Config:
    url: str = ""
    source_lang: str = ""
    target_lang: str = ""
    segment_duration: int = 0
    whisper_model: str = ""
    gpt_model: str = ""
    system_audio: bool = False
    diarize: bool = False
    batch_vod: bool = False
    benchmark: bool = False
    vad_aggressiveness: int = 0
    cookies_browser: str = ""
    start_time: int = 0
    max_minutes: int = 0
    no_cache: bool = False
    speaker_gender: str = ""
    speakers: int = 0
    verbose: bool = False
    api_key: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        load_dotenv()
        if not self.cookies_browser:
            self.cookies_browser = os.environ.get("LIVESUBS_COOKIES_BROWSER", "firefox")
        if self.cookies_browser.lower() == "none":
            self.cookies_browser = ""
        if not self.source_lang:
            self.source_lang = os.environ.get("LIVESUBS_SOURCE_LANG", "ja")
        if not self.target_lang:
            self.target_lang = os.environ.get("LIVESUBS_TARGET_LANG", "es")
        if not self.segment_duration:
            self.segment_duration = int(
                os.environ.get("LIVESUBS_SEGMENT_DURATION", "5"),
            )
        if not self.whisper_model:
            self.whisper_model = os.environ.get(
                "LIVESUBS_WHISPER_MODEL", "gpt-4o-transcribe",
            )
        if not self.gpt_model:
            self.gpt_model = os.environ.get("LIVESUBS_GPT_MODEL", "gpt-4.1")
        if not self.diarize:
            self.diarize = os.environ.get("LIVESUBS_DIARIZE", "").lower() in ("1", "true")
        if not self.vad_aggressiveness:
            self.vad_aggressiveness = int(
                os.environ.get("LIVESUBS_VAD_AGGRESSIVENESS", "2"),
            )
        if not self.api_key:
            self.api_key = os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY no encontrada. "
                "Configura la variable de entorno o crea un archivo .env"
            )
        if self.benchmark and not self.start_time:
            self.start_time = 300
        if not self.system_audio and not self.url:
            raise ValueError(
                "Se requiere una URL de YouTube, "
                "o usa --system-audio para capturar audio del sistema"
            )

    @property
    def vad_min_duration(self) -> float:
        return self.segment_duration * 0.6

    @property
    def vad_max_duration(self) -> float:
        return self.segment_duration * 1.6

    @property
    def ipc_pipe(self) -> str:
        if sys.platform == "win32":
            return r"\\.\pipe\mpvsocket"
        return "/tmp/mpvsocket"  # noqa: S108
