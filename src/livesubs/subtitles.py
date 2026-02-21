import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _format_timestamp(seconds: float) -> str:
    """Convierte segundos a formato SRT HH:MM:SS,mmm."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


class SubtitleManager:
    """Gestiona un archivo SRT append-only."""

    def __init__(self, srt_path: Path) -> None:
        self.srt_path = srt_path
        self._index = 1
        # Crear archivo vacio
        self.srt_path.write_text("", encoding="utf-8")

    def add_entry(self, start: float, end: float, text: str) -> None:
        """Agrega una entrada al archivo SRT."""
        entry = (
            f"{self._index}\n"
            f"{_format_timestamp(start)} --> {_format_timestamp(end)}\n"
            f"{text}\n\n"
        )
        with open(self.srt_path, "a", encoding="utf-8") as f:
            f.write(entry)
        logger.debug(
            "SRT #%d [%s -> %s]: %s",
            self._index,
            _format_timestamp(start),
            _format_timestamp(end),
            text[:50],
        )
        self._index += 1
