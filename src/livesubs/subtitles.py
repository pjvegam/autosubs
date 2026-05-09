import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_TS_RE = re.compile(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})")


def _format_timestamp(seconds: float) -> str:
    """Convierte segundos a formato SRT HH:MM:SS,mmm."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _parse_timestamp(ts: str) -> float:
    """Parsea HH:MM:SS,mmm a segundos."""
    if m := _TS_RE.match(ts):
        return int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]) + int(m[4]) / 1000
    return 0.0


def srt_last_end_time(srt_path: Path) -> float:
    """Retorna el end timestamp de la ultima entrada SRT en segundos, o 0.0."""
    last_end = 0.0
    with open(srt_path, encoding="utf-8") as f:
        for line in f:
            if " --> " in line:
                end_str = line.split(" --> ")[1].strip()
                last_end = _parse_timestamp(end_str)
    return last_end


class SubtitleManager:
    """Gestiona un archivo SRT append-only."""

    def __init__(self, srt_path: Path, *, append: bool = False) -> None:
        self.srt_path = srt_path
        if append and srt_path.exists():
            with open(srt_path, encoding="utf-8") as f:
                count = sum(1 for line in f if " --> " in line)
            self._index = count + 1
        else:
            self._index = 1
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
