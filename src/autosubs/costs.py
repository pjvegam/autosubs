from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Precios por modelo (USD). Fuente: https://openai.com/api/pricing/
# Whisper: $/minuto de audio
# GPT: $/millón de tokens (input, output)
PRICING: dict[str, dict[str, float]] = {
    # Whisper / transcripcion
    "whisper-1": {"per_minute": 0.006},
    "gpt-4o-transcribe": {"per_minute": 0.006},
    "faster-whisper-local": {"per_minute": 0.00},
    "local-turbo": {"per_minute": 0.00},
    "local-v3": {"per_minute": 0.00},
    "cohere-transcribe-03-2026": {"per_minute": 0.00},
    "gpt-4o-mini-transcribe": {"per_minute": 0.003},
    # Via OpenRouter. Precios derivados del campo usage.cost de la propia API
    # sobre un clip de 3s (2026-07-25).
    "microsoft/mai-transcribe-1.5": {"per_minute": 0.006},
    "microsoft/mai-transcribe-2": {"per_minute": 0.0018},
    "elevenlabs/scribe-v2": {"per_minute": 0.0018},
    "qwen/qwen3-asr-flash-2026-02-10": {"per_minute": 0.0019},
    "qwen/qwen3-asr-1.7b": {"per_minute": 0.0005},
    "qwen/qwen3-asr-0.6b": {"per_minute": 0.0002},
    "x-ai/grok-stt-1.0": {"per_minute": 0.0017},
    "google/gemini-3.5-transcribe": {"per_minute": 0.003},
    "assemblyai/universal-3-5-pro": {"per_minute": 0.0075},
    "openai/gpt-transcribe": {"per_minute": 0.0048},
    "nvidia/nemotron-3.5-asr-streaming-multilingual-0.6b": {"per_minute": 0.0002},
    "openai/whisper-large-v3-turbo": {"per_minute": 0.0002},
    "deepgram/nova-3": {"per_minute": 0.00364},
    "mistralai/voxtral-mini-transcribe": {"per_minute": 0.002},
    "openai/whisper-1": {"per_minute": 0.006},
    "gpt-4o-transcribe-diarize": {"per_minute": 0.006},
    # Gemini audio: $1/M tokens, ~1500 tokens/min => $0.0015/min
    "gemini-2.5-flash": {"per_minute": 0.0015},
    "gemini-2.5-pro": {"per_minute": 0.0015},
    # GPT / traduccion
    "gpt-4o": {"input_mtok": 2.50, "output_mtok": 10.00},
    "gpt-4o-mini": {"input_mtok": 0.15, "output_mtok": 0.60},
    "gpt-4.1": {"input_mtok": 2.00, "output_mtok": 8.00},
    "gpt-4.1-mini": {"input_mtok": 0.40, "output_mtok": 1.60},
    "gpt-4.1-nano": {"input_mtok": 0.10, "output_mtok": 0.40},
    # OpenRouter (passthrough sin markup)
    "deepseek/deepseek-v4-flash": {"input_mtok": 0.14, "output_mtok": 0.28},
    "deepseek/deepseek-v4-pro": {"input_mtok": 1.74, "output_mtok": 3.48},
    "deepseek/deepseek-chat": {"input_mtok": 0.32, "output_mtok": 0.89},
    "moonshotai/kimi-k2": {"input_mtok": 0.57, "output_mtok": 2.30},
    "anthropic/claude-haiku-5.5": {"input_mtok": 0.10, "output_mtok": 0.50},
    "anthropic/claude-haiku-4.5": {"input_mtok": 1.00, "output_mtok": 5.00},
    "google/gemini-2.5-flash": {"input_mtok": 0.30, "output_mtok": 2.50},
    "google/gemini-2.5-pro": {"input_mtok": 1.25, "output_mtok": 10.00},
}


def escape_md_table(text: str) -> str:
    """Escapa texto para celdas de tabla Markdown."""
    return text.replace("|", "\\|").replace("\n", " ")


@dataclass
class SessionEntry:
    start_time: float
    duration: float
    transcription: str
    translation: str


@dataclass
class CostTracker:
    whisper_seconds: float = 0.0
    whisper_model: str = ""
    translation_prompt_tokens: int = 0
    translation_completion_tokens: int = 0
    translation_model: str = ""
    source_lang: str = ""
    target_lang: str = ""
    entries: list[SessionEntry] = field(default_factory=list)

    def add_whisper(self, duration_seconds: float) -> None:
        self.whisper_seconds += duration_seconds

    def add_translation(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.translation_prompt_tokens += prompt_tokens
        self.translation_completion_tokens += completion_tokens

    def add_entry(
        self,
        start_time: float,
        duration: float,
        transcription: str,
        translation: str,
    ) -> None:
        self.entries.append(
            SessionEntry(start_time, duration, transcription, translation),
        )

    def whisper_cost(self) -> float:
        prices = PRICING.get(self.whisper_model, {})
        per_minute = prices.get("per_minute", 0.0)
        return (self.whisper_seconds / 60) * per_minute

    def translation_cost(self) -> tuple[float, float]:
        prices = PRICING.get(self.translation_model, {})
        input_cost = (
            (self.translation_prompt_tokens / 1_000_000)
            * prices.get("input_mtok", 0.0)
        )
        output_cost = (
            (self.translation_completion_tokens / 1_000_000)
            * prices.get("output_mtok", 0.0)
        )
        return input_cost, output_cost

    def summary(self) -> str:
        whisper_min = self.whisper_seconds / 60
        whisper_cost = self.whisper_cost()
        input_cost, output_cost = self.translation_cost()
        translation_cost = input_cost + output_cost
        total = whisper_cost + translation_cost

        lines = [
            "--- Resumen de costos ---",
            f"Whisper ({self.whisper_model}): "
            f"{whisper_min:.1f} min -> ${whisper_cost:.4f}",
            f"Traduccion ({self.translation_model}): "
            f"{self.translation_prompt_tokens:,} in + "
            f"{self.translation_completion_tokens:,} out tokens "
            f"-> ${translation_cost:.4f}",
            f"Total: ${total:.4f} USD",
        ]
        return "\n".join(lines)

    def save_srt(self, logs_dir: Path, *, video_id: str = "") -> Path | None:
        """Genera archivo SRT a partir de las entries. Retorna el path o None.

        Si se pasa video_id, usa nombre estable ({video_id}.srt) para cache.
        """
        if not self.entries:
            return None

        from autosubs.subtitles import _format_timestamp

        logs_dir.mkdir(parents=True, exist_ok=True)
        if video_id:
            filename = f"{video_id}.srt"
        else:
            now = datetime.now()
            filename = f"session_{now:%Y-%m-%d_%H-%M-%S}.srt"
        path = logs_dir / filename

        lines: list[str] = []
        for i, entry in enumerate(self.entries, 1):
            start = _format_timestamp(entry.start_time)
            end = _format_timestamp(entry.start_time + entry.duration)
            lines.append(f"{i}\n{start} --> {end}\n{entry.translation}\n")

        path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("SRT guardado en: %s", path)
        return path

    def save_markdown(self, logs_dir: Path) -> Path | None:
        """Guarda log de sesion como Markdown. Retorna el path o None."""
        if not self.entries:
            return None

        logs_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now()
        filename = f"session_{now:%Y-%m-%d_%H-%M-%S}.md"
        path = logs_dir / filename

        lines = [
            f"# AutoSubs - {now:%Y-%m-%d %H:%M}",
            "",
            f"**Whisper:** {self.whisper_model} | "
            f"**GPT:** {self.translation_model} | "
            f"**{self.source_lang} -> {self.target_lang}**",
            "",
            "| Tiempo | Whisper | Traduccion |",
            "|--------|---------|------------|",
        ]

        for entry in self.entries:
            mm, ss = divmod(int(entry.start_time), 60)
            tr = escape_md_table(entry.transcription)
            tl = escape_md_table(entry.translation)
            lines.append(f"| {mm:02d}:{ss:02d} | {tr} | {tl} |")

        lines.extend([
            "",
            "## Costos",
            "```",
            self.summary(),
            "```",
            "",
        ])

        path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Log de sesion guardado en: %s", path)
        return path
