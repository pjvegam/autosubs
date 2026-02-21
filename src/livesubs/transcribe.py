import logging
from pathlib import Path

from openai import AsyncOpenAI

from livesubs.retry import with_retries

logger = logging.getLogger(__name__)


async def transcribe_audio(
    client: AsyncOpenAI,
    audio_path: Path,
    language: str,
    model: str = "gpt-4o-transcribe",
) -> str | None:
    """Transcribe un archivo de audio usando la API de Whisper.

    Retorna el texto transcrito o None si falla despues de reintentos.
    """
    async def _call() -> str:
        with open(audio_path, "rb") as f:
            response = await client.audio.transcriptions.create(
                model=model,
                file=f,
                language=language,
                response_format="text",
            )
        return response.strip()

    text = await with_retries(_call, f"Transcripcion de {audio_path.name}")
    if text:
        logger.debug("Transcrito [%s]: %s", audio_path.name, text[:80])
        return text
    return None
