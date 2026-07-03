import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

from openai import AsyncOpenAI

from autosubs.retry import with_retries

logger = logging.getLogger(__name__)

# Patron para detectar texto que es solo puntuacion/espacios/simbolos
_EMPTY_PATTERN = re.compile(
    "^[\\s.,!?;:\\-\u2014\u2026\u3002\u3001\uff01\uff1f]+$",
)

# Longitud minima para considerar una transcripcion valida
_MIN_TEXT_LENGTH = 2

# Un mismo caracter repetido 10+ veces seguidas (ej: えーーーーーーーーーー)
_REPETITIVE_PATTERN = re.compile(r"(.)\1{9,}")


def _is_hallucination(text: str) -> bool:
    """Detecta transcripciones que probablemente son alucinaciones de Whisper.

    Filtra: texto vacio, solo puntuacion, demasiado corto, o caracteres repetitivos.
    """
    if len(text) < _MIN_TEXT_LENGTH:
        return True
    if _EMPTY_PATTERN.fullmatch(text):
        return True
    return bool(_REPETITIVE_PATTERN.search(text))


async def transcribe_audio(
    client: AsyncOpenAI,
    audio_path: Path,
    language: str,
    model: str = "gpt-4o-transcribe",
) -> str | None:
    """Transcribe un archivo de audio usando la API de Whisper.

    Retorna el texto transcrito o None si falla despues de reintentos.
    Filtra transcripciones que parecen alucinaciones.
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
        if _is_hallucination(text):
            logger.debug("Filtrado (hallucination): [%s] '%s'", audio_path.name, text)
            return None
        logger.debug("Transcrito [%s]: %s", audio_path.name, text[:80])
        return text
    return None


async def transcribe_gemini_chunk(
    audio_path: Path,
    language: str,
    model: str = "gemini-2.5-pro",
) -> str | None:
    """Transcribe un chunk WAV con Gemini multimodal (audio inline).

    Usa Files API NO — para chunks pequeños (5-8s, ~80-160KB) inline es mas rapido.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY no configurada")
        return None

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    audio_bytes = audio_path.read_bytes()

    async def _call() -> str | None:
        response = await client.aio.models.generate_content(
            model=model,
            contents=[
                f"Transcribe este audio en {language}. Devuelve SOLO el texto transcrito, "
                "sin explicaciones, sin markdown, sin comillas.",
                types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
            ],
        )
        return response.text

    try:
        text = await with_retries(_call, f"Gemini transcribe {audio_path.name}")
    except Exception as e:
        logger.error("Error transcribiendo con Gemini: %s", e)
        return None

    if not text:
        return None
    text = text.strip().strip('"').strip()
    if _is_hallucination(text):
        logger.debug("Filtrado Gemini (hallucination): [%s] '%s'", audio_path.name, text)
        return None
    logger.debug("Gemini transcrito [%s]: %s", audio_path.name, text[:80])
    return text


@dataclass
class TranscriptSegment:
    """Segmento de transcripcion con timestamps."""

    text: str
    start: float
    end: float


async def transcribe_verbose(
    client: AsyncOpenAI,
    audio_path: Path,
    language: str,
) -> list[TranscriptSegment] | None:
    """Transcribe audio con verbose_json para obtener segments con timestamps.

    Usa whisper-1 (unico modelo que soporta verbose_json con timestamps).
    Mismo precio que gpt-4o-transcribe ($0.006/min).
    Whisper segmenta por frases naturales (mucho mejor que VAD de 5s).
    """
    async def _call() -> list:
        with open(audio_path, "rb") as f:
            response = await client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language=language,
                response_format="verbose_json",
            )
        return response.segments

    raw_segments = await with_retries(
        _call, f"Verbose transcription de {audio_path.name}",
    )
    if not raw_segments:
        return None

    segments: list[TranscriptSegment] = []
    for seg in raw_segments:
        if isinstance(seg, dict):
            text = seg.get("text", "").strip()
            start = seg.get("start", 0.0)
            end = seg.get("end", 0.0)
        else:
            text = getattr(seg, "text", "").strip()
            start = getattr(seg, "start", 0.0)
            end = getattr(seg, "end", 0.0)
        if _is_hallucination(text):
            continue
        segments.append(TranscriptSegment(text=text, start=start, end=end))

    logger.info(
        "Transcrito [%s]: %d segments",
        audio_path.name, len(segments),
    )
    return segments if segments else None
