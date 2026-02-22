from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from openai import AsyncOpenAI

from livesubs.retry import with_retries

if TYPE_CHECKING:
    from livesubs.costs import CostTracker

logger = logging.getLogger(__name__)

_ROLE = "Eres un traductor de subtitulos en tiempo real. "

_RULES = (
    "Reglas estrictas:\n"
    "- Responde UNICAMENTE con la traduccion, nada mas.\n"
    "- NUNCA incluyas el texto original en tu respuesta.\n"
    "- NUNCA agregues explicaciones, notas o contexto.\n"
    "- Traduccion concisa y natural, estilo subtitulo.\n"
    "- Si no puedes traducir, responde con una traduccion aproximada."
)

SYSTEM_PROMPT = _ROLE + _RULES


def _build_system_prompt(context: str) -> str:
    """Construye el system prompt, insertando contexto si existe."""
    if not context:
        return SYSTEM_PROMPT
    return f"{_ROLE}\n{context}\n{_RULES}"


async def translate_text(
    client: AsyncOpenAI,
    text: str,
    source_lang: str,
    target_lang: str,
    model: str = "gpt-4.1",
    context: str = "",
    tracker: CostTracker | None = None,
) -> str | None:
    """Traduce texto usando la API de GPT.

    Retorna el texto traducido o None si falla despues de reintentos.
    """
    system_prompt = _build_system_prompt(context)
    user_msg = f"Traduce del {source_lang} al {target_lang}:\n{text}"

    async def _call() -> str:
        response = await client.chat.completions.create(
            model=model,
            temperature=0.3,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
        )
        if tracker and response.usage:
            tracker.add_translation(
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
            )
        return response.choices[0].message.content.strip()

    translated = await with_retries(_call, "Traduccion")
    if translated:
        logger.debug("Traducido: %s -> %s", text[:40], translated[:40])
        return translated
    return None
