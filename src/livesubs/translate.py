from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from openai import AsyncOpenAI

from livesubs.retry import with_retries

if TYPE_CHECKING:
    from livesubs.costs import CostTracker

logger = logging.getLogger(__name__)

_ROLE = "You are a real-time subtitle translator. "

_RULES = (
    "Strict rules:\n"
    "- Reply ONLY with the translation, nothing else.\n"
    "- NEVER include the original text in your response.\n"
    "- NEVER add explanations, notes, or context.\n"
    "- Concise and natural translation, subtitle style.\n"
    "- Keep consistent grammatical gender throughout the session.\n"
    "- Keep proper names (people, characters, places) consistent across lines.\n"
    "- If you cannot translate, reply with an approximate translation."
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
    user_msg = f"Translate from {source_lang} to {target_lang}:\n{text}"

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
        content = response.choices[0].message.content
        return content.strip() if content else ""

    translated = await with_retries(_call, "Traduccion")
    if translated:
        logger.debug("Traducido: %s -> %s", text[:40], translated[:40])
        return translated
    return None


async def summarize_session(
    client: AsyncOpenAI,
    recent_translations: list[str],
    source_lang: str,
    model: str = "gpt-4.1-mini",
) -> str:
    """Pide a GPT un resumen breve del tema actual para dar contexto."""
    joined = "\n".join(recent_translations)
    prompt = (
        f"Given these recent translated subtitles (original language: {source_lang}), "
        "summarize in 1-2 sentences what is being discussed. Topic only, no formatting."
    )

    async def _call() -> str:
        response = await client.chat.completions.create(
            model=model,
            temperature=0.3,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": joined},
            ],
        )
        content = response.choices[0].message.content
        return content.strip() if content else ""

    result = await with_retries(_call, "Resumen de sesion")
    return result or ""
