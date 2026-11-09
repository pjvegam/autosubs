from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from openai import AsyncOpenAI

from autosubs.retry import with_retries

if TYPE_CHECKING:
    from autosubs.costs import CostTracker

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

# El modelo a veces ignora las reglas y contesta sobre la traduccion en vez de
# traducir ("I cannot translate this text as it appears to be Georgian..."),
# tipicamente cuando el ASR alucino en otro idioma. Eso no es un subtitulo.
_META_RESPONSE = re.compile(
    r"^(i (cannot|can't|am unable to|was unable to)|"
    r"(this|the) (text|audio|input) (appears|seems|is)|"
    r"(no puedo|no es posible) (traducir|interpretar)|"
    r"(sorry|unfortunately|lo siento),)",
    re.IGNORECASE,
)


# Para poder darle a cada turno su propio tiempo en pantalla, la traduccion
# tiene que venir con una linea por turno. Medido sobre 18 chunks de dos
# personas: sin esta regla cuadran 16, con ella 18. Pero OJO, en un chunk de un
# solo turno la regla lo hizo partir en 5 lineas, asi que solo se agrega cuando
# de verdad hay varios turnos.
_LINE_RULE = (
    "- The input has one line per speaker turn. Reply with EXACTLY the same "
    "number of lines, in the same order, one translation per line. "
    "Never merge or split lines.\n"
)


def _build_system_prompt(context: str, line_count: int = 0) -> str:
    """Construye el system prompt, insertando contexto si existe."""
    rules = _RULES if line_count < 2 else _RULES.replace(
        "Strict rules:\n", "Strict rules:\n" + _LINE_RULE,
    )
    if not context:
        return _ROLE + rules
    return f"{_ROLE}\n{context}\n{rules}"


async def translate_text(
    client: AsyncOpenAI,
    text: str,
    source_lang: str,
    target_lang: str,
    model: str = "gpt-4.1",
    context: str = "",
    tracker: CostTracker | None = None,
    line_count: int = 0,
) -> str | None:
    """Traduce texto usando la API de GPT.

    Retorna el texto traducido o None si falla despues de reintentos.

    Con line_count >= 2 le pide al modelo que devuelva una linea por turno de
    habla. El que llama tiene que verificar el conteo igual: el modelo a veces
    no obedece, y ahi conviene caer al bloque unico.
    """
    system_prompt = _build_system_prompt(context, line_count)
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
    if not translated:
        return None
    if _META_RESPONSE.match(translated):
        logger.warning("Descartado meta-respuesta del traductor: %s", translated[:60])
        return None
    logger.debug("Traducido: %s -> %s", text[:40], translated[:40])
    return translated


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
