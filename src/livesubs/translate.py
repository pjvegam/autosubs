import logging

from openai import AsyncOpenAI

from livesubs.retry import with_retries

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Eres un traductor de subtitulos en tiempo real. "
    "Reglas estrictas:\n"
    "- Responde UNICAMENTE con la traduccion, nada mas.\n"
    "- NUNCA incluyas el texto original en tu respuesta.\n"
    "- NUNCA agregues explicaciones, notas o contexto.\n"
    "- Traduccion concisa y natural, estilo subtitulo.\n"
    "- Si no puedes traducir, responde con una traduccion aproximada."
)


async def translate_text(
    client: AsyncOpenAI,
    text: str,
    source_lang: str,
    target_lang: str,
    model: str = "gpt-4o-mini",
) -> str | None:
    """Traduce texto usando la API de GPT.

    Retorna el texto traducido o None si falla despues de reintentos.
    """
    user_msg = f"Traduce del {source_lang} al {target_lang}:\n{text}"

    async def _call() -> str:
        response = await client.chat.completions.create(
            model=model,
            temperature=0.3,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
        return response.choices[0].message.content.strip()

    translated = await with_retries(_call, "Traduccion")
    if translated:
        logger.debug("Traducido: %s -> %s", text[:40], translated[:40])
        return translated
    return None
