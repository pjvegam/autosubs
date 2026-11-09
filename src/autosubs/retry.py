import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
BASE_DELAY = 1.0

# Tope a la espera que pide el servidor. OpenRouter manda Retry-After: 120 al
# agotarse el presupuesto en vuelo, y respetarlo entero dejaria el chunk
# ocupando un hueco del semaforo dos minutos. Pero ignorarlo es peor: el
# backoff de 1s y 2s se agota en 3s, cuando el servidor dijo 120, y el chunk se
# pierde sin remedio (visto el 2026-11-08: chunk_0909 murio asi y su audio no
# llego nunca al SRT).
MAX_RETRY_AFTER = 30.0


def _requested_delay(error: Exception) -> float | None:
    """Cuanto pide esperar el servidor, si lo dice.

    Lo manda por dos vias distintas y hay que mirar las dos: la cabecera
    Retry-After de la respuesta HTTP, y el cuerpo de OpenRouter, que repite el
    dato en `error.metadata` (como `retry_after_seconds` en los 429 y como
    cabecera anidada en los 402).
    """
    cabeceras = getattr(getattr(error, "response", None), "headers", None)
    if cabeceras is not None:
        with contextlib.suppress(TypeError, ValueError):
            if (crudo := cabeceras.get("retry-after")) is not None:
                return float(crudo)

    cuerpo = getattr(error, "body", None)
    if not isinstance(cuerpo, dict):
        return None
    meta = cuerpo.get("error", {})
    meta = meta.get("metadata", {}) if isinstance(meta, dict) else {}
    if not isinstance(meta, dict):
        return None
    candidatos = [meta.get("retry_after_seconds")]
    if isinstance(anidadas := meta.get("headers"), dict):
        candidatos.append(anidadas.get("Retry-After"))
    for candidato in candidatos:
        if candidato is not None:
            with contextlib.suppress(TypeError, ValueError):
                return float(candidato)
    return None


async def with_retries[T](
    fn: Callable[[], Awaitable[T]],
    description: str,
    max_retries: int = MAX_RETRIES,
    base_delay: float = BASE_DELAY,
) -> T | None:
    """Ejecuta una funcion async con reintentos y backoff exponencial.

    Cuando el servidor dice cuanto esperar, manda el servidor y no el backoff:
    reintentar antes de tiempo gasta un intento garantizando otro rechazo.

    Retorna el resultado o None si falla despues de todos los intentos.
    """
    for attempt in range(max_retries):
        try:
            return await fn()
        except Exception as e:
            delay = base_delay * (2 ** attempt)
            if (pedida := _requested_delay(e)) is not None:
                delay = min(max(delay, pedida), MAX_RETRY_AFTER)
                if pedida > MAX_RETRY_AFTER:
                    logger.warning(
                        "%s: el servidor pide %.0fs de espera; esperando solo "
                        "%.0fs para no bloquear el pipeline, puede volver a "
                        "fallar.",
                        description, pedida, delay,
                    )
            logger.warning(
                "%s (intento %d/%d): %s. Reintentando en %.1fs",
                description, attempt + 1, max_retries, e, delay,
            )
            if attempt < max_retries - 1:
                await asyncio.sleep(delay)

    logger.error("%s: fallo despues de %d intentos", description, max_retries)
    return None
