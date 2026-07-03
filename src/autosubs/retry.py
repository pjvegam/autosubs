import asyncio
import logging
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
BASE_DELAY = 1.0


async def with_retries[T](
    fn: Callable[[], Awaitable[T]],
    description: str,
    max_retries: int = MAX_RETRIES,
    base_delay: float = BASE_DELAY,
) -> T | None:
    """Ejecuta una funcion async con reintentos y backoff exponencial.

    Retorna el resultado o None si falla despues de todos los intentos.
    """
    for attempt in range(max_retries):
        try:
            return await fn()
        except Exception as e:
            delay = base_delay * (2 ** attempt)
            logger.warning(
                "%s (intento %d/%d): %s. Reintentando en %.1fs",
                description, attempt + 1, max_retries, e, delay,
            )
            if attempt < max_retries - 1:
                await asyncio.sleep(delay)

    logger.error("%s: fallo despues de %d intentos", description, max_retries)
    return None
