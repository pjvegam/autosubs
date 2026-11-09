
from unittest.mock import patch

from autosubs.retry import MAX_RETRY_AFTER, _requested_delay, with_retries


class TestWithRetries:
    async def test_succeeds_first_try(self):
        async def fn():
            return "ok"

        result = await with_retries(fn, "test")
        assert result == "ok"

    async def test_succeeds_after_retries(self):
        attempts = 0

        async def fn():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise RuntimeError("fail")
            return "ok"

        result = await with_retries(fn, "test", base_delay=0.01)
        assert result == "ok"
        assert attempts == 3

    async def test_returns_none_after_all_retries_fail(self):
        async def fn():
            raise RuntimeError("always fails")

        result = await with_retries(fn, "test", max_retries=2, base_delay=0.01)
        assert result is None

    async def test_respects_max_retries(self):
        attempts = 0

        async def fn():
            nonlocal attempts
            attempts += 1
            raise RuntimeError("fail")

        await with_retries(fn, "test", max_retries=4, base_delay=0.01)
        assert attempts == 4

    async def test_returns_falsy_values(self):
        async def fn():
            return 0

        result = await with_retries(fn, "test")
        assert result == 0

    async def test_returns_empty_string(self):
        async def fn():
            return ""

        result = await with_retries(fn, "test")
        assert result == ""


# El 402 tal cual lo devolvio OpenRouter el 2026-11-08, con $2.30 de saldo y 6
# transcripciones en vuelo. El dato de la espera viene anidado dos niveles.
CUERPO_402 = {
    "error": {
        "message": "This request would exceed your available credits given "
                   "your current in-flight requests.",
        "code": 402,
        "metadata": {
            "reason": "in_flight_budget_exhausted",
            "headers": {"Retry-After": "120"},
        },
    },
}

CUERPO_429 = {
    "error": {
        "message": "Provider returned 429",
        "code": 429,
        "metadata": {"retry_after_seconds": 5, "headers": {"Retry-After": "5"}},
    },
}


class _CuerpoError(Exception):
    def __init__(self, cuerpo: dict) -> None:
        super().__init__(str(cuerpo))
        self.body = cuerpo


class _CabeceraError(Exception):
    class _Resp:
        def __init__(self, headers: dict) -> None:
            self.headers = headers

    def __init__(self, headers: dict) -> None:
        super().__init__("throttled")
        self.response = self._Resp(headers)


class TestEsperaQuePideElServidor:
    """Reintentar antes de lo que pide el servidor garantiza otro rechazo.

    Con el backoff de 1s y 2s, los 3 intentos se agotan en 3s contra un
    Retry-After de 120. Asi se perdio chunk_0909 el 2026-11-08: su audio no
    llego nunca al SRT.
    """

    def test_lee_el_402_anidado_de_openrouter(self):
        assert _requested_delay(_CuerpoError(CUERPO_402)) == 120.0

    def test_lee_el_429_con_retry_after_seconds(self):
        assert _requested_delay(_CuerpoError(CUERPO_429)) == 5.0

    def test_lee_la_cabecera_http(self):
        assert _requested_delay(_CabeceraError({"retry-after": "7"})) == 7.0

    def test_un_error_cualquiera_no_pide_nada(self):
        assert _requested_delay(RuntimeError("boom")) is None

    def test_basura_en_el_cuerpo_no_revienta(self):
        for cuerpo in [
            {"error": "no es un dict"},
            {"error": {"metadata": {"retry_after_seconds": "ya"}}},
            {"error": {"metadata": None}},
            {},
        ]:
            assert _requested_delay(_CuerpoError(cuerpo)) is None

    async def test_espera_lo_que_pide_y_no_el_backoff(self):
        esperas: list[float] = []

        async def fn():
            raise _CuerpoError(CUERPO_429)

        async def _sleep(d):
            esperas.append(d)

        with patch("autosubs.retry.asyncio.sleep", _sleep):
            await with_retries(fn, "test", max_retries=2, base_delay=1.0)
        assert esperas == [5.0], f"espero {esperas} en vez de los 5s pedidos"

    async def test_no_espera_mas_que_el_tope(self):
        esperas: list[float] = []

        async def fn():
            raise _CuerpoError(CUERPO_402)

        async def _sleep(d):
            esperas.append(d)

        with patch("autosubs.retry.asyncio.sleep", _sleep):
            await with_retries(fn, "test", max_retries=3, base_delay=1.0)
        assert esperas == [MAX_RETRY_AFTER, MAX_RETRY_AFTER]
        assert MAX_RETRY_AFTER < 120.0

    async def test_el_backoff_gana_si_pide_menos(self):
        esperas: list[float] = []

        async def fn():
            raise _CabeceraError({"retry-after": "0"})

        async def _sleep(d):
            esperas.append(d)

        with patch("autosubs.retry.asyncio.sleep", _sleep):
            await with_retries(fn, "test", max_retries=3, base_delay=1.0)
        assert esperas == [1.0, 2.0], "no puede reintentar mas rapido que el backoff"
