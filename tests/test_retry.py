

from autosubs.retry import with_retries


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
