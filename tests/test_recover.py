import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

from autosubs.config import Config
from autosubs.recover import (
    _GAP_CONCURRENCY,
    SrtEntry,
    _cargar_vacios,
    _guardar_vacios,
    fill_gaps,
    find_gaps,
    parse_srt,
    write_srt,
)

URL = "https://www.youtube.com/watch?v=abcdefghijk"
VIDEO_ID = "abcdefghijk"


def _srt_con_huecos(path: Path) -> None:
    """SRT con tres huecos de 10s entre entries de 1s."""
    write_srt(path, [
        SrtEntry(0.0, 1.0, "uno"),
        SrtEntry(11.0, 12.0, "dos"),
        SrtEntry(22.0, 23.0, "tres"),
        SrtEntry(33.0, 34.0, "cuatro"),
    ])


class TestVaciosPersistidos:
    def test_ida_y_vuelta(self, tmp_path):
        p = tmp_path / "x.gaps_vacios.json"
        _guardar_vacios(p, {(1.0, 2.0), (3.5, 4.5)})
        assert _cargar_vacios(p) == {(1.0, 2.0), (3.5, 4.5)}

    def test_sin_archivo_no_hay_nada(self, tmp_path):
        assert _cargar_vacios(tmp_path / "no_existe.json") == set()

    def test_archivo_corrupto_no_revienta(self, tmp_path):
        p = tmp_path / "roto.json"
        p.write_text("{no es json", encoding="utf-8")
        assert _cargar_vacios(p) == set()


class TestFillGapsReanuda:
    """Interrumpir y relanzar no puede costar el original ni pagar dos veces.

    La primera corrida real la interrumpio el usuario en el gap 171 de 211.
    Ahi se vio que relanzar pisaba el .bak con un SRT ya modificado, y que los
    gaps vacios (102 de 170) se volvian a transcribir enteros.
    """

    @staticmethod
    def _config(tmp_path: Path) -> Config:
        return Config(url=URL, api_key="dummy")

    @staticmethod
    def _preparar(tmp_path: Path):
        srt = tmp_path / f"{VIDEO_ID}.srt"
        _srt_con_huecos(srt)
        (tmp_path / f"{VIDEO_ID}.mkv").write_bytes(b"fake")
        return srt

    async def test_no_pisa_un_backup_que_ya_existe(self, tmp_path):
        srt = self._preparar(tmp_path)
        backup = srt.with_suffix(".srt.bak")
        backup.write_text("EL ORIGINAL", encoding="utf-8")

        with (
            patch("autosubs.recover.LOGS_DIR", tmp_path),
            patch("autosubs.recover._fill_gap", new_callable=AsyncMock, return_value=[]),
            patch("autosubs.pipeline._create_asr_client"),
            patch("autosubs.pipeline._create_translate_client"),
        ):
            await fill_gaps(self._config(tmp_path))

        assert backup.read_text(encoding="utf-8") == "EL ORIGINAL"

    async def test_crea_el_backup_la_primera_vez(self, tmp_path):
        srt = self._preparar(tmp_path)
        original = srt.read_text(encoding="utf-8")

        with (
            patch("autosubs.recover.LOGS_DIR", tmp_path),
            patch("autosubs.recover._fill_gap", new_callable=AsyncMock, return_value=[]),
            patch("autosubs.pipeline._create_asr_client"),
            patch("autosubs.pipeline._create_translate_client"),
        ):
            await fill_gaps(self._config(tmp_path))

        assert srt.with_suffix(".srt.bak").read_text(encoding="utf-8") == original

    async def test_los_gaps_vacios_no_se_reintentan(self, tmp_path):
        srt = self._preparar(tmp_path)
        cuantos = len(find_gaps(parse_srt(srt)))
        assert cuantos == 3

        vacio = AsyncMock(return_value=[])
        with (
            patch("autosubs.recover.LOGS_DIR", tmp_path),
            patch("autosubs.recover._fill_gap", vacio),
            patch("autosubs.pipeline._create_asr_client"),
            patch("autosubs.pipeline._create_translate_client"),
        ):
            await fill_gaps(self._config(tmp_path))
            primera = vacio.await_count
            await fill_gaps(self._config(tmp_path))
            segunda = vacio.await_count - primera

        assert primera == cuantos
        assert segunda == 0, f"reintento {segunda} gaps que ya salieron vacios"

    async def test_un_gap_que_dio_texto_no_queda_marcado_vacio(self, tmp_path):
        srt = self._preparar(tmp_path)

        async def _lleno(_c, _t, _cfg, _v, gs, ge, _d, _tr):
            return [SrtEntry(gs + 1, gs + 2, "recuperado")]

        with (
            patch("autosubs.recover.LOGS_DIR", tmp_path),
            patch("autosubs.recover._fill_gap", _lleno),
            patch("autosubs.pipeline._create_asr_client"),
            patch("autosubs.pipeline._create_translate_client"),
        ):
            await fill_gaps(self._config(tmp_path))

        vacios = json.loads((tmp_path / f"{VIDEO_ID}.gaps_vacios.json").read_text())
        assert vacios == []
        assert "recuperado" in srt.read_text(encoding="utf-8")

    async def test_el_srt_queda_ordenado_pese_a_la_concurrencia(self, tmp_path):
        srt = self._preparar(tmp_path)

        async def _lleno(_c, _t, _cfg, _v, gs, ge, _d, _tr):
            return [SrtEntry(gs + 1, gs + 2, f"en {gs:.0f}")]

        with (
            patch("autosubs.recover.LOGS_DIR", tmp_path),
            patch("autosubs.recover._fill_gap", _lleno),
            patch("autosubs.pipeline._create_asr_client"),
            patch("autosubs.pipeline._create_translate_client"),
        ):
            await fill_gaps(self._config(tmp_path))

        entradas = parse_srt(srt)
        assert entradas == sorted(entradas, key=lambda e: e.start)

    async def test_varios_gaps_a_la_vez(self, tmp_path):
        """De uno en uno, 211 gaps son media hora. Tienen que solaparse."""
        self._preparar(tmp_path)
        a_la_vez = 0
        pico = 0

        async def _lento(_c, _t, _cfg, _v, gs, ge, _d, _tr):
            nonlocal a_la_vez, pico
            a_la_vez += 1
            pico = max(pico, a_la_vez)
            try:
                import asyncio
                await asyncio.sleep(0.01)
            finally:
                a_la_vez -= 1
            return []

        with (
            patch("autosubs.recover.LOGS_DIR", tmp_path),
            patch("autosubs.recover._fill_gap", _lento),
            patch("autosubs.pipeline._create_asr_client"),
            patch("autosubs.pipeline._create_translate_client"),
        ):
            await fill_gaps(self._config(tmp_path))

        assert pico > 1, "los gaps se procesaron de uno en uno"
        assert pico <= _GAP_CONCURRENCY
