"""El carril que recupera los trozos caidos sin que haya que pedirlo.

Un chunk que falla (402 de OpenRouter, 429, timeout) se perdia para siempre:
su .wav se borraba en el `finally` y nadie anotaba a que momento del video
correspondia. La unica salida era `--fill-gaps` a mano, sacando otra vez el
audio de un .mkv de 12 GB. Medido sobre una sesion real: 211 huecos y ~13.8
min de habla perdida en 74 min de stream.
"""
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from autosubs.config import Config
from autosubs.pipeline import (
    _MAX_PENDIENTES,
    _recovery_worker,
    _transcribe_worker_parallel,
)
from autosubs.subtitles import SubtitleManager, parse_entries


def _config() -> Config:
    return Config(url="https://youtu.be/x", api_key="dummy")


def _chunk(tmp_path: Path, nombre: str, duracion: float = 8.0):
    p = tmp_path / nombre
    p.write_bytes(b"RIFFfake")
    c = MagicMock()
    c.path = p
    c.duration = duracion
    return c


async def _correr_transcriptor(chunks, *, retry_queue, lado_efecto):
    audio_queue: asyncio.Queue = asyncio.Queue()
    text_queue: asyncio.Queue = asyncio.Queue()
    for c in chunks:
        await audio_queue.put(c)
    await audio_queue.put(None)

    with (
        patch("autosubs.pipeline.supports_turns", return_value=False),
        patch("autosubs.pipeline.transcribe_audio", side_effect=lado_efecto),
        patch("autosubs.pipeline._should_skip_chunk", return_value=False),
    ):
        await _transcribe_worker_parallel(
            MagicMock(), _config(), audio_queue, text_queue,
            retry_queue=retry_queue,
        )
    return text_queue


class TestElAudioDelTrozoCaido:
    async def test_un_fallo_conserva_su_wav_y_su_momento(self, tmp_path):
        bueno = _chunk(tmp_path, "ok.wav")
        malo = _chunk(tmp_path, "malo.wav")
        retry: asyncio.Queue = asyncio.Queue()

        async def _asr(_cli, path, *_a, **_k):
            if path.name == "malo.wav":
                raise RuntimeError("Error code: 402")
            return "hola"

        await _correr_transcriptor([bueno, malo], retry_queue=retry, lado_efecto=_asr)

        assert not bueno.path.exists(), "el que salio bien tiene que limpiarse"
        assert malo.path.exists(), "sin el .wav no hay nada que recuperar"

        item = retry.get_nowait()
        assert item is not None
        start, duration, path = item
        assert path == malo.path
        assert start == 8.0, f"el trozo caido iba en 8.0s, no en {start}"
        assert duration == 8.0

    async def test_sin_carril_se_borra_como_antes(self, tmp_path):
        """En vivo no hay recuperacion: no puede dejar .wav tirados."""
        malo = _chunk(tmp_path, "malo.wav")

        async def _asr(*_a, **_k):
            raise RuntimeError("boom")

        await _correr_transcriptor([malo], retry_queue=None, lado_efecto=_asr)
        assert not malo.path.exists()

    async def test_un_chunk_sin_voz_no_va_a_recuperacion(self, tmp_path):
        """Devolver vacio no es fallar: ahi de verdad no habia nada."""
        mudo = _chunk(tmp_path, "mudo.wav")
        retry: asyncio.Queue = asyncio.Queue()

        async def _asr(*_a, **_k):
            return ""

        await _correr_transcriptor([mudo], retry_queue=retry, lado_efecto=_asr)
        assert retry.get_nowait() is None, "solo el centinela de cierre"
        assert not mudo.path.exists()

    async def test_cierra_el_carril_al_terminar(self, tmp_path):
        """Sin el centinela, el TaskGroup no cierra y cuelga el fin del VOD."""
        retry: asyncio.Queue = asyncio.Queue()

        async def _asr(*_a, **_k):
            return "hola"

        await _correr_transcriptor(
            [_chunk(tmp_path, "a.wav")], retry_queue=retry, lado_efecto=_asr,
        )
        assert retry.get_nowait() is None


class TestCarrilDeRecuperacion:
    @staticmethod
    def _manager(tmp_path: Path) -> SubtitleManager:
        srt = tmp_path / "subs.srt"
        m = SubtitleManager(srt)
        m.add_entry(0.0, 2.0, "primero")
        m.add_entry(16.0, 18.0, "tercero")
        return m

    @staticmethod
    async def _correr(retry, manager, player, recuperar):
        with (
            patch("autosubs.pipeline._RETRY_PAUSA", 0.0),
            patch("autosubs.pipeline._recuperar_trozo", recuperar),
        ):
            await asyncio.wait_for(
                _recovery_worker(
                    MagicMock(), MagicMock(), _config(), retry, manager, player,
                ),
                timeout=5,
            )

    async def test_inserta_en_su_sitio_y_recarga_mpv(self, tmp_path):
        manager = self._manager(tmp_path)
        player = AsyncMock()
        wav = tmp_path / "caido.wav"
        wav.write_bytes(b"x")

        retry: asyncio.Queue = asyncio.Queue()
        await retry.put((8.0, 8.0, wav))
        await retry.put(None)

        async def _rec(*_a, **_k):
            return [(8.0, 10.0, "segundo")]

        await self._correr(retry, manager, player, _rec)

        textos = [t for _, _, t in parse_entries(manager.srt_path)]
        assert textos == ["primero", "segundo", "tercero"]
        assert player.sub_reload.await_count == 1
        assert not wav.exists(), "el .wav recuperado tiene que limpiarse"

    async def test_renumera_el_srt(self, tmp_path):
        manager = self._manager(tmp_path)
        wav = tmp_path / "c.wav"
        wav.write_bytes(b"x")
        retry: asyncio.Queue = asyncio.Queue()
        await retry.put((8.0, 8.0, wav))
        await retry.put(None)

        async def _rec(*_a, **_k):
            return [(8.0, 10.0, "segundo")]

        await self._correr(retry, manager, AsyncMock(), _rec)

        indices = [
            lin for lin in manager.srt_path.read_text(encoding="utf-8").splitlines()
            if lin.strip().isdigit()
        ]
        assert indices == ["1", "2", "3"]

    async def test_reintenta_hasta_que_sale(self, tmp_path):
        manager = self._manager(tmp_path)
        wav = tmp_path / "c.wav"
        wav.write_bytes(b"x")
        retry: asyncio.Queue = asyncio.Queue()
        await retry.put((8.0, 8.0, wav))
        intentos = {"n": 0}

        async def _rec(*_a, **_k):
            intentos["n"] += 1
            if intentos["n"] < 3:
                return None
            await retry.put(None)
            return [(8.0, 10.0, "al tercer intento")]

        await self._correr(retry, manager, AsyncMock(), _rec)
        assert intentos["n"] == 3
        assert "al tercer intento" in manager.srt_path.read_text(encoding="utf-8")

    async def test_un_trozo_vacio_no_se_reintenta_para_siempre(self, tmp_path):
        manager = self._manager(tmp_path)
        wav = tmp_path / "c.wav"
        wav.write_bytes(b"x")
        retry: asyncio.Queue = asyncio.Queue()
        await retry.put((8.0, 8.0, wav))
        await retry.put(None)
        intentos = {"n": 0}

        async def _rec(*_a, **_k):
            intentos["n"] += 1
            return []

        await self._correr(retry, manager, AsyncMock(), _rec)
        assert intentos["n"] == 1
        assert not wav.exists()

    async def test_no_deja_wavs_al_cerrar(self, tmp_path):
        """Si no se recupero, el temporal igual se limpia."""
        manager = self._manager(tmp_path)
        wav = tmp_path / "c.wav"
        wav.write_bytes(b"x")
        retry: asyncio.Queue = asyncio.Queue()
        await retry.put((8.0, 8.0, wav))
        await retry.put(None)

        async def _rec(*_a, **_k):
            return None

        await self._correr(retry, manager, AsyncMock(), _rec)
        assert not wav.exists()

    async def test_la_cola_tiene_techo(self, tmp_path):
        """Una caida total de la API no puede llenarle el disco a nadie."""
        manager = self._manager(tmp_path)
        retry: asyncio.Queue = asyncio.Queue()
        wavs = []
        for i in range(_MAX_PENDIENTES + 20):
            w = tmp_path / f"c{i}.wav"
            w.write_bytes(b"x")
            wavs.append(w)
            await retry.put((float(i * 8), 8.0, w))
        await retry.put(None)

        async def _rec(*_a, **_k):
            return None

        await self._correr(retry, manager, AsyncMock(), _rec)
        assert not [w for w in wavs if w.exists()]


class TestInsertEntries:
    def test_no_pierde_lo_que_ya_estaba(self, tmp_path):
        m = SubtitleManager(tmp_path / "s.srt")
        m.add_entry(0.0, 1.0, "a")
        m.add_entry(10.0, 11.0, "c")
        m.insert_entries([(5.0, 6.0, "b")])
        assert [t for _, _, t in parse_entries(m.srt_path)] == ["a", "b", "c"]

    def test_vacio_no_toca_el_archivo(self, tmp_path):
        m = SubtitleManager(tmp_path / "s.srt")
        m.add_entry(0.0, 1.0, "a")
        antes = m.srt_path.read_text(encoding="utf-8")
        m.insert_entries([])
        assert m.srt_path.read_text(encoding="utf-8") == antes

    def test_sigue_agregando_bien_despues_de_insertar(self, tmp_path):
        m = SubtitleManager(tmp_path / "s.srt")
        m.add_entry(0.0, 1.0, "a")
        m.insert_entries([(5.0, 6.0, "b")])
        m.add_entry(20.0, 21.0, "c")
        assert [t for _, _, t in parse_entries(m.srt_path)] == ["a", "b", "c"]
        indices = [
            lin for lin in m.srt_path.read_text(encoding="utf-8").splitlines()
            if lin.strip().isdigit()
        ]
        assert indices == ["1", "2", "3"]

    def test_al_retomar_un_srt_lee_lo_que_ya_habia(self, tmp_path):
        srt = tmp_path / "s.srt"
        m = SubtitleManager(srt)
        m.add_entry(0.0, 1.0, "viejo")
        m.add_entry(10.0, 11.0, "viejo2")

        retomado = SubtitleManager(srt, append=True)
        retomado.insert_entries([(5.0, 6.0, "nuevo")])
        assert [t for _, _, t in parse_entries(srt)] == ["viejo", "nuevo", "viejo2"]

    def test_no_deja_el_temporal(self, tmp_path):
        m = SubtitleManager(tmp_path / "s.srt")
        m.add_entry(0.0, 1.0, "a")
        m.insert_entries([(5.0, 6.0, "b")])
        assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("texto", ["una linea", "dos\nlineas"])
def test_parse_entries_ida_y_vuelta(tmp_path, texto):
    m = SubtitleManager(tmp_path / "s.srt")
    m.add_entry(1.5, 3.25, texto)
    assert parse_entries(m.srt_path) == [(1.5, 3.25, texto)]


class TestSrtSiempreValido:
    """Una linea en blanco dentro de una entrada rompe el archivo entero.

    En SRT la linea en blanco es el separador de entradas, asi que todo lo que
    va despues queda huerfano, sin indice ni tiempos. Pasó de verdad: el
    traductor solto su comentario dentro del subtitulo y dejo 9 entradas
    partidas en el SRT del usuario.
    """

    REAL = (
        "Las matematicas... no, la fisica es mas dificil.\n\n"
        "Wait, I need to correct myself. The Japanese text means "
        "\"physics is more...\". Following the rules, the translation is:\n\n"
        "La fisica es mas"
    )

    def test_el_texto_con_parrafos_no_parte_el_archivo(self, tmp_path):
        m = SubtitleManager(tmp_path / "s.srt")
        m.add_entry(0.0, 2.0, self.REAL)
        m.add_entry(5.0, 6.0, "siguiente")

        # Contar entradas parseadas no basta: el parser se salta los bloques
        # huerfanos, asi que el numero sale bien aunque el archivo este roto.
        # Lo que hay que mirar es que no sobre ningun bloque.
        texto = m.srt_path.read_text(encoding="utf-8")
        bloques = [b for b in texto.split("\n\n") if b.strip()]
        assert len(bloques) == 2, f"{len(bloques)} bloques para 2 entradas"
        assert len(parse_entries(m.srt_path)) == 2

    def test_tambien_al_insertar(self, tmp_path):
        m = SubtitleManager(tmp_path / "s.srt")
        m.add_entry(10.0, 11.0, "despues")
        m.insert_entries([(0.0, 2.0, self.REAL)])
        entradas = parse_entries(m.srt_path)
        assert len(entradas) == 2
        assert "\n\n" not in entradas[0][2]

    def test_un_subtitulo_de_dos_lineas_sigue_entero(self, tmp_path):
        """El dialogo con guiones es legitimo y no se puede tocar."""
        m = SubtitleManager(tmp_path / "s.srt")
        m.add_entry(0.0, 2.0, "- Hola\n- Adios")
        assert parse_entries(m.srt_path) == [(0.0, 2.0, "- Hola\n- Adios")]
