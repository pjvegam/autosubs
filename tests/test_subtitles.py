
from itertools import pairwise
from typing import ClassVar

from autosubs.subtitles import (
    MAX_LINES,
    MIN_DISPLAY,
    SubtitleManager,
    _format_timestamp,
    layout_turns,
)


class TestFormatTimestamp:
    def test_zero(self):
        assert _format_timestamp(0.0) == "00:00:00,000"

    def test_seconds_only(self):
        assert _format_timestamp(5.0) == "00:00:05,000"

    def test_with_millis(self):
        assert _format_timestamp(5.5) == "00:00:05,500"

    def test_minutes(self):
        assert _format_timestamp(65.0) == "00:01:05,000"

    def test_hours(self):
        assert _format_timestamp(3661.123) == "01:01:01,123"

    def test_large_value(self):
        assert _format_timestamp(36000.0) == "10:00:00,000"

    def test_fractional_millis(self):
        # 1.9999 -> 1999ms -> deberia truncar a 999
        assert _format_timestamp(1.999) == "00:00:01,999"


class TestSubtitleManager:
    def test_creates_empty_file(self, tmp_path):
        srt = tmp_path / "test.srt"
        SubtitleManager(srt)
        assert srt.exists()
        assert srt.read_text(encoding="utf-8") == ""

    def test_add_single_entry(self, tmp_path):
        srt = tmp_path / "test.srt"
        manager = SubtitleManager(srt)
        manager.add_entry(0.0, 10.0, "Hola mundo")

        content = srt.read_text(encoding="utf-8")
        assert "1\n" in content
        assert "00:00:00,000 --> 00:00:10,000\n" in content
        assert "Hola mundo\n" in content

    def test_add_multiple_entries(self, tmp_path):
        srt = tmp_path / "test.srt"
        manager = SubtitleManager(srt)
        manager.add_entry(0.0, 10.0, "Primera linea")
        manager.add_entry(10.0, 20.0, "Segunda linea")

        content = srt.read_text(encoding="utf-8")
        lines = content.strip().split("\n\n")
        assert len(lines) == 2
        assert lines[0].startswith("1\n")
        assert lines[1].startswith("2\n")

    def test_entry_format_is_valid_srt(self, tmp_path):
        srt = tmp_path / "test.srt"
        manager = SubtitleManager(srt)
        manager.add_entry(61.5, 71.5, "Texto de prueba")

        content = srt.read_text(encoding="utf-8")
        expected = (
            "1\n"
            "00:01:01,500 --> 00:01:11,500\n"
            "Texto de prueba\n"
            "\n"
        )
        assert content == expected

    def test_append_does_not_overwrite(self, tmp_path):
        srt = tmp_path / "test.srt"
        manager = SubtitleManager(srt)
        manager.add_entry(0.0, 10.0, "Primero")
        manager.add_entry(10.0, 20.0, "Segundo")

        content = srt.read_text(encoding="utf-8")
        assert "Primero" in content
        assert "Segundo" in content


class TestLayoutTurns:
    """Los turnos reales de mai-transcribe-2 duran lo que duran hablados.

    Varios son de centesimas ("うん" de 0.06s). Si se escriben tal cual, el
    subtitulo existe pero no se alcanza a leer.
    """

    # Turnos medidos de verdad sobre S8bhG9N0my0 (dos personas), chunk 2300.
    REAL: ClassVar[list[tuple[float, float, str]]] = [
        (0.16, 5.90, "Hay que sacar el tiempo a la fuerza."),
        (5.92, 5.98, "Bueno"),
        (6.00, 6.52, "parece imposible"),
        (6.54, 6.58, "si"),
        (6.60, 6.80, "eso"),
        (6.92, 8.04, "quiza sea dificil"),
    ]

    def test_ninguna_entrada_es_ilegible(self):
        for inicio, fin, _ in layout_turns(self.REAL, 0.0):
            assert fin - inicio >= MIN_DISPLAY - 1e-6, (
                f"entrada de {fin - inicio:.2f}s: no se alcanza a leer"
            )

    def test_no_se_pisan_entre_si(self):
        entradas = layout_turns(self.REAL, 0.0)
        for (_, fin, _), (inicio, _, _) in pairwise(entradas):
            assert inicio >= fin, f"la entrada arranca en {inicio} y la previa termina en {fin}"

    def test_no_se_pierde_texto(self):
        junto = " ".join(texto for _, _, texto in layout_turns(self.REAL, 0.0))
        for _, _, original in self.REAL:
            assert original in junto

    def test_respeta_el_tope_de_lineas(self):
        for _, _, texto in layout_turns(self.REAL, 0.0):
            assert len(texto.splitlines()) <= MAX_LINES

    def test_agrupa_con_guion_de_dialogo(self):
        entradas = layout_turns(self.REAL, 0.0)
        agrupadas = [t for _, _, t in entradas if len(t.splitlines()) > 1]
        assert agrupadas, "los turnos de centesimas tenian que agruparse"
        for texto in agrupadas:
            assert all(linea.startswith("- ") for linea in texto.splitlines())

    def test_un_turno_comodo_no_lleva_guion(self):
        [(_, _, texto)] = layout_turns([(0.0, 5.0, "Tengo tiempo de sobra.")], 0.0)
        assert texto == "Tengo tiempo de sobra."

    def test_desplaza_al_tiempo_absoluto_del_chunk(self):
        [(inicio, fin, _)] = layout_turns([(1.0, 4.0, "hola")], 600.0)
        assert inicio == 601.0
        assert fin > 601.0

    def test_no_pisa_el_subtitulo_del_chunk_anterior(self):
        # el chunk arranca en 600 pero el subtitulo previo termina en 602.5
        [(inicio, _, _)] = layout_turns([(0.5, 2.0, "hola")], 600.0, floor=602.5)
        assert inicio >= 602.5

    def test_sin_piso_arranca_en_su_tiempo_real(self):
        [(inicio, _, _)] = layout_turns([(0.5, 2.0, "hola")], 600.0)
        assert inicio == 600.5

    def test_sin_turnos(self):
        assert layout_turns([], 0.0) == []


class TestLayoutNoSeAtrasa:
    """Dar tiempo de lectura no puede costar sincronia con el audio.

    Si una entrada se estira por encima del siguiente turno, empuja al que
    viene, y ese empujon se hereda. Medido sobre dos minutos densos de un
    stream real: 35 entradas seguidas pegadas y 2.8s de atraso contra el audio.
    """

    # Charla cruzada: un turno cada 1.2s, cada uno con mas texto del que da
    # tiempo a leer. Es el caso que destapo el atraso.
    DENSA: ClassVar[list[tuple[float, float, str]]] = [
        (i * 1.2, i * 1.2 + 1.0, f"Frase {i} con texto de sobra para no poder leerse entera")
        for i in range(20)
    ]

    def _atrasos(self) -> list[float]:
        entradas = layout_turns(self.DENSA, 0.0)
        atrasos, i = [], 0
        for inicio, _, texto in entradas:
            atrasos.append(inicio - self.DENSA[i][0])
            i += len(texto.splitlines())
        return atrasos

    def test_el_atraso_no_crece_con_el_tiempo(self):
        atrasos = self._atrasos()
        assert max(atrasos) < MIN_DISPLAY, (
            f"el atraso llego a {max(atrasos):.1f}s; se va acumulando"
        )

    def test_la_ultima_entrada_sigue_pegada_al_audio(self):
        assert self._atrasos()[-1] < MIN_DISPLAY

    def test_aun_asi_no_se_pisan(self):
        for (_, fin, _), (inicio, _, _) in pairwise(layout_turns(self.DENSA, 0.0)):
            assert inicio >= fin

    def test_aun_asi_ninguna_es_ilegible(self):
        for inicio, fin, _ in layout_turns(self.DENSA, 0.0):
            assert fin - inicio >= MIN_DISPLAY - 1e-6
