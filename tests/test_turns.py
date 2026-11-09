"""Turnos de habla: de los segments del ASR a lineas traducidas."""

# ruff: noqa: RUF001  (el japones del ASR trae signos fullwidth de verdad)

from autosubs.pipeline import _align_turns
from autosubs.transcribe import TranscriptSegment, supports_turns


def _turnos() -> list[TranscriptSegment]:
    return [
        TranscriptSegment(text="大丈夫？", start=2.56, end=3.20),
        TranscriptSegment(text="いいんじゃないの？", start=4.48, end=5.54),
    ]


class TestSupportsTurns:
    def test_mai_2_los_da(self):
        assert supports_turns("microsoft/mai-transcribe-2")

    def test_mai_15_no(self):
        """mai-transcribe-1.5 contesta 400 a verbose_json."""
        assert not supports_turns("microsoft/mai-transcribe-1.5")

    def test_gpt_4o_no(self):
        assert not supports_turns("gpt-4o-transcribe")

    def test_modelo_desconocido_no(self):
        assert not supports_turns("algun/modelo-nuevo")


class TestAlignTurns:
    def test_reparte_una_linea_por_turno(self):
        alineados = _align_turns(_turnos(), "¿Estas bien?\n¿No esta bien asi?")
        assert [t.text for t in alineados] == ["¿Estas bien?", "¿No esta bien asi?"]

    def test_conserva_los_tiempos_del_audio(self):
        alineados = _align_turns(_turnos(), "¿Estas bien?\n¿No esta bien asi?")
        assert [(t.start, t.end) for t in alineados] == [(2.56, 3.20), (4.48, 5.54)]

    def test_si_el_traductor_junta_lineas_vuelve_al_bloque(self):
        """Repartir mal los tiempos es peor que no repartirlos."""
        assert _align_turns(_turnos(), "¿Estas bien? ¿No esta bien asi?") is None

    def test_si_el_traductor_parte_de_mas_vuelve_al_bloque(self):
        assert _align_turns(_turnos(), "¿Estas\nbien?\n¿No esta\nbien asi?") is None

    def test_ignora_lineas_vacias_del_traductor(self):
        alineados = _align_turns(_turnos(), "¿Estas bien?\n\n¿No esta bien asi?\n")
        assert alineados is not None
        assert len(alineados) == 2

    def test_sin_turnos_no_hay_reparto(self):
        assert _align_turns(None, "cualquier cosa") is None

    def test_un_solo_turno_tambien_se_aprovecha(self):
        """Con un turno el tiempo real sigue siendo mejor que el chunk entero."""
        uno = [TranscriptSegment(text="そうね。", start=1.5, end=4.0)]
        alineados = _align_turns(uno, "Pues si.")
        assert alineados is not None
        assert (alineados[0].start, alineados[0].end) == (1.5, 4.0)
