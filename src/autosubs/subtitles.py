import logging
import re
from collections.abc import Sequence
from pathlib import Path

logger = logging.getLogger(__name__)

_TS_RE = re.compile(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})")

# Tiempos de lectura. Netflix y TED usan 5/6s (0.83s) de minimo y ~17 caracteres
# por segundo para adultos; redondeo el minimo a 1s. Sin esto un turno corto
# (un "うん" de 0.06s) se mostraria lo que dura hablado, o sea nada.
READING_CPS = 17.0
MIN_DISPLAY = 1.0
MAX_DISPLAY = 7.0

# Separacion entre dos entradas seguidas: suficiente para que no se pisen,
# imperceptible al leer.
ENTRY_GAP = 0.08

# Maximo de lineas por entrada. La convencion de subtitulado son 2; permito 3
# porque en un ida y vuelta rapido hace falta agrupar, pero no mas: el bloque de
# 8s con 5 lineas apelmazadas es justo lo que esto viene a arreglar.
MAX_LINES = 3


def _needed_display(text: str) -> float:
    """Cuanto tiene que durar en pantalla para poder leerse."""
    return min(max(len(text) / READING_CPS, MIN_DISPLAY), MAX_DISPLAY)


def layout_turns(
    turns: Sequence[tuple[float, float, str]],
    chunk_start: float,
    floor: float = 0.0,
) -> list[tuple[float, float, str]]:
    """Convierte turnos de habla en entradas SRT legibles.

    `turns` son (start, end, texto) relativos al chunk; `floor` es el final
    absoluto del ultimo subtitulo ya escrito. Devuelve entradas absolutas que
    no se pisan entre si y que duran lo que tardan en leerse, no lo que duran
    habladas.

    Cuando un turno no tiene hueco para leerse antes de que arranque el
    siguiente, los agrupa en una sola entrada de dialogo con guiones, que es la
    convencion de subtitulado de toda la vida:

        - ¿Estas bien?
        - Si, perdon.

    Una entrada puede terminar despues del chunk, y hace falta que pueda: en un
    chunk medido de verdad, los ultimos 2.1s de audio dan 5.5s de lectura. Por
    eso el que llama pasa `floor` y los subtitulos se corren hacia adelante en
    los tramos densos. No se acumula sin techo porque en cuanto hay una pausa
    el turno vuelve a arrancar en su tiempo real.
    """
    if not turns:
        return []

    grupos: list[tuple[float, float, list[str]]] = []
    i = 0
    while i < len(turns):
        start = turns[i][0]
        lineas = [turns[i][2]]
        j = i + 1
        while j < len(turns) and len(lineas) < MAX_LINES:
            if turns[j][0] - start >= _needed_display("\n".join(lineas)) + ENTRY_GAP:
                break
            lineas.append(turns[j][2])
            j += 1
        grupos.append((start, turns[j - 1][1], lineas))
        i = j

    entradas: list[tuple[float, float, str]] = []
    cursor = floor - chunk_start
    for i, (start, fin_hablado, lineas) in enumerate(grupos):
        texto = "\n".join(f"- {lin}" for lin in lineas) if len(lineas) > 1 else lineas[0]
        inicio = max(start, cursor)
        end = max(fin_hablado, inicio + _needed_display(texto))
        if i + 1 < len(grupos):
            # Estirarse por encima de quien ya esta hablando empuja al que
            # sigue, y ese empujon se hereda entrada tras entrada. Medido
            # sobre dos minutos densos de un stream real: cadenas de 35
            # entradas seguidas y hasta 2.8s de atraso contra el audio.
            end = min(end, grupos[i + 1][0] - ENTRY_GAP)
        end = max(end, inicio + MIN_DISPLAY)
        entradas.append((chunk_start + inicio, chunk_start + end, texto))
        cursor = end + ENTRY_GAP
    return entradas


def _format_timestamp(seconds: float) -> str:
    """Convierte segundos a formato SRT HH:MM:SS,mmm."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _parse_timestamp(ts: str) -> float:
    """Parsea HH:MM:SS,mmm a segundos."""
    if m := _TS_RE.match(ts):
        return int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]) + int(m[4]) / 1000
    return 0.0


def srt_last_end_time(srt_path: Path) -> float:
    """Retorna el end timestamp de la ultima entrada SRT en segundos, o 0.0."""
    last_end = 0.0
    with open(srt_path, encoding="utf-8") as f:
        for line in f:
            if " --> " in line:
                end_str = line.split(" --> ")[1].strip()
                last_end = _parse_timestamp(end_str)
    return last_end


def parse_entries(srt_path: Path) -> list[tuple[float, float, str]]:
    """Lee un SRT a (inicio, fin, texto). Lo que no cuadre se ignora."""
    entradas: list[tuple[float, float, str]] = []
    bloques = srt_path.read_text(encoding="utf-8").split("\n\n")
    for bloque in bloques:
        lineas = [lin for lin in bloque.splitlines() if lin.strip()]
        if len(lineas) < 3 or " --> " not in lineas[1]:
            continue
        ini, fin = lineas[1].split(" --> ")
        entradas.append((
            _parse_timestamp(ini.strip()),
            _parse_timestamp(fin.strip()),
            "\n".join(lineas[2:]),
        ))
    return entradas


class SubtitleManager:
    """Gestiona un archivo SRT, normalmente append-only.

    Normalmente las entradas llegan en orden y se van agregando al final, que
    es lo barato. Pero un chunk que fallo y se recupero despues pertenece a un
    momento ya pasado, y meterlo obliga a reescribir el archivo entero para
    que los indices sigan siendo consecutivos: de eso se encarga
    `insert_entries`.
    """

    def __init__(self, srt_path: Path, *, append: bool = False) -> None:
        self.srt_path = srt_path
        self._entries: list[tuple[float, float, str]] = []
        if append and srt_path.exists():
            self._entries = parse_entries(srt_path)
            self._index = len(self._entries) + 1
        else:
            self._index = 1
            self.srt_path.write_text("", encoding="utf-8")

    def add_entry(self, start: float, end: float, text: str) -> None:
        """Agrega una entrada al archivo SRT."""
        entry = (
            f"{self._index}\n"
            f"{_format_timestamp(start)} --> {_format_timestamp(end)}\n"
            f"{text}\n\n"
        )
        with open(self.srt_path, "a", encoding="utf-8") as f:
            f.write(entry)
        logger.debug(
            "SRT #%d [%s -> %s]: %s",
            self._index,
            _format_timestamp(start),
            _format_timestamp(end),
            text[:50],
        )
        self._index += 1
        self._entries.append((start, end, text))

    def insert_entries(self, entradas: list[tuple[float, float, str]]) -> None:
        """Mete entradas en su sitio por tiempo y reescribe el SRT.

        Es para lo que llega tarde: un chunk que fallo en su momento y se
        recupero despues va a un timestamp ya pasado, asi que no se puede
        agregar al final sin romper el orden ni la numeracion.

        Escribe a un temporal y reemplaza porque MPV esta leyendo este mismo
        archivo: si lo pillara a medias se quedaria sin subtitulos.
        """
        if not entradas:
            return
        self._entries.extend(entradas)
        self._entries.sort(key=lambda e: e[0])
        texto = "".join(
            f"{i}\n{_format_timestamp(ini)} --> {_format_timestamp(fin)}\n{txt}\n\n"
            for i, (ini, fin, txt) in enumerate(self._entries, 1)
        )
        tmp = self.srt_path.with_suffix(".srt.tmp")
        tmp.write_text(texto, encoding="utf-8")
        tmp.replace(self.srt_path)
        self._index = len(self._entries) + 1
