"""Reconstruccion y scoring del dataset TEDxJP-10K (Laboro.AI, Apache-2.0).

TEDxJP-10K son 10.000 utterances de charlas TEDx en japones, extraidas de
subtitulos manuales de YouTube y corregidas a mano. El repo oficial no
distribuye texto: publica un md5 por utterance y un diff con las correcciones.
Se reconstruye bajando el .vtt, normalizando cada linea, buscando su md5 en la
tabla y aplicando el parche.

Consecuencia: `clean_text` debe ser identica a la del script original o los
md5 no coinciden y las utterances se pierden en silencio.

Ver https://github.com/laboroai/TEDxJP-10K
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import jaconv

logger = logging.getLogger(__name__)

# El video_id de YouTube siempre ocupa 11 caracteres; el resto del utt_id son
# los timestamps en centesimas de segundo: <video_id>-<start8>-<end8>
_VIDEO_ID_LEN = 11
_START_SLICE = slice(12, 20)
_END_SLICE = slice(22, 30)


@dataclass(frozen=True)
class Utterance:
    """Una utterance de referencia con su audio asociado."""

    utt_id: str
    video_id: str
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start


def clean_text(text: str) -> str:
    """Normaliza una linea de subtitulo.

    Portado literalmente de compose_tedxjp10k.py: los md5 del dataset se
    calculan sobre esta salida, asi que no se puede tocar.
    """
    text = text.replace("​", "")

    # Han-kaku -> Zen-kaku
    text = jaconv.h2z(text, kana=False, ascii=True, digit=True)

    # Creditos de traduccion/subtitulado que no estan en el audio
    text = re.sub(r"翻訳：.*", "", text)
    text = re.sub(r"字幕：.*", "", text)

    # Fragmentos en otros idiomas
    text = re.sub(r"（.*語）.*", "", text)

    # Anotaciones que solo existen en el subtitulo: （笑）, ［の］...
    text = re.sub(r"（.*?）", "", text)
    text = re.sub(r"［.*?］", "", text)

    text = re.sub(r"「|」|『|』|［|］|\"|“|”|＂", " ", text)
    text = re.sub(r"！|？|、|。|…|—|・・・|．．．", " ", text)
    text = re.sub(r"〜", "ー", text)
    text = re.sub(r"♪", "", text)
    text = re.sub(r"―$", "", text)

    text = re.sub(r"\s+", " ", text)
    return re.sub(r"\s+$|^\s+", "", text)


def normalize_for_scoring(text: str) -> str:
    """Normaliza una hipotesis de ASR para compararla contra la referencia.

    Aplica la misma limpieza que la referencia y elimina todos los espacios:
    en japones no hay separacion de palabras, y donde el ASR la ponga seria
    ruido puro en el CER.
    """
    return "".join(clean_text(text).split())


def utt_hash(video_id: str, clean_subtext: str) -> str:
    """Calcula el md5 con que el dataset indexa una utterance."""
    payload = f"{video_id}+{clean_subtext}".encode()
    return hashlib.md5(payload, usedforsecurity=False).hexdigest()


def _timestamp_to_seconds(timestamp: str) -> float:
    hours, minutes, seconds = timestamp.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def load_vtt(vtt_path: Path) -> list[tuple[float, float, str]]:
    """Parsea un .vtt y retorna (inicio, fin, texto_normalizado) por cue."""
    raw: list[tuple[str, str, str]] = []
    current: list[str] = []
    start = end = "00:00:00.000"
    reading = False

    for line in vtt_path.read_text(encoding="utf8").splitlines():
        line = line.rstrip()
        if "-->" in line:
            start, _, end = line.split()[:3]
            reading = True
        elif re.match(r"^\s*$", line):
            if current:
                raw.append((start, end, " ".join(current)))
                current = []
            reading = False
        elif reading:
            current.append(line)

    if current:  # ultimo cue sin linea en blanco final
        raw.append((start, end, " ".join(current)))

    return [
        (_timestamp_to_seconds(a), _timestamp_to_seconds(b), clean_text(t))
        for a, b, t in raw
    ]


def apply_patch(oldtext: str, patch: list[str]) -> str:
    """Aplica un diff caracter a caracter del dataset.

    Cada elemento es "" (conservar), "+ X" (insertar X) o "- X" (borrar).
    """
    cursor = 0
    newtext: list[str] = []
    for op in patch:
        if op == "":
            if cursor >= len(oldtext):
                raise ValueError("El parche excede el largo del texto original")
            newtext.append(oldtext[cursor])
            cursor += 1
        elif op[0] == "+":
            newtext.append(op[2])
        elif op[0] == "-":
            cursor += 1
    return "".join(newtext)


def build_utterances(
    vtt_dir: Path, data_dir: Path, video_ids: set[str] | None = None,
) -> tuple[list[Utterance], int]:
    """Reconstruye las utterances de los videos cuyo .vtt este en vtt_dir.

    video_ids restringe el resultado a esos videos: sin el filtro se generarian
    utterances de videos cuyo audio no se pudo bajar.

    Retorna (utterances, cantidad_no_encontrada). Lo segundo es la senal de
    salud: si YouTube edito un subtitulo desde 2020, su md5 ya no coincide.
    """
    subtitle_index: dict[str, tuple[str, str]] = {}
    available: set[str] = set()

    for vtt_path in sorted(vtt_dir.glob("*.ja.vtt")):
        video_id = vtt_path.name.split(".")[0]
        if video_ids is not None and video_id not in video_ids:
            continue
        available.add(video_id)
        for _, _, subtext in load_vtt(vtt_path):
            if not subtext:
                continue
            subtext = "".join(subtext.split())
            subtitle_index[utt_hash(video_id, subtext)] = (video_id, subtext)

    utterances: list[Utterance] = []
    missing = 0

    for line in (data_dir / "utt_id_table.csv").read_text(encoding="utf8").splitlines():
        if not line.strip():
            continue
        md5hash, utt_id = line.strip().split()
        if utt_id[:_VIDEO_ID_LEN] not in available:
            continue

        entry = subtitle_index.get(md5hash)
        if entry is None:
            logger.warning("Subtitulo cambiado en YouTube, se descarta: %s", utt_id)
            missing += 1
            continue

        video_id, rawtext = entry
        diff_path = data_dir / "diffs" / f"{utt_id}.diff"
        text = (
            apply_patch(rawtext, json.loads(diff_path.read_text(encoding="utf8")))
            if diff_path.exists()
            else rawtext
        )

        utterances.append(Utterance(
            utt_id=utt_id,
            video_id=video_id,
            start=int(utt_id[_START_SLICE]) / 100,
            end=int(utt_id[_END_SLICE]) / 100,
            text=text,
        ))

    utterances.sort(key=lambda u: (u.video_id, u.start))
    return utterances, missing


def edit_distance(ref: str, hyp: str) -> int:
    """Distancia de Levenshtein a nivel caracter."""
    if ref == hyp:
        return 0
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)

    previous = list(range(len(hyp) + 1))
    for i, ref_char in enumerate(ref, start=1):
        current = [i]
        for j, hyp_char in enumerate(hyp, start=1):
            current.append(min(
                previous[j] + 1,                                   # borrado
                current[j - 1] + 1,                                # insercion
                previous[j - 1] + (ref_char != hyp_char),          # sustitucion
            ))
        previous = current
    return previous[-1]


@dataclass
class CerResult:
    """CER agregado sobre un corpus."""

    errors: int
    ref_chars: int
    utterances: int

    @property
    def cer(self) -> float:
        return self.errors / self.ref_chars if self.ref_chars else 0.0


def corpus_cer(pairs: list[tuple[str, str]]) -> CerResult:
    """CER agregado: suma de errores / suma de caracteres de referencia.

    Es el estandar en ASR y no equivale al promedio de los CER por utterance
    (ese sobrepondera las utterances cortas).
    """
    errors = sum(edit_distance(ref, hyp) for ref, hyp in pairs)
    ref_chars = sum(len(ref) for ref, _ in pairs)
    return CerResult(errors=errors, ref_chars=ref_chars, utterances=len(pairs))
