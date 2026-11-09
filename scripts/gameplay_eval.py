"""Compara motores ASR sobre audio real de gameplay, sin ground truth.

En un stream de gameplay no hay transcripcion verificada, asi que no se puede
medir CER. Lo que si se mide —y es lo que realmente duele en produccion— es si
el motor inventa texto cuando no hay voz.

    uv run --extra local-whisper scripts/gameplay_eval.py NSfLVABT8ck \
        --start-time 8160 --duration 1200 \
        --engines local-turbo,gpt-4o-transcribe,microsoft/mai-transcribe-1.5

Tres metricas:
  - Frases de outro: coincidencias con la blacklist de despedidas de YouTube,
    que en medio de un stream son siempre alucinacion.
  - Texto sin kana: en un stream japones, una salida sin un solo hiragana ni
    katakana es el modelo alucinando en otro idioma. Captura la basura que la
    blacklist japonesa no ve ("Bang!", texto en arabe, persa o checo).
  - Texto en silencio: chunks que el VAD marca con poca voz pero que igual
    produjeron texto.

El audio se segmenta con el mismo VAD que usa el pipeline, para que los chunks
sean los que veria autosubs en vivo.
"""

import argparse
import asyncio
import functools
import json
import re
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

print = functools.partial(print, flush=True)  # type: ignore[assignment]

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

load_dotenv(_PROJECT_ROOT / ".env")

from autosubs.asr_eval import KNOWN_ENGINES, EngineReport, run_engine
from autosubs.audio import TRANSCRIPTION_AUDIO_FILTER
from autosubs.costs import CostTracker
from autosubs.vad import SAMPLE_RATE, AudioChunk, VadSegmenter

LOGS_DIR = _PROJECT_ROOT / "autosubs_logs"

# Un chunk con menos voz que esto es, para el VAD, practicamente silencio.
SILENCE_RATIO = 0.30

# Hiragana (U+3041-309F) y katakana (U+30A0-30FF). El japones hablado casi
# siempre los lleva (particulas, terminaciones), asi que su ausencia total
# delata una alucinacion en otro idioma. Se excluyen los kanji a proposito:
# el chino tambien los usa.
KANA = re.compile("[\u3041-\u309f\u30a0-\u30ff]")

# Despedidas de video de YouTube. En medio de un stream son alucinacion:
# el modelo las aprendio de millones de finales de video subtitulados.
OUTRO_PATTERNS = re.compile(
    "|".join([
        "ご視聴ありがとう",
        "ご覧いただきありがとう",
        "最後まで見て",
        "チャンネル登録",
        "高評価",
        "次回もお楽しみ",
        "また次の動画",
        "また次回",
        "お疲れ様でした",
        "バイバイ",
    ]),
)


def resolve_source(video: str) -> Path:
    """Acepta una ruta directa o un video_id presente en autosubs_logs/."""
    if (path := Path(video)).exists():
        return path
    for ext in (".mkv", ".mp4", ".webm", ".m4a", ".wav"):
        if (candidate := LOGS_DIR / f"{video}{ext}").exists():
            return candidate
    sys.exit(f"No encuentro '{video}' ni como ruta ni en {LOGS_DIR}")


async def segment_audio(
    source: Path, out_dir: Path, start: int, duration: int, segment_duration: int,
) -> list[AudioChunk]:
    """Extrae el tramo pedido y lo corta con el VAD del pipeline."""
    out_dir.mkdir(parents=True, exist_ok=True)
    segmenter = VadSegmenter(
        out_dir,
        min_duration=segment_duration * 0.6,
        target_duration=float(segment_duration),
        max_duration=segment_duration * 1.6,
        aggressiveness=2,
    )

    cmd = [
        "ffmpeg", "-loglevel", "error",
        "-ss", str(start), "-i", str(source), "-t", str(duration),
        "-af", TRANSCRIPTION_AUDIO_FILTER,
        "-ar", str(SAMPLE_RATE), "-ac", "1",
        "-f", "s16le", "-acodec", "pcm_s16le",
        "pipe:1",
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    if proc.stdout is None:
        await proc.wait()
        raise RuntimeError("ffmpeg no abrio stdout")

    chunks: list[AudioChunk] = []
    while data := await proc.stdout.read(SAMPLE_RATE * 2):
        chunks.extend(segmenter.feed(data))
    if remaining := segmenter.flush_remaining():
        chunks.append(remaining)

    await proc.wait()
    return chunks


def score(report: EngineReport, chunks: list[AudioChunk], audio_seconds: float) -> dict:
    """Cuenta alucinaciones de outro y texto producido sobre silencio."""
    tracker = CostTracker(whisper_model=report.engine)
    tracker.add_whisper(audio_seconds)

    outro: list[dict] = []
    no_kana: list[dict] = []
    silent_with_text: list[dict] = []
    silent_total = 0
    with_text = 0

    for chunk in chunks:
        text = report.hypotheses.get(chunk.path.stem, "")
        is_silent = chunk.speech_ratio < SILENCE_RATIO
        silent_total += is_silent
        if text:
            with_text += 1
        entry = {
            "chunk": chunk.path.stem,
            "speech_ratio": round(chunk.speech_ratio, 2),
            "text": text,
        }
        if text and OUTRO_PATTERNS.search(text):
            outro.append(entry)
        if text and not KANA.search(text):
            no_kana.append(entry)
        if is_silent and text:
            silent_with_text.append(entry)

    total = len(chunks)
    garbage = len(outro) + len(no_kana)
    return {
        "engine": report.engine,
        "chunks": total,
        "with_text": with_text,
        "empty": total - with_text,
        "outro_count": len(outro),
        "outro_rate": len(outro) / total if total else 0.0,
        "no_kana_count": len(no_kana),
        "no_kana_rate": len(no_kana) / total if total else 0.0,
        "garbage_count": garbage,
        "garbage_rate": garbage / with_text if with_text else 0.0,
        "silent_chunks": silent_total,
        "silent_with_text": len(silent_with_text),
        "silent_leak_rate": len(silent_with_text) / silent_total if silent_total else 0.0,
        "total_chars": sum(len(t) for t in report.hypotheses.values()),
        "elapsed_seconds": round(report.elapsed, 1),
        "realtime_factor": round(report.elapsed / audio_seconds, 4),
        "cost_usd": round(tracker.whisper_cost(), 4),
        "outro_examples": outro[:8],
        "no_kana_examples": no_kana[:8],
    }


def render_report(scores: list[dict], meta: dict) -> str:
    lines = [
        "# Benchmark sobre gameplay real (sin ground truth)\n",
        f"**Fuente:** `{meta['source']}` — tramo {meta['start'] // 60}-"
        f"{(meta['start'] + meta['duration']) // 60} min "
        f"({meta['duration'] / 60:.0f} min de audio)\n",
        f"**Chunks:** {meta['chunks']} (VAD, target {meta['segment_duration']}s) — "
        f"{meta['silent_chunks']} con menos de {SILENCE_RATIO:.0%} de voz\n",
        "**No hay CER**: sin transcripcion verificada no se mide exactitud. "
        "Se mide invencion sobre silencio.\n",
        "## Resultados\n",
        "| Motor | Basura | Outro | Sin kana | Con texto | Chars | Tiempo | xRT | Costo |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in sorted(scores, key=lambda d: d["garbage_rate"]):
        # elapsed 0 = hipotesis reusadas de cache, no se midio el tiempo
        timing = (
            f"{s['elapsed_seconds']:.0f}s | {s['realtime_factor']:.3f}x"
            if s["elapsed_seconds"] else "n/d | n/d"
        )
        lines.append(
            f"| {s['engine']} | **{s['garbage_count']}/{s['with_text']}** "
            f"({s['garbage_rate']:.1%}) "
            f"| {s['outro_count']} | {s['no_kana_count']} "
            f"| {s['with_text']}/{s['chunks']} | {s['total_chars']:,} "
            f"| {timing} | ${s['cost_usd']:.3f} |",
        )

    lines.append(
        "\n**Basura** = outro + sin kana, sobre los chunks en que el motor escribio algo. "
        "Es la fraccion de subtitulos emitidos que son invencion.\n",
    )

    for title, key in (
        ("Frases de outro", "outro_examples"),
        ("Texto sin kana (alucinacion de idioma)", "no_kana_examples"),
    ):
        lines.append(f"\n## {title}\n")
        for s in sorted(scores, key=lambda d: d["garbage_rate"]):
            lines.append(f"### {s['engine']}\n")
            if not s[key]:
                lines.append("_Ninguna._\n")
                continue
            lines.append("| Chunk | Voz | Texto inventado |")
            lines.append("|---|---|---|")
            for e in s[key]:
                lines.append(f"| {e['chunk']} | {e['speech_ratio']:.0%} | {e['text']} |")
            lines.append("")

    lines.append(
        "\n> 'Texto sobre silencio' es la metrica que importa: son chunks que el VAD "
        "considera casi mudos y en los que el motor igual escribio algo. No depende "
        "de la blacklist, asi que tambien captura alucinaciones que no son despedidas.",
    )
    return "\n".join(lines) + "\n"


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="ruta a un archivo o video_id en autosubs_logs/")
    parser.add_argument("--start-time", type=int, default=0, help="segundo inicial")
    parser.add_argument("--duration", type=int, default=1200, help="segundos a evaluar")
    parser.add_argument("--segment-duration", type=int, default=8, help="target del VAD")
    parser.add_argument(
        "--engines", default="local-turbo,microsoft/mai-transcribe-1.5",
        help="lista separada por comas: " + ",".join(KNOWN_ENGINES),
    )
    parser.add_argument("--language", default="ja")
    parser.add_argument("--out", type=Path, default=_PROJECT_ROOT / "benchmark_gameplay.md")
    parser.add_argument("--work-dir", type=Path, default=_PROJECT_ROOT / "gameplay_data")
    parser.add_argument(
        "--reuse", action="store_true",
        help="reusar hipotesis ya guardadas en work-dir (no vuelve a llamar a la API)",
    )
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg no esta en PATH")

    source = resolve_source(args.video)
    chunk_dir = args.work_dir / "chunks"
    print(f"Fuente: {source}")
    print(f"Segmentando {args.duration / 60:.0f} min desde {args.start_time // 60} min...")

    chunks = await segment_audio(
        source, chunk_dir, args.start_time, args.duration, args.segment_duration,
    )
    if not chunks:
        sys.exit("El VAD no produjo ningun chunk")

    silent = sum(1 for c in chunks if c.speech_ratio < SILENCE_RATIO)
    audio_seconds = sum(c.duration for c in chunks)
    print(
        f"  {len(chunks)} chunks / {audio_seconds / 60:.1f} min de voz — "
        f"{silent} casi mudos ({silent / len(chunks):.1%})\n",
    )

    items = [(c.path.stem, c.path) for c in chunks]
    scores: list[dict] = []
    for engine in args.engines.split(","):
        engine = engine.strip()
        print(f"=== {engine} ===")
        cache = args.work_dir / f"hyp_{engine.replace('/', '_')}.json"

        if args.reuse and cache.exists():
            report = EngineReport(
                engine=engine,
                hypotheses=json.loads(cache.read_text(encoding="utf-8")),
            )
            print(f"  reusando {cache.name} (sin costo ni tiempo medidos)")
        else:
            try:
                report = await run_engine(
                    engine, items, args.language,
                    on_progress=lambda done, total: print(f"  {done}/{total}"),
                )
            except (RuntimeError, OSError) as e:
                print(f"  fallo: {e}\n")
                continue
            cache.write_text(
                json.dumps(report.hypotheses, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        s = score(report, chunks, audio_seconds)
        scores.append(s)
        print(
            f"  basura {s['garbage_count']}/{s['with_text']} ({s['garbage_rate']:.1%}) "
            f"= {s['outro_count']} outro + {s['no_kana_count']} sin kana "
            f"| {s['elapsed_seconds']:.0f}s | ${s['cost_usd']:.3f}\n",
        )

    if not scores:
        sys.exit("Ningun motor completo la evaluacion")

    meta = {
        "source": source.name,
        "start": args.start_time,
        "duration": args.duration,
        "chunks": len(chunks),
        "silent_chunks": silent,
        "segment_duration": args.segment_duration,
    }
    args.out.write_text(render_report(scores, meta), encoding="utf-8")
    (args.work_dir / "scores.json").write_text(
        json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"Reporte: {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
