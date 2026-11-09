"""Mide CER real contra el ground truth de TEDxJP-10K.

A diferencia de los otros benchmarks del repo (que comparan modelos entre si
sin referencia), aca hay transcripcion humana verificada, asi que el numero es
absoluto y comparable con la literatura.

    uv run --extra tedx scripts/tedx_build.py --videos 10
    uv run --extra tedx --extra local-whisper scripts/tedx_eval.py \
        --engines local-turbo,gpt-4o-transcribe

Protocolo: un clip de audio por utterance, transcripcion independiente, CER
agregado (suma de errores / suma de caracteres de referencia).

OJO: TEDx es charla preparada con microfono de solapa. Un CER bajo aca NO
predice el rendimiento sobre streams con audio de juego encima.
"""

import argparse
import asyncio
import functools
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

print = functools.partial(print, flush=True)  # type: ignore[assignment]

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

load_dotenv(_PROJECT_ROOT / ".env")

from autosubs.asr_eval import KNOWN_ENGINES, EngineReport, run_engine
from autosubs.costs import CostTracker
from autosubs.tedx import corpus_cer, edit_distance, normalize_for_scoring


def score(report: EngineReport, utterances: list[dict], audio_seconds: float) -> dict:
    """Calcula CER agregado, costo y factor de tiempo real."""
    pairs = [
        (
            normalize_for_scoring(u["text"]),
            normalize_for_scoring(report.hypotheses.get(u["utt_id"], "")),
        )
        for u in utterances
    ]
    result = corpus_cer(pairs)

    tracker = CostTracker(whisper_model=report.engine)
    tracker.add_whisper(audio_seconds)
    empty = report.empty

    per_utt = sorted(
        (
            {
                "utt_id": u["utt_id"],
                "cer": edit_distance(ref, hyp) / len(ref) if ref else 0.0,
                "ref": ref,
                "hyp": hyp,
            }
            for u, (ref, hyp) in zip(utterances, pairs, strict=True)
        ),
        key=lambda d: d["cer"],
        reverse=True,
    )

    return {
        "engine": report.engine,
        "cer": result.cer,
        "errors": result.errors,
        "ref_chars": result.ref_chars,
        "utterances": result.utterances,
        "empty": empty,
        "elapsed_seconds": round(report.elapsed, 1),
        "realtime_factor": round(report.elapsed / audio_seconds, 4),
        "cost_usd": round(tracker.whisper_cost(), 4),
        "worst": per_utt[:10],
    }


def render_report(scores: list[dict], manifest: dict) -> str:
    audio_min = manifest["total_audio_seconds"] / 60
    lines = [
        "# Benchmark TEDxJP-10K (ground truth verificado)\n",
        f"**Dataset:** TEDxJP-10K v{manifest['version']} — {manifest['utterance_count']} "
        f"utterances / {audio_min:.1f} min de audio / {len(manifest['videos'])} videos\n",
        f"**Descartadas por subtitulo editado en YouTube:** {manifest['discarded_count']}\n",
        "**Metrica:** CER agregado (suma de errores / suma de caracteres de referencia), "
        "texto normalizado sin espacios ni puntuacion.\n",
        "## Resultados\n",
        "| Motor | CER | Errores | Vacias | Tiempo | xRT | Costo |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in sorted(scores, key=lambda d: d["cer"]):
        lines.append(
            f"| {s['engine']} | **{s['cer']:.2%}** | {s['errors']:,}/{s['ref_chars']:,} "
            f"| {s['empty']} | {s['elapsed_seconds']:.0f}s | {s['realtime_factor']:.3f}x "
            f"| ${s['cost_usd']:.3f} |",
        )

    lines.append("\n## Peores utterances por motor\n")
    for s in sorted(scores, key=lambda d: d["cer"]):
        lines.append(f"### {s['engine']}\n")
        lines.append("| CER | Referencia | Hipotesis |")
        lines.append("|---|---|---|")
        for w in s["worst"]:
            hyp = w["hyp"] or "_(vacia)_"
            lines.append(f"| {w['cer']:.0%} | {w['ref']} | {hyp} |")
        lines.append("")

    lines.append(
        "\n> TEDx es habla preparada con microfono de solapa y sin ruido de fondo. "
        "Estos numeros miden calidad de transcripcion pura, no el comportamiento "
        "sobre streams con audio de juego ni las alucinaciones en silencio.",
    )
    return "\n".join(lines) + "\n"


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=_PROJECT_ROOT / "tedx_data/manifest.json")
    parser.add_argument(
        "--engines", default="local-turbo,gpt-4o-transcribe",
        help="lista separada por comas: " + ",".join(KNOWN_ENGINES),
    )
    parser.add_argument("--limit", type=int, default=None, help="usar solo N utterances")
    parser.add_argument("--language", default="ja")
    parser.add_argument("--out", type=Path, default=_PROJECT_ROOT / "benchmark_tedx.md")
    args = parser.parse_args()

    if not args.manifest.exists():
        sys.exit(f"Falta {args.manifest}. Corre primero: scripts/tedx_build.py")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    root = args.manifest.parent
    utterances = manifest["utterances"][: args.limit]

    clips = [(u["utt_id"], root / u["clip"]) for u in utterances]
    if missing := [c for _, c in clips if not c.exists()]:
        sys.exit(f"Faltan {len(missing)} clips (ej: {missing[0]}). Recorre tedx_build.py")

    audio_seconds = sum(u["end"] - u["start"] for u in utterances)
    print(
        f"Evaluando {len(utterances)} utterances "
        f"({audio_seconds / 60:.1f} min de audio)\n",
    )

    scores: list[dict] = []
    for engine in args.engines.split(","):
        engine = engine.strip()
        print(f"=== {engine} ===")
        try:
            report = await run_engine(
                engine, clips, args.language,
                on_progress=lambda done, total: print(f"  {done}/{total}"),
            )
        except (RuntimeError, OSError) as e:
            print(f"  fallo: {e}\n")
            continue

        s = score(report, utterances, audio_seconds)
        scores.append(s)
        print(
            f"  CER {s['cer']:.2%} | {s['elapsed_seconds']:.0f}s "
            f"| {s['realtime_factor']:.3f}xRT | ${s['cost_usd']:.3f}\n",
        )

    if not scores:
        sys.exit("Ningun motor completo la evaluacion")

    args.out.write_text(render_report(scores, manifest), encoding="utf-8")
    (root / "scores.json").write_text(
        json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"Reporte: {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
