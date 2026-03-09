"""Benchmark: comparacion de modelos de traduccion.

Mismos segmentos japoneses (YT captions) traducidos por 4 modelos.
Compara calidad y costo de: gpt-4o-mini, gpt-4.1-mini, gpt-4o, gpt-4.1.
"""

import asyncio
import functools
import json
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

# Forzar flush en cada print para ver progreso en tiempo real
print = functools.partial(print, flush=True)  # type: ignore[assignment]

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

load_dotenv(_PROJECT_ROOT / ".env")

from livesubs.stream import _ytdlp_base_args
from livesubs.translate import _build_system_prompt

# --- Constantes ---

VOD_URL = "https://www.youtube.com/watch?v=2kmA5_p3iYA"
MAX_MINUTES = 5
SOURCE_LANG = "japones"
TARGET_LANG = "espanol"
CONTEXT = (
    "Canal: Sakuna Ch. 結城さくな (Yuuki Sakuna) | "
    "Titulo: Revisando el nuevo pollo tori tatsuta de Mos Burger"
)

MODELS = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o", "gpt-4.1"]

# Precios por millon de tokens (USD)
PRICING: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
}


# --- Extraer captions ---

def extract_youtube_captions(url: str, lang: str = "ja") -> list[tuple[float, float, str]]:
    """Descarga auto-captions de YouTube en json3 y retorna segmentos agrupados ~5s."""
    tmp_dir = Path("tmp_captions")
    tmp_dir.mkdir(exist_ok=True)

    cmd = [
        *_ytdlp_base_args(),
        "--write-auto-subs",
        "--sub-langs", lang,
        "--sub-format", "json3",
        "--skip-download",
        "-o", str(tmp_dir / "captions"),
        url,
    ]

    print(f"Descargando captions de YouTube ({lang})...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp fallo: {result.stderr.strip()}")

    json3_files = list(tmp_dir.glob("*.json3"))
    if not json3_files:
        raise FileNotFoundError("No se encontro archivo json3 de captions")

    with json3_files[0].open(encoding="utf-8") as f:
        data = json.load(f)

    for f in tmp_dir.iterdir():
        f.unlink()
    tmp_dir.rmdir()

    raw_segments: list[tuple[float, float, str]] = []
    for event in data.get("events", []):
        start_ms = event.get("tStartMs", 0)
        duration_ms = event.get("dDurationMs", 0)
        segs = event.get("segs")
        if not segs:
            continue
        text = "".join(s.get("utf8", "") for s in segs).strip().replace("\n", " ")
        if not text:
            continue
        start_s = start_ms / 1000.0
        end_s = (start_ms + duration_ms) / 1000.0
        if start_s < MAX_MINUTES * 60:
            raw_segments.append((start_s, end_s, text))

    return _group_segments(raw_segments)


def _group_segments(
    segments: list[tuple[float, float, str]],
    group_duration: float = 5.0,
) -> list[tuple[float, float, str]]:
    if not segments:
        return []

    grouped: list[tuple[float, float, str]] = []
    current_start = segments[0][0]
    current_end = segments[0][1]
    current_texts: list[str] = [segments[0][2]]

    for start, end, text in segments[1:]:
        if start - current_start >= group_duration:
            grouped.append((current_start, current_end, " ".join(current_texts)))
            current_start = start
            current_end = end
            current_texts = [text]
        else:
            current_end = end
            current_texts.append(text)

    grouped.append((current_start, current_end, " ".join(current_texts)))
    return grouped


# --- Traducir ---

async def translate_batch(
    client: AsyncOpenAI,
    segments: list[tuple[float, float, str]],
    model: str,
) -> list[tuple[str, int, int]]:
    """Traduce segmentos y retorna (traduccion, prompt_tokens, completion_tokens)."""
    system_prompt = _build_system_prompt(CONTEXT)
    results: list[tuple[str, int, int]] = []

    for start, _end, text in segments:
        ts = f"{int(start // 60):02d}:{int(start % 60):02d}"
        print(f"  [{ts}] {model}...")

        user_msg = f"Traduce del {SOURCE_LANG} al {TARGET_LANG}:\n{text}"
        kwargs: dict = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
        }
        if not model.startswith("o"):
            kwargs["temperature"] = 0.3

        response = await client.chat.completions.create(**kwargs)
        translated = response.choices[0].message.content.strip()
        usage = response.usage
        results.append((translated, usage.prompt_tokens, usage.completion_tokens))

    return results


# --- Reporte ---

def _ts(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


def _cost(prompt_tokens: int, completion_tokens: int, model: str) -> float:
    p = PRICING[model]
    return (prompt_tokens * p["input"] + completion_tokens * p["output"]) / 1_000_000


def generate_report(
    segments: list[tuple[float, float, str]],
    all_results: dict[str, list[tuple[str, int, int]]],
    output_path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# Benchmark: Comparacion de Modelos de Traduccion\n")
    lines.append(f"**VOD:** `{VOD_URL}`\n")
    lines.append(f"**Muestra:** Primeros {MAX_MINUTES} minutos\n")
    lines.append(f"**Contexto dado:** {CONTEXT}\n")

    # Tabla comparativa
    lines.append("## Tabla Comparativa\n")
    header = "| Timestamp | " + " | ".join(MODELS) + " |"
    sep = "|---" * (len(MODELS) + 1) + "|"
    lines.append(header)
    lines.append(sep)

    for i, (start, _end, _ja) in enumerate(segments):
        ts = _ts(start)
        cols = [ts]
        for model in MODELS:
            results = all_results[model]
            cols.append(results[i][0] if i < len(results) else "")
        lines.append("| " + " | ".join(cols) + " |")

    # Costos
    lines.append("\n## Costos Reales (muestra de 5 min)\n")
    lines.append("| Modelo | Prompt Tokens | Completion Tokens | Costo (5 min) | Costo est. (1 hora) | Total con Whisper (1 hora) |")
    lines.append("|---|---|---|---|---|---|")

    for model in MODELS:
        results = all_results[model]
        tp = sum(r[1] for r in results)
        tc = sum(r[2] for r in results)
        c = _cost(tp, tc, model)
        lines.append(
            f"| {model} | {tp:,} | {tc:,} "
            f"| ${c:.4f} | ${c * 12:.4f} | ${c * 12 + 0.36:.4f} |"
        )

    lines.append("\n**Costo Whisper:** ~$0.36/hora (fijo)")

    # Captions originales
    lines.append("\n## Texto Fuente (YT captions japones)\n")
    lines.append("| Timestamp | Texto |")
    lines.append("|---|---|")
    for start, _end, text in segments:
        lines.append(f"| {_ts(start)} | {text} |")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReporte generado: {output_path}")


# --- Main ---

async def main() -> None:
    output_path = _PROJECT_ROOT / "benchmark_models.md"

    print("=" * 60)
    print("Extrayendo captions de YouTube")
    print("=" * 60)
    segments = extract_youtube_captions(VOD_URL)
    print(f"  -> {len(segments)} segmentos\n")

    client = AsyncOpenAI()
    all_results: dict[str, list[tuple[str, int, int]]] = {}

    for model in MODELS:
        print("=" * 60)
        print(f"Traduciendo con {model}")
        print("=" * 60)
        all_results[model] = await translate_batch(client, segments, model)
        print()

    print("=" * 60)
    print("Generando reporte")
    print("=" * 60)
    generate_report(segments, all_results, output_path)
    print("\nBenchmark completado!")


if __name__ == "__main__":
    asyncio.run(main())
