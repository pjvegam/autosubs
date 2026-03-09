"""Benchmark: YouTube captions vs Whisper + modelos de traduccion.

Compara calidad y costo de:
- Whisper + gpt-4o-mini (baseline, de test.md)
- YouTube auto-captions + gpt-4o-mini
- YouTube auto-captions + o3-mini
"""

import asyncio
import functools
import json
import re
import subprocess
import sys
from pathlib import Path

# Forzar flush en cada print para ver progreso en tiempo real
print = functools.partial(print, flush=True)  # type: ignore[assignment]

from dotenv import load_dotenv
from openai import AsyncOpenAI

# Agregar src/ al path para importar livesubs
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
    "Canal: Sakuna Ch. 結城さくな | "
    "Titulo: Revisando el nuevo pollo tori tatsuta de Mos Burger"
)

# Precios por millon de tokens (USD)
PRICING: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "o3-mini": {"input": 1.10, "output": 4.40},
}


# --- Paso 1: Extraer captions de YouTube ---

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

    # Buscar el archivo json3 generado
    json3_files = list(tmp_dir.glob("*.json3"))
    if not json3_files:
        raise FileNotFoundError("No se encontro archivo json3 de captions")

    json3_path = json3_files[0]
    with json3_path.open(encoding="utf-8") as f:
        data = json.load(f)

    # Limpiar archivos temporales
    for f in tmp_dir.iterdir():
        f.unlink()
    tmp_dir.rmdir()

    # Parsear eventos json3
    raw_segments: list[tuple[float, float, str]] = []
    for event in data.get("events", []):
        start_ms = event.get("tStartMs", 0)
        duration_ms = event.get("dDurationMs", 0)
        segs = event.get("segs")
        if not segs:
            continue
        text = "".join(s.get("utf8", "") for s in segs).strip()
        text = text.replace("\n", " ")
        if not text:
            continue
        start_s = start_ms / 1000.0
        end_s = (start_ms + duration_ms) / 1000.0
        raw_segments.append((start_s, end_s, text))

    # Filtrar primeros 5 minutos
    max_seconds = MAX_MINUTES * 60
    raw_segments = [(s, e, t) for s, e, t in raw_segments if s < max_seconds]

    # Agrupar en bloques de ~5s
    return _group_segments(raw_segments, group_duration=5.0)


def _group_segments(
    segments: list[tuple[float, float, str]],
    group_duration: float = 5.0,
) -> list[tuple[float, float, str]]:
    """Agrupa segmentos cortos en bloques de ~group_duration segundos."""
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

    # Ultimo grupo
    grouped.append((current_start, current_end, " ".join(current_texts)))
    return grouped


# --- Paso 2: Parsear test.md ---

def parse_test_md(path: Path, max_minutes: int = 5) -> list[tuple[str, str]]:
    """Extrae pares (timestamp, texto) de las lineas 'Subtitulo [MM:SS]: texto' del test.md."""
    pattern = re.compile(r"Subtitulo \[(\d{2}:\d{2})\]: (.+)")
    results: list[tuple[str, str]] = []

    with path.open(encoding="utf-8") as f:
        for line in f:
            if m := pattern.search(line):
                ts = m.group(1)
                text = m.group(2)
                minutes = int(ts.split(":")[0])
                if minutes < max_minutes:
                    results.append((ts, text))

    return results


# --- Paso 3: Traducir con modelos ---

async def translate_batch(
    client: AsyncOpenAI,
    segments: list[tuple[float, float, str]],
    model: str,
) -> list[tuple[str, int, int]]:
    """Traduce una lista de segmentos y retorna (traduccion, prompt_tokens, completion_tokens)."""
    system_prompt = _build_system_prompt(CONTEXT)
    results: list[tuple[str, int, int]] = []

    for start, end, text in segments:
        ts = f"{int(start // 60):02d}:{int(start % 60):02d}"
        print(f"  [{ts}] Traduciendo con {model}...")

        user_msg = f"Traduce del {SOURCE_LANG} al {TARGET_LANG}:\n{text}"
        kwargs: dict = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
        }
        # o3-mini es modelo de razonamiento, no soporta temperature
        if not model.startswith("o3"):
            kwargs["temperature"] = 0.3
        response = await client.chat.completions.create(**kwargs)

        translated = response.choices[0].message.content.strip()
        usage = response.usage
        results.append((translated, usage.prompt_tokens, usage.completion_tokens))

    return results


# --- Paso 4: Generar reporte ---

def _format_timestamp(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


def _calc_cost(prompt_tokens: int, completion_tokens: int, model: str) -> float:
    prices = PRICING[model]
    return (prompt_tokens * prices["input"] + completion_tokens * prices["output"]) / 1_000_000


def generate_report(
    whisper_results: list[tuple[str, str]],
    yt_captions: list[tuple[float, float, str]],
    yt_mini_results: list[tuple[str, int, int]],
    yt_o3_results: list[tuple[str, int, int]],
    output_path: Path,
) -> None:
    """Genera benchmark_results.md con tabla comparativa y costos."""
    lines: list[str] = []
    lines.append("# Benchmark: YouTube Captions vs Whisper + Modelos de Traduccion\n")
    lines.append(f"**VOD:** `{VOD_URL}`\n")
    lines.append(f"**Muestra:** Primeros {MAX_MINUTES} minutos\n")

    # Tabla comparativa
    lines.append("## Tabla Comparativa\n")
    lines.append(
        "| Timestamp | Whisper + gpt-4o-mini | YT Captions + gpt-4o-mini | YT Captions + o3-mini |"
    )
    lines.append("|---|---|---|---|")

    # Crear mapa de timestamps de whisper
    whisper_map: dict[str, str] = {ts: text for ts, text in whisper_results}

    # Iterar por los segmentos de YT (son la referencia temporal)
    for i, (start, _end, _ja_text) in enumerate(yt_captions):
        ts = _format_timestamp(start)

        # Buscar whisper mas cercano
        whisper_text = whisper_map.get(ts, "")
        if not whisper_text:
            # Buscar timestamp cercano (+/- 4s)
            start_int = int(start)
            for offset in range(-4, 5):
                candidate = f"{(start_int + offset) // 60:02d}:{(start_int + offset) % 60:02d}"
                if candidate in whisper_map:
                    whisper_text = whisper_map[candidate]
                    break

        yt_mini_text = yt_mini_results[i][0] if i < len(yt_mini_results) else ""
        yt_o3_text = yt_o3_results[i][0] if i < len(yt_o3_results) else ""

        lines.append(f"| {ts} | {whisper_text} | {yt_mini_text} | {yt_o3_text} |")

    # Costos
    lines.append("\n## Costos Reales (muestra de 5 min)\n")

    total_mini_prompt = sum(r[1] for r in yt_mini_results)
    total_mini_completion = sum(r[2] for r in yt_mini_results)
    total_o3_prompt = sum(r[1] for r in yt_o3_results)
    total_o3_completion = sum(r[2] for r in yt_o3_results)

    cost_mini = _calc_cost(total_mini_prompt, total_mini_completion, "gpt-4o-mini")
    cost_o3 = _calc_cost(total_o3_prompt, total_o3_completion, "o3-mini")

    lines.append("| Combinacion | Prompt Tokens | Completion Tokens | Costo (5 min) | Costo est. (1 hora) |")
    lines.append("|---|---|---|---|---|")
    lines.append(
        f"| YT Captions + gpt-4o-mini | {total_mini_prompt:,} | {total_mini_completion:,} "
        f"| ${cost_mini:.4f} | ${cost_mini * 12:.4f} |"
    )
    lines.append(
        f"| YT Captions + o3-mini | {total_o3_prompt:,} | {total_o3_completion:,} "
        f"| ${cost_o3:.4f} | ${cost_o3 * 12:.4f} |"
    )

    # Nota sobre Whisper
    lines.append(
        f"\n**Nota:** Whisper + gpt-4o-mini no reporta tokens de traduccion en test.md. "
        f"Costo estimado de Whisper: ~$0.006/min = $0.03 por 5 min, $0.36/hora."
    )

    # Captions originales de YouTube (para referencia)
    lines.append("\n## Captions Originales de YouTube (japones)\n")
    lines.append("| Timestamp | Texto Original |")
    lines.append("|---|---|")
    for start, _end, text in yt_captions:
        ts = _format_timestamp(start)
        lines.append(f"| {ts} | {text} |")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReporte generado: {output_path}")


# --- Main ---

async def main() -> None:
    project_root = _PROJECT_ROOT
    test_md_path = project_root / "test.md"
    output_path = project_root / "benchmark_results.md"

    # Paso 1: Extraer captions de YouTube
    print("=" * 60)
    print("PASO 1: Extrayendo captions de YouTube")
    print("=" * 60)
    yt_captions = extract_youtube_captions(VOD_URL)
    print(f"  -> {len(yt_captions)} segmentos agrupados (~5s c/u)")

    # Paso 2: Parsear test.md
    print("\n" + "=" * 60)
    print("PASO 2: Parseando test.md (Whisper + gpt-4o-mini)")
    print("=" * 60)
    whisper_results = parse_test_md(test_md_path, MAX_MINUTES)
    print(f"  -> {len(whisper_results)} subtitulos encontrados")

    # Paso 3: Traducir con ambos modelos
    client = AsyncOpenAI()

    print("\n" + "=" * 60)
    print("PASO 3a: Traduciendo YT captions con gpt-4o-mini")
    print("=" * 60)
    yt_mini_results = await translate_batch(client, yt_captions, "gpt-4o-mini")

    print("\n" + "=" * 60)
    print("PASO 3b: Traduciendo YT captions con o3-mini")
    print("=" * 60)
    yt_o3_results = await translate_batch(client, yt_captions, "o3-mini")

    # Paso 4: Generar reporte
    print("\n" + "=" * 60)
    print("PASO 4: Generando reporte")
    print("=" * 60)
    generate_report(whisper_results, yt_captions, yt_mini_results, yt_o3_results, output_path)

    print("\nBenchmark completado!")


if __name__ == "__main__":
    asyncio.run(main())
