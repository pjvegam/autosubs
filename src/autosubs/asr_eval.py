"""Runners de transcripcion compartidos por los scripts de benchmark.

Un "item" es (id, ruta_wav). Cada runner transcribe todos los items con un
motor y devuelve las hipotesis indexadas por id.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from openai import AsyncOpenAI

from autosubs.transcribe import transcribe_audio, transcribe_gemini_chunk

logger = logging.getLogger(__name__)

API_CONCURRENCY = 8
GEMINI_CONCURRENCY = 4  # Gemini tira 429 antes que el resto
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
_SRC_DIR = str(Path(__file__).resolve().parent.parent)

# engine -> tamano de modelo faster-whisper. El resto se asume API remota.
LOCAL_ENGINES = {"local-turbo": "turbo", "local-v3": "large-v3"}

# Aparecieron en OpenRouter despues del benchmark de julio
NUEVOS_2026_10 = [
    "microsoft/mai-transcribe-2",
    "elevenlabs/scribe-v2",
    "qwen/qwen3-asr-flash-2026-02-10",
    "qwen/qwen3-asr-1.7b",
    "qwen/qwen3-asr-0.6b",
    "x-ai/grok-stt-1.0",
    "google/gemini-3.5-transcribe",
    "assemblyai/universal-3-5-pro",
    "openai/gpt-transcribe",
    "nvidia/nemotron-3.5-asr-streaming-multilingual-0.6b",
]

KNOWN_ENGINES = [
    *LOCAL_ENGINES,
    "gpt-4o-transcribe",
    "microsoft/mai-transcribe-1.5",
    *NUEVOS_2026_10,
    "deepgram/nova-3",
    "mistralai/voxtral-mini-transcribe",
    "openai/whisper-1",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
]


@dataclass
class EngineReport:
    """Hipotesis y metricas de rendimiento de un motor."""

    engine: str
    hypotheses: dict[str, str] = field(default_factory=dict)
    elapsed: float = 0.0

    @property
    def empty(self) -> int:
        return sum(1 for t in self.hypotheses.values() if not t)


def is_openrouter(engine: str) -> bool:
    """Los slugs con barra (microsoft/mai-transcribe-1.5) van por OpenRouter."""
    return "/" in engine


def client_for(engine: str) -> AsyncOpenAI:
    """Construye el cliente segun el proveedor del motor."""
    if not is_openrouter(engine):
        return AsyncOpenAI()
    if not (key := os.environ.get("OPENROUTER_API_KEY")):
        raise RuntimeError(f"{engine} necesita OPENROUTER_API_KEY en .env")
    return AsyncOpenAI(base_url=OPENROUTER_BASE, api_key=key)


async def run_api_engine(
    engine: str,
    items: list[tuple[str, Path]],
    language: str,
    on_progress: callable | None = None,
) -> EngineReport:
    """Transcribe cada item contra una API remota, en paralelo acotado."""
    client = client_for(engine)
    # OpenRouter rechaza response_format="text"
    response_format = "json" if is_openrouter(engine) else "text"
    sem = asyncio.Semaphore(API_CONCURRENCY)
    done = 0

    async def _one(item_id: str, path: Path) -> tuple[str, str]:
        nonlocal done
        async with sem:
            text = await transcribe_audio(
                client, path, language, model=engine, response_format=response_format,
            )
            done += 1
            if on_progress and done % 25 == 0:
                on_progress(done, len(items))
            return item_id, text or ""

    t0 = time.perf_counter()
    results = await asyncio.gather(*(_one(i, p) for i, p in items))
    return EngineReport(
        engine=engine,
        hypotheses=dict(results),
        elapsed=time.perf_counter() - t0,
    )


async def run_gemini_engine(
    engine: str,
    items: list[tuple[str, Path]],
    language: str,
    on_progress: callable | None = None,
) -> EngineReport:
    """Transcribe con Gemini multimodal (audio inline)."""
    sem = asyncio.Semaphore(GEMINI_CONCURRENCY)
    done = 0

    async def _one(item_id: str, path: Path) -> tuple[str, str]:
        nonlocal done
        async with sem:
            text = await transcribe_gemini_chunk(path, language, model=engine)
            done += 1
            if on_progress and done % 25 == 0:
                on_progress(done, len(items))
            return item_id, text or ""

    t0 = time.perf_counter()
    results = await asyncio.gather(*(_one(i, p) for i, p in items))
    return EngineReport(
        engine=engine,
        hypotheses=dict(results),
        elapsed=time.perf_counter() - t0,
    )


async def run_local_engine(
    engine: str,
    items: list[tuple[str, Path]],
    language: str,
    on_progress: callable | None = None,
) -> EngineReport:
    """Transcribe con faster-whisper cargando el modelo una sola vez.

    Corre en subprocess: ctranslate2 aborta el proceso al liberar CUDA en
    Windows, y os._exit(0) evita ese cleanup.
    """
    # La lista va por archivo: incrustada en `python -c` supera el limite de
    # 32767 caracteres de la linea de comandos de Windows (WinError 206).
    fd, items_file = tempfile.mkstemp(suffix=".json", prefix="asr_items_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump([[i, str(p)] for i, p in items], f)

    script = f"""
import json, os, sys, time
sys.path.insert(0, {_SRC_DIR!r})
from pathlib import Path
from autosubs.local_whisper import LocalWhisperModel

with open({items_file!r}, encoding="utf-8") as f:
    items = json.load(f)
model = LocalWhisperModel(model_size={LOCAL_ENGINES[engine]!r})

out = {{}}
t0 = time.perf_counter()
for i, (item_id, path) in enumerate(items, 1):
    segments = model.transcribe(Path(path), {language!r})
    out[item_id] = " ".join(s.text for s in segments).strip()
    if i % 25 == 0:
        print(f"  {{i}}/{{len(items)}}", file=sys.stderr, flush=True)
elapsed = time.perf_counter() - t0

# A bytes: con la consola en cp1252 el japones reventaria al escribirlo
payload = json.dumps({{"elapsed": elapsed, "hypotheses": out}}, ensure_ascii=False)
sys.stdout.buffer.write(payload.encode("utf-8"))
sys.stdout.buffer.flush()
os._exit(0)
"""
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", script,
            stdout=asyncio.subprocess.PIPE,
            stderr=None,  # progreso del hijo directo a la consola
        )
        stdout, _ = await proc.communicate()
    finally:
        Path(items_file).unlink(missing_ok=True)

    if proc.returncode != 0:
        raise RuntimeError(f"{engine}: subprocess fallo (exit {proc.returncode})")

    data = json.loads(stdout.decode("utf-8"))
    return EngineReport(
        engine=engine,
        hypotheses=data["hypotheses"],
        elapsed=data["elapsed"],
    )


async def run_engine(
    engine: str,
    items: list[tuple[str, Path]],
    language: str,
    on_progress: callable | None = None,
) -> EngineReport:
    """Despacha al runner que corresponde segun el motor."""
    if engine in LOCAL_ENGINES:
        runner = run_local_engine
    elif engine.startswith("gemini-"):
        runner = run_gemini_engine
    else:
        runner = run_api_engine
    return await runner(engine, items, language, on_progress)
