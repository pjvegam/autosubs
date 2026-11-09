# AutoSubs

> Subtítulos traducidos en tiempo real para YouTube. Mirás un stream japonés en MPV y ves los subtítulos en español mientras pasa.

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/pjvegam/autosubs/actions/workflows/ci.yml/badge.svg)](https://github.com/pjvegam/autosubs/actions/workflows/ci.yml)
[![License MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Style: ruff](https://img.shields.io/badge/style-ruff-orange.svg)](https://github.com/astral-sh/ruff)

## Instalar

```bash
winget install yt-dlp.FFmpeg shinchiro.mpv DenoLand.Deno   # macOS: brew install ffmpeg mpv deno
uv sync
echo "OPENROUTER_API_KEY=sk-or-..." > .env
```

## Comandos

```bash
# Lo normal
uv run autosubs "https://youtu.be/VIDEO_ID"

# Desde un punto
uv run autosubs "URL" --start-time 45:00

# Un rango: del 45:00 al 55:00, MPV también para ahí
uv run autosubs "URL" --start-time 45:00 --end-time 55:00

# Transcribir en la GPU en vez de pagar API
uv run autosubs "URL" --whisper-model local-turbo

# Audio de cualquier cosa que suene en el PC
uv run autosubs --system-audio
```

Acepta `45:00`, `1:30:00`, `90m`, `5400`. Si cortás a medias, al volver sigue desde donde quedó el SRT.

| Flag | Para qué |
|---|---|
| `--start-time` / `--end-time` | Acotar el tramo a traducir |
| `--whisper-model` | `local-turbo` (GPU, gratis) o un modelo de API |
| `--gpt-model` | Traductor (default `anthropic/claude-haiku-5.5`) |
| `--system-audio` | Capturar el audio del PC en vez de una URL |
| `--no-cache` | Re-traducir aunque exista SRT |

## Qué modelo usar

Medido sobre [TEDxJP-10K](https://github.com/laboroai/TEDxJP-10K) (343 utterances con transcripción humana verificada) y sobre 20 min de gameplay real. **Basura** = subtítulos inventados: despedidas de YouTube que nunca se dijeron, o texto en otro idioma.

| Motor | CER | Basura | $/min de audio |
|---|---|---|---|
| **microsoft/mai-transcribe-2** | 7.3% | 0.8% | 0.0018 |
| microsoft/mai-transcribe-1.5 | 7.2% | 0.8% | 0.0060 |
| elevenlabs/scribe-v2 | 8.4% | 5.1% | 0.0018 |
| x-ai/grok-stt-1.0 | 8.8% | 0.8% | 0.0017 |
| qwen/qwen3-asr-0.6b | 11.8% | 4.7% | 0.0002 |
| local-turbo (GPU) | 11.8% | 4.5% | 0 |

El default es **mai-transcribe-2**: la misma exactitud que 1.5 a un tercio del precio. `local-turbo` es gratis pero inventa `ご視聴ありがとうございました` ("gracias por vernos") en los silencios.

La traducción va aparte: `haiku-5.5` cuesta ~$0.00004 por línea de subtítulo.

## Config

Todo por flag o por `.env`:

```bash
OPENROUTER_API_KEY=sk-or-...      # ASR + traducción
OPENAI_API_KEY=sk-...             # opcional, fallback
AUTOSUBS_SOURCE_LANG=ja
AUTOSUBS_TARGET_LANG=es
AUTOSUBS_WHISPER_MODEL=microsoft/mai-transcribe-2
AUTOSUBS_GPT_MODEL=anthropic/claude-haiku-5.5
```

Los SRT y los videos quedan en `autosubs_logs/`.

## Desarrollo

```bash
uv run pytest -q
uv run ruff check .
uv run --extra tedx scripts/tedx_build.py --videos 10   # rearmar el dataset
uv run --extra tedx scripts/tedx_eval.py                # CER con ground truth
uv run scripts/gameplay_eval.py VIDEO_ID --start-time 2160   # alucinaciones
```

Si yt-dlp deja de funcionar, YouTube cambió algo: `uv sync --upgrade-package yt-dlp`.

## Licencia

MIT
