# AutoSubs

> Subtítulos traducidos en tiempo real para YouTube. Multi-provider ASR + LLM translation con fallback automático, circuit breaker, cache incremental y recuperación de transcripciones perdidas.

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Tests 213 passing](https://img.shields.io/badge/tests-213%20passing-brightgreen.svg)](#tests)
[![License MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Style: ruff](https://img.shields.io/badge/style-ruff-orange.svg)](https://github.com/astral-sh/ruff)

## ¿Qué hace?

Mientras miras un stream de YouTube en MPV, **AutoSubs transcribe el audio, lo traduce, y te muestra subtítulos en tu idioma** — todo en vivo. Pensado para consumir contenido de creadores en otros idiomas (en mi caso, streamers japoneses) sin esperar a que alguien los traduzca.

```
┌──────────┐    ┌──────┐    ┌─────┐    ┌────────────┐    ┌─────┐    ┌─────┐
│ YouTube  │───▶│yt-dlp│───▶│ VAD │───▶│ ASR (chunk)│───▶│ LLM │───▶│ SRT │──▶ MPV
│ stream   │    │  +   │    │     │    │            │    │     │    │     │
└──────────┘    │ffmpeg│    │WebRTC│   │ Gemini Pro │    │Claude│   │live │
                └──────┘    └─────┘    │ o gpt-4o-* │    │Haiku │   │write│
                                       │ o local    │    │o GPT │   └─────┘
                                       └────────────┘    └─────┘
```

## Por qué importa (lo que muestro acá)

Este proyecto **no es un wrapper de Whisper**. Es un sistema que aprendí a hacer **robusto, barato y mantenible** después de chocar con problemas reales:

- 🔁 **Multi-provider con fallback automático**: cuando Gemini Pro tira `503 high demand`, el pipeline cambia transparentemente a `gpt-4o-transcribe` sin perder chunks
- 🚦 **Circuit breaker**: 5 fallos consecutivos = saca a Gemini de juego por 5min (evita ~21s de retries por chunk)
- 💾 **Cache incremental de SRT**: si interrumpes, la próxima corrida continúa exacto donde quedó
- 🔧 **Recuperación de gaps** (`--fill-gaps`): si Gemini perdió chunks durante un outage, re-transcribe los gaps del video local con gpt-4o (estable)
- 💰 **Cost engineering**: ~$0.18-$0.43 por stream de 2h según setup. Documentado con [benchmarks reales](docs/)
- 📊 **Decisiones documentadas**: cada elección de modelo está justificada con [datos](docs/transcription-benchmark-2026-04.md), no marketing
- ⚙️ **Backends intercambiables**: 6 ASR (Gemini Pro/Flash, gpt-4o-transcribe, faster-whisper turbo/large-v3, Cohere Transcribe local) y 4+ LLM translators via OpenRouter

## Requisitos del sistema

Además de Python 3.13+ y [uv](https://docs.astral.sh/uv/), necesitas estos binarios nativos instalados (no vienen por pip):

| Binario | Para qué | Obligatorio |
|---------|----------|-------------|
| **ffmpeg** | Procesamiento de audio (chunking + EQ de transcripción) | Sí |
| **mpv** | Reproductor donde se muestran los subtítulos | Sí |
| **deno** | yt-dlp lo usa para resolver los JS challenges de YouTube | Recomendado |

`yt-dlp` se instala solo como dependencia de Python con `uv sync` — no requiere instalación aparte.

```bash
# Windows (winget)  —  yt-dlp.FFmpeg trae los builds de ffmpeg parcheados para yt-dlp
winget install yt-dlp.FFmpeg shinchiro.mpv DenoLand.Deno

# macOS (Homebrew)
brew install ffmpeg mpv deno

# Linux (Debian/Ubuntu)
sudo apt install ffmpeg mpv
curl -fsSL https://deno.land/install.sh | sh   # deno
```

> **Mantené yt-dlp actualizado**: YouTube cambia seguido y rompe la extracción; tener yt-dlp al día es vital. Si algo deja de funcionar, actualizá con `uv sync --upgrade-package yt-dlp`.

## Quickstart

```bash
git clone https://github.com/pjvegam/autosubs.git
cd autosubs
uv sync --extra gemini  # opcional: añade soporte Gemini

# Crear .env con tu API key
echo "OPENAI_API_KEY=sk-..." > .env

# Mirar un stream japonés con subs en español (default)
uv run autosubs "https://www.youtube.com/watch?v=..."
```

## Setup recomendado (mejor calidad/precio)

```env
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...                                  # Setup A: ASR
OPENROUTER_API_KEY=sk-or-v1-...                     # Setup A: traducción
AUTOSUBS_WHISPER_MODEL=gemini-2.5-pro
AUTOSUBS_GPT_MODEL=anthropic/claude-haiku-4.5
AUTOSUBS_SOURCE_LANG=ja
AUTOSUBS_TARGET_LANG=es
```

**Costo estimado**: ~$0.43 por stream de 2h (con fallback a gpt-4o cuando Gemini falla).

Ver [docs/pricing-reference-2026-04.md](docs/pricing-reference-2026-04.md) para alternativas.

## Modos de uso

### Stream en vivo / VOD de YouTube
```bash
uv run autosubs "https://www.youtube.com/watch?v=XXXXXX"
```

### Empezar desde un punto específico
```bash
uv run autosubs "URL" --start-time 1h30m     # también acepta 5400, 1:30:00, 90m
```

### Limitar duración a traducir
```bash
uv run autosubs "URL" --max-minutes 30        # o 30m, 0:30:00
```

### Audio del sistema (cualquier audio que salga de tu PC)
```bash
uv run autosubs --system-audio                # juegos, Discord, otros sitios
```

### Recuperar transcripciones perdidas
Si Gemini tuvo outage durante una sesión y perdiste chunks:
```bash
uv run autosubs "URL" --fill-gaps             # usa video local + gpt-4o-transcribe
```

### Forzar provider específico
```bash
uv run autosubs "URL" --whisper-model gpt-4o-transcribe        # ASR sólido garantizado
uv run autosubs "URL" --gpt-model deepseek/deepseek-chat       # traductor barato
```

## Stack técnico

| Capa | Tecnología | Por qué |
|------|-----------|---------|
| Async runtime | `asyncio` + `TaskGroup` | I/O-bound pipeline con multiple workers |
| ASR APIs | OpenAI, Google AI (Gemini), HuggingFace | Multi-provider con [benchmarks](docs/transcription-benchmark-2026-04.md) |
| ASR local | `faster-whisper` + CTranslate2 (CUDA) | Backup gratis cuando no quieres gastar API |
| LLM translator | OpenAI, Anthropic, OpenRouter (DeepSeek/Kimi/Gemini) | [Comparados](docs/translation-benchmark-2026-04.md) |
| Audio I/O | `ffmpeg` (subprocess), WebRTC VAD | Chunking inteligente por silencio |
| Player | MPV via IPC (JSON-RPC sobre named pipe) | Subtítulos live update |
| Stream | `yt-dlp` (subprocess) | Auth con cookies del browser |
| Tests | `pytest` + `pytest-asyncio` (213 tests) | Mocks de APIs externas |
| Linting | `ruff` (`E`, `W`, `F`, `I`, `N`, `UP`, `B`, `SIM`, `ASYNC`, `S`, `RUF`) | Strict mode |

## Decisiones arquitectónicas destacadas

(detalle completo en [`docs/decisions.md`](docs/decisions.md))

| Decisión | Por qué |
|---------|---------|
| Setup A (Gemini Pro + Claude Haiku) como default | Mejor calidad-precio en benchmark vs gpt-4o + gpt-4.1 |
| Cache de SRT incremental con `append=True` | Si interrumpes, continúas exacto donde quedó. Sin SRTs incompletos pretendiendo ser completos |
| Routing por nombre de modelo (`/` → OpenRouter) | 1 cliente OpenAI-compatible cubre 4+ providers sin SDKs separados |
| Circuit breaker sobre fallback simple | Outages persistentes saturaban retries; ahora skipea Gemini por 5min después de 5 fails |
| Subprocess isolation para ctranslate2 | CUDA cleanup en Windows abortaba el proceso. `os._exit(0)` + subprocess lo evita |

## Benchmarks

Resultados de testing real, no marketing:

- **[Transcripción](docs/transcription-benchmark-2026-04.md)** — 6 modelos comparados (Gemini Pro/Flash, gpt-4o, faster-whisper turbo/large-v3, Cohere). Spot-checks por timestamp + métricas de cobertura, alucinaciones, costo.
- **[Traducción](docs/translation-benchmark-2026-04.md)** — 4 LLMs (gpt-4.1, DeepSeek V3, Kimi K2, Claude Haiku 4.5). Detección de errores culturales (ej: DeepSeek confunde `ジム`/gym con "Jim") y consistencia de speaker.
- **[E2E](docs/e2e-benchmark-2026-04.md)** — Gemini 2.5 Pro audio→ES en single call vs pipeline modular.
- **[Modelos rechazados](docs/models-rejected.md)** — qué probé y por qué no quedó (kotoba-whisper out-of-domain, MiMo/Parakeet sin japonés, etc.)

## Limitaciones conocidas del benchmark

El benchmark de transcripción usa Claude como evaluador **sin ground truth**, es decir, sin una transcripción manual verificada como referencia. Esto significa que lo que se está midiendo es **coherencia y fluidez del output**, no accuracy absoluta.

**Lo que sí detecta:**

- Alucinaciones evidentes (texto inventado)
- Cambios de idioma en modelos multilingües
- Errores estructurales obvios (nombres propios en nonsense, timestamps rotos)

**Lo que NO detecta:**

- Palabras mal transcritas que suenan plausibles en el contexto
- Contenido que el modelo silenció sin decir "no entendí"
- Errores en números, fechas o nombres propios ambiguos

El benchmark de traducción es más confiable porque compara bilateralmente fuente (transcripción) y target (traducción), lo que sí permite detectar inconsistencias semánticas.

**Mejora pendiente para v2**: usar como referencia un video con transcripción oficial verificada (charla académica, keynote con subtítulos manuales) y comparar los outputs de cada backend con métrica objetiva (Word Error Rate).

## Cost engineering

| Setup | $/2h stream | CLP/hora | Calidad |
|-------|-------------|----------|---------|
| Local (turbo) + gpt-4.1 | $0.36 | 180 | Media (22% outros en gameplay silencioso) |
| **Setup A** (Gemini Pro + Claude Haiku) ⭐ | $0.43 | 215 | Alta confirmada |
| gpt-4o-transcribe + Claude Haiku | $0.97 | 485 | Alta |
| Premium (gpt-4o-transcribe + gpt-4.1) | $1.08 | 540 | Alta |

Para 60h streaming/mes (~2h/día): **$13/mes en Setup A**.

## Configuración completa

Ver `.env.example` y output de `uv run autosubs --help` para todas las opciones (~25 flags).

Variables principales:

| Variable | Default | Descripción |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (requerida) | Para fallback ASR y traducciones gpt-* |
| `GEMINI_API_KEY` | (opcional) | Activa Setup A ASR |
| `OPENROUTER_API_KEY` | (opcional) | Activa Setup A traducción (Claude Haiku, DeepSeek, Kimi, etc.) |
| `HF_TOKEN` | (opcional) | Para Cohere Transcribe (modelo gated) |
| `AUTOSUBS_WHISPER_MODEL` | auto-detect | `gemini-2.5-pro`, `gpt-4o-transcribe`, `local-turbo`, `local-v3` |
| `AUTOSUBS_GPT_MODEL` | auto-detect | `gpt-4.1`, `anthropic/claude-haiku-4.5`, `deepseek/deepseek-chat`, etc. |

## Hardware

Validado en:
- **RTX 5060 Blackwell sm_120** (8GB VRAM) — requiere PyTorch nightly cu128 (ver [`docs/hardware.md`](docs/hardware.md))
- Windows 11 + WSL no testeado pero debería funcionar

## Tests

```bash
uv run pytest tests/ -q                  # 213 tests passing
uv run ruff check src/                   # 0 issues
```

## Estructura del repositorio

```
autosubs/
├── src/autosubs/
│   ├── pipeline.py        # orquestación async (TaskGroup workers)
│   ├── transcribe.py      # ASR routing: OpenAI / Gemini / faster-whisper
│   ├── translate.py       # LLM translation con context window
│   ├── recover.py         # --fill-gaps: re-transcribe gaps del SRT
│   ├── benchmark.py       # ASR + translator + E2E benchmarks
│   ├── stream.py          # yt-dlp wrapper, parse_time_value
│   ├── audio.py           # ffmpeg + VAD chunking
│   ├── subtitles.py       # SRT cache append mode
│   ├── player.py          # MPV IPC (named pipe, JSON-RPC)
│   ├── costs.py           # tracking + pricing por modelo
│   └── config.py          # pydantic-like Config con defaults inteligentes
├── tests/                 # 213 tests, mocks de APIs externas
└── docs/                  # benchmarks, decisiones, pricing, hardware
```

## Roadmap

Cosas que tengo en backlog pero todavía no resuelvan los datos como problema:

- [ ] Validar Cohere Transcribe en modo chunked (5-8s) — pendiente porque benchmark batch fue prometedor pero no comparable a producción
- [ ] Soporte de prompt caching de Anthropic para reducir ~10% costo de traducción
- [ ] Streaming partial transcription (Gemini soporta SSE en algunos modelos)
- [ ] Dockerfile para reproducibilidad (con/sin GPU)

## Licencia

MIT.
