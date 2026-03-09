# livesubs

Subtitulos traducidos en tiempo real para videos y streams de YouTube.

Transcribe el audio con Whisper y lo traduce con GPT. Los subtitulos aparecen directamente en MPV mientras ves el video.

## Que necesitas

- [Python 3.13+](https://www.python.org/downloads/)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (gestor de paquetes)
- [ffmpeg](https://ffmpeg.org/download.html) (en PATH)
- [mpv](https://mpv.io/installation/) (en PATH)
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) (en PATH)
- Una API key de [OpenAI](https://platform.openai.com/api-keys)

## Instalacion

```bash
git clone <repo>
cd livesubs
uv sync
```

Crea un archivo `.env` en la carpeta del proyecto:

```
OPENAI_API_KEY=sk-tu-api-key-aqui
```

## Uso

```bash
# Asi de simple
uv run livesubs "https://www.youtube.com/watch?v=..."
```

Por default traduce de japones a espanol. Para cambiar idiomas edita el `.env`:

```
OPENAI_API_KEY=sk-...
LIVESUBS_SOURCE_LANG=en
LIVESUBS_TARGET_LANG=es
```

O usa flags:

```bash
uv run livesubs "https://www.youtube.com/watch?v=..." --source-lang en --target-lang es
```

Tambien puedes traducir el audio del sistema (lo que suena en la PC):

```bash
# Traduce cualquier audio que salga de la computadora (juegos, Discord, otros sitios, etc.)
uv run livesubs --system-audio

# Con idiomas especificos
uv run livesubs --system-audio --source-lang en --target-lang es
```

## Diarizacion (experimental)

Identifica quien habla y etiqueta los subtitulos por speaker. Requiere GPU NVIDIA y Python 3.12 (NeMo no soporta 3.13 aun):

```bash
# Instalar dependencias de diarizacion
uv sync --python 3.12 --extra diarize

# Usar con cualquier modo
uv run --python 3.12 --extra diarize livesubs "https://www.youtube.com/watch?v=..." --diarize

# Con overlay
uv run --python 3.12 --extra diarize livesubs "https://www.youtube.com/watch?v=..." --subs-only --diarize

# Con audio del sistema
uv run --python 3.12 --extra diarize livesubs --system-audio --diarize
```

## Opciones

Todo se puede configurar en `.env` o con flags (los flags tienen prioridad):

### Configuracion general

| `.env` | Flag | Default | Que hace |
|---|---|---|---|
| `OPENAI_API_KEY` | - | (requerido) | API key de OpenAI |
| `LIVESUBS_SOURCE_LANG` | `--source-lang` | `ja` | Idioma del audio |
| `LIVESUBS_TARGET_LANG` | `--target-lang` | `es` | Idioma de los subtitulos |
| `LIVESUBS_WHISPER_MODEL` | `--whisper-model` | `gpt-4o-transcribe` | Modelo de transcripcion |
| `LIVESUBS_GPT_MODEL` | `--gpt-model` | `gpt-4.1` | Modelo de traduccion |
| `LIVESUBS_COOKIES_BROWSER` | `--cookies-browser` | `firefox` | Browser para cookies de yt-dlp. `none` para desactivar |

### Modos de operacion

| Flag | Que hace |
|---|---|
| `--system-audio` | Captura audio del sistema (loopback WASAPI, solo Windows) |
| `--batch-vod` | Pre-procesa VOD completo con whisper-1 + verbose_json (chunks de 20min). Mejor timestamps pero peor calidad en habla casual. Ver [Benchmark](#benchmark) |
| `--benchmark` | Compara gpt-4o-transcribe vs whisper-1 sobre 20min del mismo audio. Genera un .md en `livesubs_logs/`. Ver [Benchmark](#benchmark) |
| `--diarize` | Activar diarizacion de speakers (requiere `livesubs[diarize]`) |

### Ajustes de audio y traduccion

| `.env` | Flag | Default | Que hace |
|---|---|---|---|
| `LIVESUBS_SEGMENT_DURATION` | `--segment-duration` | `5` | Duracion objetivo de cada chunk de audio (segundos) |
| `LIVESUBS_VAD_AGGRESSIVENESS` | `--vad-aggressiveness` | `2` | Agresividad del VAD 0-3. Mas alto = mas agresivo filtrando silencio |
| - | `--start-time` | `0` | Tiempo de inicio en segundos (override sobre `?t=` en URL) |
| - | `--max-minutes` | `0` | Limite de minutos a traducir (0 = sin limite) |
| - | `--no-cache` | - | Forzar re-traduccion aunque exista SRT en cache |
| - | `--speaker-gender` | - | Genero del hablante principal (`f`/`m`) para consistencia gramatical |
| - | `--speakers` | `0` | Numero de speakers en el video (da contexto a GPT) |
| - | `--verbose` | - | Muestra logs de debug |

## Benchmark

El modo `--benchmark` compara los dos pipelines de transcripcion sobre el mismo audio:

- **Path A** (default): `gpt-4o-transcribe` + VAD ~5s — transcribe muchos chunks cortos
- **Path B** (batch-vod): `whisper-1` + `verbose_json` — transcribe 20min de golpe

```bash
# Extrae 20min desde el minuto 5 (default) y compara ambos pipelines
uv run livesubs "https://www.youtube.com/watch?v=..." --benchmark

# Empezar desde otro punto
uv run livesubs "https://www.youtube.com/watch?v=..." --benchmark --start-time 600
```

Genera un `.md` en `livesubs_logs/` con metricas (costo, alucinaciones, segmentos) y muestras lado a lado cada 2 minutos.

**Resultado del primer benchmark** (conversacion casual japonesa, 20min):

| Metrica | gpt-4o-transcribe | whisper-1 |
|---|---|---|
| Subtitulos | 191 | 481 |
| Alucinaciones | 0 | 3 |
| Segmentos cortos | 3 | 31 |
| Costo total | $0.18 | $0.25 |

gpt-4o-transcribe captura frases completas con contexto mientras whisper-1 fragmenta en trozos cortos. El pipeline default (Path A) es mas barato y de mejor calidad para habla casual. `--batch-vod` podria ser util para contenido mas estructurado (conferencias, noticias) donde los timestamps precisos por frase importan.

Costo estimado de un benchmark: ~$0.40-0.50 por 20 minutos de audio.
