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

## Opciones

Todo se puede configurar en `.env` o con flags (los flags tienen prioridad):

| `.env` | Flag | Default | Que hace |
|---|---|---|---|
| `OPENAI_API_KEY` | - | (requerido) | API key de OpenAI |
| `LIVESUBS_SOURCE_LANG` | `--source-lang` | `ja` | Idioma del audio |
| `LIVESUBS_TARGET_LANG` | `--target-lang` | `es` | Idioma de los subtitulos |
| `LIVESUBS_SEGMENT_DURATION` | `--segment-duration` | `5` | Segundos por segmento |
| `LIVESUBS_WHISPER_MODEL` | `--whisper-model` | `gpt-4o-transcribe` | Modelo de transcripcion |
| `LIVESUBS_GPT_MODEL` | `--gpt-model` | `gpt-4o-mini` | Modelo de traduccion |
| - | `--subs-only` | - | Solo subtitulos en ventana overlay (sin MPV) |
| - | `--system-audio` | - | Captura audio del sistema (loopback WASAPI, solo Windows) |
| - | `--verbose` | - | Muestra logs de debug |
