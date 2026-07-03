# Hardware y setup

## Especificaciones del usuario

- **GPU**: NVIDIA RTX 5060 Blackwell (sm_120)
- **VRAM**: 8GB
- **CPU**: x64
- **OS**: Windows 11
- **Python**: 3.13.5 (managed by uv)

## Software stack

### CUDA / PyTorch

⚠️ **PyTorch stable NO funciona con Blackwell sm_120 al momento.**

Necesitas:
```bash
uv pip install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

Versiones confirmadas funcionando:
- `torch==2.12.0.dev20260408+cu128`
- `torchaudio==2.11.0.dev20260407+cu128`

Verificar:
```python
import torch
print(torch.cuda.is_available())          # True
print(torch.cuda.get_device_name(0))      # "NVIDIA GeForce RTX 5060"
print(torch.cuda.get_device_capability(0)) # (12, 0)
```

### faster-whisper / CTranslate2

faster-whisper (basado en CTranslate2) funciona en sm_120 con float16. INT8 deshabilitado en CTranslate2 para sm_120.

⚠️ **CTranslate2 aborta el proceso al destruir modelos CUDA en Windows**. Solución: correr en subprocess y usar `os._exit(0)` para evitar la limpieza de Python que triggerea el abort.

Esto está implementado en `_transcribe_local_subprocess()` de `benchmark.py`.

### CUDA DLLs preload

En Windows, las DLLs de CUDA (cublas) requieren pre-carga via `ctypes.WinDLL` desde `site-packages/nvidia/`. Implementado en `local_whisper.py:_preload_cuda_dlls()`.

## Modelos en disco

Cache de modelos de HuggingFace en `~/.cache/huggingface/hub/`:

| Modelo | Tamaño | Notas |
|--------|--------|-------|
| faster-whisper turbo | ~800MB | Activo, default local |
| faster-whisper large-v3 | ~1.5GB | Para benchmark |
| ~~kotoba-whisper-v2.0-faster~~ | ~~1.5GB~~ | **Borrado** (descartado) |
| Cohere Transcribe 03-2026 | ~4GB | Si se valida, mantener |

## Audio cache

Cache de segmentos de benchmark en `autosubs_logs/`:

- `segment_{video_id}_{start_time}.mp3` — audio de 20min para benchmarks
- `benchmark_cache_{video_id}_{start_time}.json` — resultados de transcripción
- `benchmark_translators_cache_{video_id}_{start_time}.json` — resultados de traducción

Estos NO se borran entre benchmarks. Reusar cuando sea posible.

## Dependencias del proyecto

Optional dependencies en `pyproject.toml`:

```toml
[project.optional-dependencies]
diarize = ["nemo_toolkit[asr]>=2.0", "torch>=2.0", "torchaudio>=2.0"]
local-whisper = ["faster-whisper>=1.1.0", "nvidia-cublas-cu12>=12.4"]
gemini = ["google-genai>=1.0"]
cohere-asr = ["transformers>=4.46", "librosa>=0.10", "accelerate>=0.30"]
```

Instalación completa:
```bash
uv sync --extra local-whisper --extra gemini --extra cohere-asr
```

⚠️ **Si `uv sync` instala torch CPU**: reinstalar con `--index-url` apuntando a cu128 nightly.

## Variables de entorno (`.env`)

```
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
OPENROUTER_API_KEY=sk-or-v1-...
HF_TOKEN=hf_...
AUTOSUBS_WHISPER_MODEL=local-turbo  # opcional
AUTOSUBS_SOURCE_LANG=ja              # opcional
AUTOSUBS_TARGET_LANG=es              # opcional
```

## Issues conocidos del entorno

### `autosubs.exe` se queda bloqueada

Cuando una sesión de PowerShell ejecuta `uv run autosubs`, queda un proceso `autosubs.exe` en memoria. La próxima `uv sync` falla:

```
error: failed to remove file ... autosubs.exe: El proceso no tiene acceso al archivo
```

**Workaround**: cerrar la terminal vieja, abrir nueva. O bypass uv:

```bash
.venv/Scripts/python -m autosubs.main "URL" --benchmark ...
```

### `ご視聴ありがとうございました` en local-turbo

faster-whisper-turbo hallucina esta frase de cierre de YouTube en silencios/audio ambiguo. En sesiones reales con gameplay silencioso: 12-22% de los outputs son esta basura.

**Mitigación parcial**: filtro de repetitivos detecta `(.)\1{9,}` (ej: えーーーーー).
**Mitigación completa pendiente**: blacklist BoH-style.

### `cp1252` UnicodeEncodeError en Windows subprocess

Cuando un subprocess imprime JSON con caracteres japoneses, falla en Windows porque stdout default es cp1252.

**Workaround**: usar `json.dumps(data)` SIN `ensure_ascii=False`. El JSON escapa a `\uXXXX` y el padre lo decodifica fine.

Implementado en `_transcribe_cohere_subprocess()`.
