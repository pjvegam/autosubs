# Benchmark de Transcripción — Abril 2026

## Setup del test

- **Audio**: Stream de YouTube `Ndlo93lI0xI`, segmento 60:00-80:00 (20 min)
- **Tipo**: Chica japonesa hablando sola sobre fitness/gym (audio limpio, sin gameplay/música)
- **Idioma**: Japonés
- **Hardware**: RTX 5060 Blackwell sm_120, 8GB VRAM
- **Pipeline**: archivo MP3 batch al modelo, pre-cacheado en `autosubs_logs/segment_Ndlo93lI0xI_3600.mp3`

## Resultados

| Modelo | Entries | Chars | Halluc | Vacíos | Tiempo | RTF | Costo |
|--------|---------|-------|--------|--------|--------|-----|-------|
| **gpt-4o-transcribe** | 191 | 2914 | 2 | 5 | 67.8s | 0.06x | $0.12 |
| **Gemini 2.5 Pro** ⭐ | 148 | 3472 | 4 | 5 | 81.0s | 0.07x | $0.03 |
| Cohere Transcribe 03-2026 | 83¹ | 2944 | 1 | 1 | 19.5s | 0.02x | $0 |
| faster-whisper-turbo | 605 | 4031 | 28 | 37 | 44.1s | 0.04x | $0 |
| faster-whisper-large-v3 | 400 | 3478 | 62 | 61 | 192.4s | 0.16x | $0 |
| Gemini 2.5 Flash | 851 | 4721 | **649** ❌ | **656** | 156.4s | 0.13x | $0.03 |

¹ Cohere requirió splitting manual por oraciones (no soporta `return_timestamps` en transformers pipeline)

## Análisis cualitativo

### Spot-check ~00:00 — "えーっと、体脂肪率どこだ?"

| Modelo | Output | Calidad |
|--------|--------|---------|
| turbo | えっと、体脂肪率どこだ? | ✅ |
| large-v3 | えーっと | ❌ cortado |
| **gpt-4o** | えーっと、体脂肪率どこだ? | ✅ mejor |
| cohere | えっと体脂肪率どこだ? | ✅ |
| flash | えっと 体脂 脂 率 どこ だ | ❌ fragmentado raro |
| **pro** | えっと、体脂肪率どこだ? | ✅ |

### Spot-check ~10:00 — Habla de 腹筋 (abdominales)

```
gpt-4o:  "これジムで撮ったデータじゃないかな。"
pro:     "あの、こう日常生活を生きてきて腹筋っていう言葉を目にし
         なさすぎて読めなかった。すぐに言葉が変換できなかった。
         腹筋ね。腹筋にえ、血管が浮くようになりました。"
```

**Pro captura 4 oraciones completas** vs gpt-4o un fragmento. Diferencia clave: **+18% chars transcritos** que gpt-4o.

## Hallazgos clave

### 🥇 Gemini 2.5 Pro es el ganador (sorpresa)

- Más contenido que gpt-4o (3472 vs 2914 chars, +18%)
- Similar nivel de alucinaciones (4 vs 2)
- **4x más barato que gpt-4o** ($0.03 vs $0.12 por 20min)
- Oraciones más completas y naturales
- Native multimodal con structured output (JSON)

### ❌ Gemini 2.5 Flash es BASURA para ASR

**649 de 851 entries (76%) son alucinaciones.** Cuando se le pide JSON estructurado con timestamps, Flash inventa segmentos. **No usar para transcripción.**

### ⚠️ Cohere Transcribe — datos incompletos

- Splitting manual por oraciones inflactuó las "83 entries" (no es comportamiento real del modelo)
- Velocidad excelente: 19.5s para 20min batch (más rápido que cualquiera)
- Solo 1 alucinación
- **Pendiente**: validar en modo chunked (5-8s chunks como hace VAD en producción)

### faster-whisper-turbo (default actual)

- Velocidad excelente, pero 28 alucinaciones en este audio limpio
- Fragmentación agresiva (605 entries para 20min)
- En audio "difícil" (gameplay con silencio) sube a 12-22% de alucinaciones outro tipo "ご視聴ありがとうございました"

### faster-whisper-large-v3

- Peor que turbo en todo: más lento (4x), más alucinaciones (2x), mismo costo
- Descartado en favor de turbo

## Verdaderos finalistas

Para audio **limpio** (un hablante, sin ruido):
1. **Gemini 2.5 Pro** — mejor calidad-precio, $0.03/20min
2. **gpt-4o-transcribe** — más caro ($0.12) pero estable y probado
3. **Cohere** — único viable open-weight (validar en chunked)

Para audio **con ruido/silencios** (gameplay):
- **gpt-4o-transcribe** ya validado con 0 alucinaciones outro
- Cohere y Gemini Pro **requieren benchmark adicional** en este caso
- turbo confirmado problemático en este caso

## Notas técnicas

### Cohere Transcribe — limitaciones
- HuggingFace pipeline no soporta `return_timestamps=True` para non-CTC, non-Whisper models
- Requiere `language="ja"` al **processor**, NO a `generate()`
- Se rompe el subprocess si stdout no está en utf-8 (no usar `ensure_ascii=False`)
- Modelo es "gated" en HuggingFace — requiere `HF_TOKEN` con read access

### Gemini Pro — uso óptimo
- Usar Files API (no inline) para audio >20MB
- Pedir JSON estructurado con `response_schema` para timestamps
- `response_mime_type="application/json"` para evitar markdown wrap

### Hardware Blackwell sm_120
- PyTorch stable (cu124) NO funciona con Blackwell
- Necesita PyTorch nightly + CUDA 12.8
- `torch==2.12.0.dev+cu128` confirmado funcionando

## Replicación

Para correr este mismo benchmark:

```powershell
uv run autosubs "https://www.youtube.com/watch?v=Ndlo93lI0xI" --benchmark --start-time 1h --benchmark-no-translate
```

El segmento de 20min ya está cacheado. Modelos requieren API keys: `OPENAI_API_KEY`, `GEMINI_API_KEY`, `HF_TOKEN`.
