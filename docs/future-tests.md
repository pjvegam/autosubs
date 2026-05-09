# Plan de tests futuros

Cuando salga un modelo nuevo o querramos re-evaluar, estos son los tests que valen la pena.

## Cómo probar un nuevo TRANSCRIPTOR

### Comando
```powershell
uv run livesubs "https://www.youtube.com/watch?v=Ndlo93lI0xI" --benchmark --start-time 1h --benchmark-no-translate
```

(reusa el segmento de 20min cacheado, ahorra costo)

### Pasos
1. Agregar el nuevo modelo al `_TRANSLATORS_TO_TEST` en `benchmark.py`
2. Implementar wrapper si necesario (ej: nuevo SDK, nuevo provider)
3. Correr el benchmark
4. Analizar el `benchmark_*.md` generado
5. Crear nuevo doc `transcription-benchmark-YYYY-MM.md` con los resultados

### Métricas que importan
- **Cobertura de chars** (vs gpt-4o-transcribe = 2914 como referencia)
- **Alucinaciones** (especialmente outros: "ご視聴ありがとうございました")
- **Velocidad** RTF (real-time factor) — debe ser < 0.1x para uso live
- **Costo per minuto de audio**
- **Spot-checks** en momentos clave (00:00, 02:00, 10:00) comparando con gpt-4o

### Test adicional para audio "difícil"
Cuando un modelo gana en audio limpio, validar también con:
```powershell
uv run livesubs "URL_RESIDENT_EVIL_GAMEPLAY" --benchmark --start-time XXX --benchmark-no-translate
```

Audio con silencios largos / gameplay / música. Acá es donde turbo falla con outros.

## Cómo probar un nuevo TRADUCTOR

### Comando
```powershell
uv run livesubs "https://www.youtube.com/watch?v=Ndlo93lI0xI" --benchmark-translators --start-time 1h
```

(reusa transcripción cacheada de gpt-4o-transcribe, solo costo de traducción)

### Pasos
1. Agregar al `_TRANSLATORS_TO_TEST` en `benchmark.py`:
   ```python
   ("provider/model-id", "openrouter"),  # o "openai" si es directo
   ```
2. Si nuevo provider, agregar a `_PROVIDER_CONFIG` con `base_url` y `api_key_env`
3. Agregar pricing a `costs.py` `PRICING` dict
4. Correr — el smoke test detecta si funciona antes de las 191 entries
5. Crear nuevo doc `translation-benchmark-YYYY-MM.md`

### Métricas que importan
- **Errores culturales** (¿confunde katakana con nombres? ej Jim/gym)
- **Detección de speaker** (¿usa primera persona cuando corresponde?)
- **Naturalidad** (vs traducciones literales raras)
- **Velocidad** s/entry (debe ser < 2s para live)
- **Costo per 2h stream**

### Tests cualitativos clave (spot-checks)
Mantener estos timestamps como tests fijos:
- ~00:00 "えーっと、体脂肪率どこだ?" — basics + filler "えーっと"
- ~05:59 "腹筋っていう言葉にゆかりがなさすぎて。" — meta-referencia a palabra
- ~07:56 "その割に体重は普通なのよ。だけど筋肉なさすぎ。" — TEST DE SPEAKER (1ra persona femenina)
- ~10:03 "これジムで撮ったデータじゃないかな。" — TEST CULTURAL (gym vs Jim)
- ~14:03 "青春の輝き。" — modismo

## Cómo validar un setup E2E (audio → traducción directa)

### Comando (TODO: implementar)
```powershell
uv run livesubs "URL" --benchmark-e2e --start-time 1h
```

### Plan
- Pasar audio a Gemini Pro / GPT-4o-audio
- Pedir JSON con segments {start, end, ja, es}
- Comparar contra Setup A (Pro ASR + Haiku translation) que tenemos confirmado
- Métricas: tiempo total, tokens, calidad subjetiva

### Modelos candidatos para E2E
- Gemini 2.5 Pro (current candidate)
- Gemini 3.x Pro (cuando salga)
- GPT-4o-audio-preview (closed source, $$)
- Cualquier nuevo modelo audio→text multimodal

## Modelos que esperamos / monitorear

### En camino
- **Gemini 3.5 / 4** (futuro): probable mejora marginal
- **Claude Sonnet 5 / Opus 5**: si bajan precio, considerar para live
- **DeepSeek V5**: si arreglan tema cultural katakana → ganador potencial
- **OpenAI GPT-5 Audio Preview**: cuando salga publicly
- **Mistral Voxtral 2.0**: actualmente solo europeas, esperar JA support
- **Qwen3-ASR multilingual**: mencionado en research como bueno para JA, no testeado aún

### Modelos especulativos a vigilar
- **Cohere Transcribe v2**: si sale, probablemente mejor
- **NVIDIA Canary multilingual**: siempre que agreguen JA
- **kotoba-whisper-v3**: si entrenan en data más diversa que TV

## Cuándo re-correr todo

Eventos que merecen re-evaluación completa:
1. Bajada de precio >50% en algún provider
2. Nuevo modelo top-3 en HuggingFace ASR Leaderboard con JA support
3. Cambio de hardware (más VRAM permite modelos más grandes)
4. Cambio en tu uso (más streamers, otros idiomas, etc.)

## Templates

### Template de doc para nuevo benchmark de transcripción

```markdown
# Benchmark de Transcripción — {YYYY-MM}

## Setup del test
- **Audio**: ...
- **Tipo**: ...
- **Hardware**: ...

## Resultados
| Modelo | Entries | Chars | Halluc | Tiempo | RTF | Costo |
| ... |

## Análisis cualitativo
### Spot-check ~00:00
| Modelo | Output | Calidad |
| ... |

## Hallazgos clave
...

## Recomendación
...
```

### Template para nuevo benchmark de traducción

```markdown
# Benchmark de Traducción — {YYYY-MM}

## Setup del test
- **Source**: ...
- **Pair**: ja → es

## Resultados
| Modelo | Tiempo | s/entry | $/2h | CLP/hora |
| ... |

## Análisis cualitativo
### Errores culturales detectados
...

### Detección de speaker
...

## Recomendación
...
```
