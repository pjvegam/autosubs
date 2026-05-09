# LiveSubs - Documentación de Benchmarks

Histórico de pruebas de modelos para el pipeline JA → ES (transcripción + traducción).

## Indice

| Documento | Fecha | Descripción |
|-----------|-------|-------------|
| [transcription-benchmark-2026-04.md](transcription-benchmark-2026-04.md) | 2026-04-24 | 6 modelos de ASR comparados |
| [translation-benchmark-2026-04.md](translation-benchmark-2026-04.md) | 2026-04-24 | 4 traductores LLM comparados |
| [e2e-benchmark-2026-04.md](e2e-benchmark-2026-04.md) | 2026-04-24 | Gemini 2.5 Pro audio→ES single call |
| [models-rejected.md](models-rejected.md) | varios | Modelos descartados con razones |
| [pricing-reference-2026-04.md](pricing-reference-2026-04.md) | 2026-04 | Snapshot de pricing al momento |
| [decisions.md](decisions.md) | live | Decisiones arquitectónicas |
| [hardware.md](hardware.md) | live | Notas de hardware/setup |
| [future-tests.md](future-tests.md) | live | Plan de tests cuando salgan nuevos modelos |

## Resumen ejecutivo (estado actual abril 2026)

### Componentes recomendados
- **Transcripción**: gpt-4o-transcribe (API, $0.12/20min) o Gemini 2.5 Pro (API, $0.03/20min)
- **Traducción**: Claude Haiku 4.5 via OpenRouter ($0.25/2h) — mejor calidad
- **Local backup**: Cohere Transcribe 03-2026 (validar en chunked mode aún)

### Setup ganador (validado)
**Setup A: Gemini 2.5 Pro + Claude Haiku 4.5** = ~$0.43/2h = **210 CLP/hora**

### Setup B validado (con sorpresas)
**Gemini 2.5 Pro E2E** = $0.92/2h batch (más caro de lo esperado por output JA+ES doble).
- ✅ Calidad confirmada (detecta 1ra persona, gender, localización)
- ✅ Latencia 2.3x mejor que Setup A en batch
- ❌ **2x más caro en batch** que Setup A
- 🎯 Para **real-time live** (chunks 5s): ~150 CLP/hora estimado, sigue siendo mejor que Setup A
- 🔧 Optimización pendiente: remover `ja` del schema → reduce 50% costo

### Lo que NO usar
- ❌ Gemini 2.5 Flash para ASR (76% alucinaciones)
- ❌ kotoba-whisper-v2.0 (out-of-domain para streamer audio)
- ❌ DeepSeek V4 Flash via OpenRouter (rate limits compartidos)
- ❌ DeepSeek como traductor (confunde "ジム"/"Jim")
- ❌ Xiaomi MiMo, Parakeet (no soportan japonés)

## Cómo usar esta documentación

Cuando salga un modelo nuevo:
1. Lee [future-tests.md](future-tests.md) para ver el plan de testeo
2. Corré el benchmark relevante (transcripción o traducción)
3. Crea un nuevo archivo `*-benchmark-YYYY-MM.md` con los resultados
4. Compará contra los datos históricos en este folder
5. Actualizá [decisions.md](decisions.md) si cambia la recomendación
