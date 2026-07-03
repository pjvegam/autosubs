# Decisiones arquitectónicas

Registro de decisiones tomadas con justificación. Sirve para cuando alguien (o nosotros mismos en 6 meses) pregunte "¿por qué hicimos X?"

---

## 2026-04-24: Cache de SRT incremental con append

**Contexto**: El cache anterior guardaba SRTs incompletos con el mismo nombre que los completos. Si una sesión se interrumpía, el cache quedaba parcial pero se reusaba como "completo".

**Decisión**: SRT cache es append-only y se calcula `translate_from = max(cache_end, start_time)`.

**Por qué**: Si hay 3600s cacheados y el usuario pide `--start-time 7200`, traducir solo desde 7200 (no rellenar el gap, no re-traducir).

**Trade-off**: Si la sesión interrumpe a la mitad de un chunk, ese chunk se pierde (no es atómico).

**Cuándo reconsiderar**: Si descubrimos que los gaps molestan al usuario, agregar opción de fill.

---

## 2026-04-24: faster-whisper-turbo como default local

**Contexto**: Comparamos turbo, large-v3 y kotoba para uso local en GPU.

**Decisión**: turbo es el default. large-v3 y kotoba descartados.

**Por qué**:
- turbo: 4x más rápido que large-v3 con MENOS alucinaciones (28 vs 62)
- kotoba: out-of-domain para streamer audio (29% cobertura, errores semánticos)
- turbo tiene problema conocido: 12-22% outros en gameplay silencioso, pero es el menor mal de los locales

**Cuándo reconsiderar**:
- Si Cohere Transcribe se valida en chunked mode → cambiar a Cohere
- Si sale whisper-v4-turbo o algo equivalente

---

## 2026-04-24: Gemini 2.5 Pro NO reemplaza gpt-4o-transcribe en producción aún

**Contexto**: Benchmark mostró Gemini Pro mejor calidad y 4x más barato que gpt-4o-transcribe.

**Decisión**: NO migrar todavía.

**Por qué**:
- gpt-4o-transcribe está validado en producción (sesiones reales del usuario)
- Gemini Pro solo testeado en modo batch 20min, no chunked 5-8s
- No sabemos cómo Gemini Pro responde a chunks individuales pequeños
- API key Gemini ya configurada → fácil agregar como opción

**Cuándo reconsiderar**: Después de un benchmark chunked o un test live de 1h con Gemini Pro.

---

## 2026-04-24: Claude Haiku 4.5 como next-best traductor (vs gpt-4.1)

**Contexto**: Benchmark de traductores mostró Haiku superior en calidad (entiende speaker, sin errores) y más barato.

**Decisión**: Recomendado para migración (pendiente implementación en pipeline).

**Por qué**:
- Único modelo que detectó primera persona en streamer JA
- Sin errores graves observados
- 30% más barato ($0.25 vs $0.36 por 2h)
- Velocidad similar a gpt-4.1 (1.15s vs 0.75s/entry, ambos viable real-time)

**Cuándo reconsiderar**: Si nuevo Claude Sonnet/Opus baja precio significativamente.

---

## 2026-04-24: OpenRouter como agregador de traductores

**Contexto**: Querés probar DeepSeek, Kimi, Claude Haiku sin abrir 3 cuentas.

**Decisión**: Usar OpenRouter como único provider.

**Por qué**:
- 0% markup en inferencia (passthrough)
- Privacidad: no loggea por default, evita providers chinos directos
- 1 API key para todos los modelos
- Stripe USA para pago (no exponer CC a chinos)

**Trade-offs**:
- 20% IVA en Chile al cargar (irreversible)
- Rate limits de free tier providers (DeepInfra para DeepSeek V4 Flash)
- 1 API key = 1 punto de falla

**Cuándo reconsiderar**: Si necesitás throughput muy alto, BYOK directo da mejores rate limits.

---

## 2026-04-24: PyTorch nightly cu128 para Blackwell

**Contexto**: RTX 5060 (Blackwell sm_120) no funciona con PyTorch stable.

**Decisión**: Instalar `torch==2.12.0.dev+cu128` desde nightly index.

**Por qué**: stable cu124 no tiene kernels para sm_120. Hasta que llegue a stable.

**Riesgo**: Nightly puede romper en cualquier release. Si lo hace, downgrade a versión nightly previa.

**Cuándo reconsiderar**: Cuando PyTorch 2.7+ stable soporte sm_120 oficialmente.

---

## 2026-04-24: NO usar el prompt de Whisper (initial_prompt)

**Contexto**: Probamos pasar transcripciones previas como `prompt` a Whisper para mantener consistencia (técnica de Whisper-Streaming paper).

**Decisión**: Revertido. NO usamos el prompt feature.

**Por qué**: 
- Loop de retroalimentación: si Whisper alucina "でも", el prompt en el siguiente chunk lo refuerza, generando "でも でも でも..."
- Solo útil con LocalAgreement-2 (texto confirmado), no con cualquier output
- Para faster-whisper local, `initial_prompt` es aún más agresivo en la influencia

**Cuándo reconsiderar**: Si implementamos LocalAgreement-2 (latencia ~2x chunks → no viable real-time).

---

## 2026-04-24: Filtro de alucinaciones — caracteres repetitivos

**Contexto**: Whisper alucina patrones tipo `えーーーーーーーーー...` en audio silencioso/ruido.

**Decisión**: Agregado regex `(.)\1{9,}` en `_is_hallucination()`.

**Por qué**: Repeticiones de 10+ veces del mismo char nunca ocurren en habla natural.

**No agregamos blacklist BoH (Bag of Hallucinations) todavía**: pendiente cuando se confirme que turbo sigue siendo el local default. Si migramos a Cohere o Gemini, BoH se vuelve menos urgente.

---

## 2026-04-25: Migración a Setup A en producción

**Contexto**: Benchmarks confirmaron Gemini 2.5 Pro + Claude Haiku 4.5 como mejor calidad/precio. E2E es-only validado pero más caro en batch.

**Decisión**: Setup A (Pro ASR + Haiku translation) implementado como default cuando las API keys están disponibles.

**Cambios implementados**:
1. `transcribe.py` agregó `transcribe_gemini_chunk()` — audio inline para chunks 5-8s
2. `pipeline.py` agregó `_is_gemini_asr()` y `_create_translate_client()` para routing
3. `_run_mpv_pipeline`, `_run_batch_vod_pipeline`, `_run_system_audio_pipeline` separan `asr_client` (OpenAI) de `translate_client` (OpenRouter o OpenAI según modelo)
4. `config.py` defaults inteligentes:
   - ASR: Gemini Pro si `GEMINI_API_KEY`, sino local-turbo, sino gpt-4o-transcribe
   - Traducción: Haiku via OpenRouter si `OPENROUTER_API_KEY`, sino gpt-4.1

**Detección automática**: modelos con `/` en el nombre (ej: `anthropic/claude-haiku-4.5`) routean a OpenRouter. Modelos sin `/` (ej: `gpt-4.1`) routean a OpenAI directo.

**Override manual**:
- `--whisper-model gpt-4o-transcribe` para forzar OpenAI ASR
- `--gpt-model gpt-4.1` para forzar OpenAI translator
- `AUTOSUBS_WHISPER_MODEL` y `AUTOSUBS_GPT_MODEL` env vars

**Cuándo reconsiderar**:
- Si Gemini sube precio significativamente
- Si sale traductor mejor que Haiku para JA→ES
- Si Cohere se valida en chunked y queremos local default

---

## Decisiones pendientes

### Cohere chunked validation
- Aún no testeado en pipeline real con VAD chunks
- Bloquea decisión "Cohere reemplaza turbo como local default"
