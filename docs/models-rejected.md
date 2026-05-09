# Modelos descartados

Documentación de modelos que probamos o investigamos y por qué NO los recomendamos.

## kotoba-whisper-v2.0 (descartado 2026-04)

### Qué prometía
Fine-tune de whisper-large-v3 entrenado en ReazonSpeech (TV japonés). Benchmark CER de 17.0 en ReazonSpeech (mejor que whisper-large-v3 con 20.5).

### Por qué falló
**Out-of-domain para audio de streamers.**

Test con video `Ndlo93lI0xI` (chica habla sola fitness, 20 min):
- **Cubrió solo 29% del audio** (se detuvo a 5:52 de 20min)
- 230 chars total vs gpt-4o 2914 chars
- Errores semánticos graves:
  - `体脂肪率` (% grasa corporal) → transcribió `大死亡率` (tasa mortalidad)
  - "de 18% a 8%" → "llegó al 18%" (invierte sentido)
  - Tokens duplicados corruptos: `腹筋読キン読めない`

### Lección
Fine-tuning en TV audio (ReazonSpeech) NO transfiere a streamer audio.
Benchmark del paper original ya lo indicaba:
- CommonVoice 8: kotoba 17.7 vs whisper-v3 15.3 (kotoba peor)
- JSUT Basic: kotoba 15.4 vs whisper-v3 13.4 (kotoba peor)
- Solo gana en ReazonSpeech (in-domain)

### Reconsiderar si
Sale un kotoba-whisper-v3 entrenado en audio general (no solo TV).

---

## Gemini 2.5 Flash para ASR (descartado 2026-04)

### Qué prometía
Audio multimodal nativo, free tier disponible, $0.03/20min en paid. Económico vs gpt-4o-transcribe.

### Por qué falló
**76% de las entries son alucinaciones cuando se le pide JSON estructurado.**

Test con video `Ndlo93lI0xI`:
- 851 entries totales
- **649 alucinaciones** (76%)
- 656 segmentos vacíos/cortos

Cuando le pedimos JSON con timestamps + texto, Flash inventa segmentos. Posible causa: el modelo es demasiado pequeño/rápido para ASR estructurado preciso.

### Reconsiderar si
- Sale Gemini 3.x Flash con mejor capacidad audio
- Encontramos un prompt que Flash entiende mejor
- Para tareas que NO requieren JSON estructurado (raw transcription text)

### Sí útil para
**Traducción de texto** — donde no necesita audio understanding. Pero ese benchmark queda pendiente.

---

## DeepSeek V4 Flash via OpenRouter (descartado 2026-04)

### Qué prometía
$0.14/$0.28 per M tokens. **20x más barato que gpt-4.1**. 22 CLP/hora estimado.

### Por qué falló
**Rate limits upstream del provider DeepInfra.**

OpenRouter sirve DeepSeek V4 Flash via DeepInfra en free credits. DeepInfra tiene rate limits compartidos por todos los usuarios. En el test:
- Funcionó por las primeras 30 entries (~4 min)
- Después: error 429 "deepseek/deepseek-v4-flash is temporarily rate-limited upstream"
- Errores persistentes hasta abandonar

Además:
- **8s/entry** vs gpt-4.1 0.8s/entry (10x más lento)
- No viable para real-time

### Reconsiderar si
- DeepSeek API directa (BYOK via OpenRouter)
- DeepInfra escala su capacidad para este modelo
- Para batch processing (no live)

---

## DeepSeek V3 como traductor (no recomendado para live)

### Qué prometía
$0.32/$0.89 per M tokens. ~$0.05/2h = 25 CLP/hora. **Más barato que cualquier otro**.

### Problema
**Errores culturales graves.** Confunde palabras inglesas en katakana con nombres propios:
- `ジム` (gym) → traducido como "Jim" (nombre persona)

### Cuándo SÍ usar
Post-batch de contenido técnico/no-coloquial. Donde podés revisar antes de publicar.

### Por qué NO usar para live
- Errores no recuperables en stream en vivo
- 2s/entry latencia = lentísimo
- Cultura de katakana (préstamos del inglés) muy común en streamer JA

---

## Kimi K2 como traductor (no destacable)

### Qué prometía
"Más natural/idiomático para contenido asiático".

### Por qué no destacó
- 1.52s/entry (medio en velocidad)
- $0.11/2h (no el más barato)
- Calidad media: pierde detalle de "la palabra X" → traduce solo "X"
- Sin ventaja única vs Claude Haiku 4.5

### Reconsiderar si
- Sale Kimi K3 con mejoras significativas
- Para casos específicos donde naturalidad asiática matters

---

## Xiaomi MiMo-V2.5-ASR (no investigado seriamente)

### Por qué descartado de entrada
- **NO soporta japonés** (solo chino mandarín + dialectos + inglés)
- 8B params — no entra en 8GB VRAM (necesita 16GB+)
- Custom framework (no faster-whisper compatible)

---

## NVIDIA Parakeet TDT 0.6B v2/v3 (no aplicable)

### Por qué descartado
- v2: solo inglés
- v3: 25 idiomas europeos, **sigue sin japonés**

Útil para inglés. No para nuestro caso.

---

## faster-whisper-large-v3 (descartado vs turbo)

### Por qué turbo > large-v3
- Turbo: 605 entries, 28 alucinaciones, 44s
- Large-v3: 400 entries, **62 alucinaciones**, 192s

Large-v3 es 4x más lento Y más alucinador que turbo. Sin ventaja.

### Reconsiderar
Nunca. Turbo es estrictamente mejor.
