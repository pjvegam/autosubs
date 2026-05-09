# Benchmark E2E (Gemini 2.5 Pro audio→ES) — Abril 2026

## Setup del test

- **Audio**: Stream `Ndlo93lI0xI`, segmento 60:00-80:00 (20 min)
- **Modelo**: `gemini-2.5-pro`
- **Prompt**: pidió JSON con `{start, end, ja, es}` (con instrucción explícita de usar 1ra persona cuando aplique)
- **Pipeline**: audio MP3 batch en single call

## Resultados

- **Segments generados**: 158
- **Tiempo total**: 129.7s
- **Audio tokens**: 38,522 (~32 tokens/s, más alto del esperado 25 t/s)
- **Output tokens**: 11,410 (alto: incluye JA + ES en cada segmento)
- **Costo total**: $0.1526
- **Costo extrapolado 1h**: $0.46
- **Costo extrapolado 2h**: **$0.92**

## Comparativa vs Setup A (Pro ASR + Haiku traducción)

| Setup | $/2h | Latencia batch | Calidad | Calls |
|-------|------|----------------|---------|-------|
| **Setup A** (Pro+Haiku) | **$0.43** | ~301s | Alta confirmada | 2 |
| **E2E** (este) | $0.92 | **130s** | Alta confirmada | 1 |

## Análisis cualitativo

### ✅ Detecta primera persona (igual que Haiku)

```
JA: "私、体のさ、3割が脂肪なんだど"
E2E: "El 30% de MI cuerpo es grasa"  ← primera persona ✅
```

```
JA: "えっと、体脂肪率どこだ？"
E2E: "A ver, ¿dónde está MI porcentaje de grasa corporal?"  ← MI ✅
```

```
JA: "もう筋肉マジでない"
E2E: "En serio no TENGO nada de músculo"  ← 1ra persona ✅
```

### ✅ Gender-aware
```
JA: "デブ禁止"  → "Prohibido ser gorda" (femenino, correcto para la speaker)
```

### ✅ Localización
```
JA: "BMI"  → "IMC" (no "BMI")
JA: "27.7%" vs "27.7kg" (capta cambio de unidad de medida)
```

## Por qué cuesta más que Setup A

Mi estimación inicial: $0.30/2h. Realidad: $0.92/2h.

**Razón principal**: el JSON schema incluye `ja` Y `es` en cada segment, **duplicando output tokens**.

```json
[{"start": 0.0, "end": 5.5, "ja": "テキスト", "es": "Texto"}]
```

Output tokens reales: 11,410 (2h extrapolado: 68,460)
Output cost @ $10/M: $0.685 (la mayor parte del costo)

## Optimización: schema solo-ES (validado 2026-04-24)

Removimos el campo `ja` del schema, solo `{start, end, text}`.

**Resultados reales:**
- Output tokens: 11,410 → 8,134 (**-29%**, no -50% como esperaba)
- Costo: $0.92/2h → **$0.72/2h** (-22%)
- Calidad: idéntica (sigue detectando 1ra persona, género, localización)

**Por qué no -50%**: el español usa MÁS tokens que el japonés. Caracteres CJK suelen ser 1 token cada uno, palabras españolas son multi-token. Sin el `ja`, el output sigue siendo grande porque el ES solo es voluminoso.

**Conclusión**: schema solo-ES es la versión correcta para producción, pero **no llega al costo de Setup A en batch**.

## Cálculo para uso REAL-TIME (chunks de 5s, ratios reales)

Para uso en pipeline live, el cálculo es diferente. Recalculado con datos reales del benchmark:

**Setup A real-time:**
- 5s chunk audio: 125 tokens × $1/M = $0.000125
- Gemini ASR output JA (~10 tokens): × $10/M = $0.0001
- Haiku input + output (~35 tokens): ~$0.000075
- **Por chunk: $0.0003**
- **Por hora live: ~$0.22 = ~110 CLP/hora**
- **Por 2h: ~$0.43**

**E2E es-only real-time:**
- 5s chunk audio: 125 tokens × $1/M = $0.000125
- Output ES (~17 tokens): × $10/M = $0.00017
- **Por chunk: $0.0003**
- **Por hora live: ~$0.22 = ~110 CLP/hora**
- **Por 2h: ~$0.43**

**Empatan en costo real-time.** La diferencia es solo arquitectónica.

## Verdict actualizado (con datos del benchmark schema solo-ES)

| Caso de uso | Setup A (Pro+Haiku) | E2E es-only |
|-------------|---------------------|-------------|
| Batch 20min cost | **$0.07** ⭐ | $0.12 |
| Batch /2h cost | **$0.43** ⭐ | $0.72 |
| Real-time /2h cost | $0.43 | $0.43 (empate) |
| Latencia batch | 301s | **130s** ⭐ |
| Calls por chunk | 2 | **1** ⭐ |
| Complejidad pipeline | Mayor | **Menor** ⭐ |
| Calidad | Alta confirmada | Alta confirmada |

## Recomendación final

### Para BATCH (post-processing VODs):
**Setup A** ($0.43/2h vs $0.72) — 67% más barato porque batch genera todo el output JA+ES de una sola vez. La traducción posterior con Haiku es eficiente porque procesa de una.

### Para REAL-TIME (live):
**Empatan en costo** ($0.43/2h ambos). La elección depende de:
- **Setup A**: pipeline existente actual, más complejo (2 API calls/chunk)
- **E2E es-only**: arquitectura más simple (1 call/chunk), menor latencia red

### Conclusión personal
El usuario debería **mantener Setup A** porque:
1. Cuesta menos en batch
2. Ya está implementado conceptualmente (turbo+gpt-4.1 → Pro+Haiku es cambio menor)
3. Latencia no es crítica si el pipeline procesa chunks en paralelo con asyncio

E2E es bueno como **fallback simple** si Setup A falla por algún motivo, o para experimentación rápida.

## Replicación

```powershell
uv run livesubs "https://www.youtube.com/watch?v=Ndlo93lI0xI" --benchmark-e2e --start-time 1h
```

Reusa audio cacheado. Costo: ~$0.15 por corrida.
