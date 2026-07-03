# Benchmark de Traducción — Abril 2026

## Setup del test

- **Source**: 191 entries de gpt-4o-transcribe del [transcription benchmark](transcription-benchmark-2026-04.md)
- **Pair**: ja → es
- **Cada entry**: ~5-30 caracteres japoneses
- **Total tokens**: ~22-25k input, ~2-3k output por traductor

## Resultados

| Modelo | Tiempo total | s/entry | Tokens IN | Tokens OUT | $/20min | $/2h | $/hora |
|--------|--------------|---------|-----------|------------|---------|------|--------|
| **gpt-4.1** (baseline) | 144s | 0.75 | 22,155 | 1,920 | $0.060 | $0.36 | 180 CLP |
| **Claude Haiku 4.5** ⭐ | 220s | 1.15 | 25,290 | 3,179 | $0.041 | $0.25 | 125 CLP |
| Kimi K2 | 291s | 1.52 | 22,527 | 2,662 | $0.019 | $0.11 | 55 CLP |
| **DeepSeek V3** | 374s | 1.96 | 20,469 | 2,393 | $0.009 | $0.05 | 25 CLP |

V4 Pro y V4 Flash no completaron por rate limits de DeepInfra en OpenRouter.

## Análisis cualitativo crítico

### 🚨 DeepSeek confunde "ジム" (gym) con "Jim" (nombre)

```
JA: これジムで撮ったデータじゃないかな
gpt-4.1:  ✅ "Creo que estos datos fueron tomados en el gimnasio."
DeepSeek: ❌ "Ojalá estos sean los datos que Jim recopiló."
Kimi:     ⚠️ "¿No será una foto que sacó en el gimnasio?" (foto vs datos)
Haiku:    ✅ "Esto podría ser datos tomados en el gimnasio."
```

**Error grave de contexto cultural.** Descalifica a DeepSeek para audio con préstamos del inglés/katakana.

### 🥇 Solo Claude Haiku detecta speaker en primera persona

La streamer habla de SÍ MISMA en muchos momentos. JA usa muchos sujetos elididos.

```
JA: その割に体重は普通なのよ。だけど筋肉なさすぎ。
gpt-4.1:  ❌ "Aun así, **su** peso es normal." (3era persona)
DeepSeek: ❌ "A pesar de eso, **su** peso es normal." (3era)
Kimi:     ❌ "Aun así, **pesa** lo normal." (ambiguo)
Haiku:    ✅ "A pesar de eso, **mi** peso es normal." (1era ✓)
```

Claude Haiku es el único que entendió que la chica habla de sí misma. **Esto importa para subtítulos donde el contexto del speaker es crucial.**

### Estilo natural (~00:00) — "えーっと、体脂肪率どこだ?"

```
gpt-4.1:  "A ver, ¿dónde está el porcentaje de grasa corporal?"  ✅
DeepSeek: "Umm, ¿dónde está el porcentaje de grasa corporal?"   ⚠️ comillas raras
Kimi:     "Eh... ¿dónde está el porcentaje de grasa corporal?"  ✅
Haiku:    "Eh, ¿dónde está el porcentaje de grasa corporal?"    ✅
```

## Veredicto

| Criterio | Ganador |
|----------|---------|
| **Calidad absoluta** | Claude Haiku 4.5 (entiende speaker, sin errores) |
| **Velocidad** | gpt-4.1 (0.75s/entry, único viable real-time tight) |
| **Costo absoluto** | DeepSeek V3 ($0.05/2h, 7x más barato que gpt-4.1) |
| **Errores graves** | DeepSeek (Jim/gym) |
| **Real-time viable (<2s)** | gpt-4.1, Haiku, Kimi |

## Recomendaciones

### Para pipeline en VIVO ⭐
**Claude Haiku 4.5** via OpenRouter — mejor calidad confirmada, 30% más barato que gpt-4.1, viable en real-time.

### Para post-batch de VODs
**DeepSeek V3** — 5x más barato que Haiku. El error ocasional "Jim" es tolerable en batch (donde podés revisar).

### Como fallback
**gpt-4.1** — sigue funcional, calidad alta, sin errores graves observados.

### Descartar
- **Kimi K2**: ni rápido ni mejor en calidad. Sin ventaja única.
- **DeepSeek para live**: 2s/entry + errores culturales = mal trade.

## Implementación técnica

### Configuración OpenRouter
```python
# config.py - ya implementado
"openrouter": {
    "base_url": "https://openrouter.ai/api/v1",
    "api_key_env": "OPENROUTER_API_KEY",
}
```

OpenRouter:
- 0% markup en inferencia (passthrough)
- Stripe USA para pagos (no se exponen datos a providers chinos directamente)
- No loggea por default
- 1 API key para múltiples providers

### IDs de modelos en OpenRouter
- `deepseek/deepseek-chat` (V3, más estable que V4 Flash)
- `deepseek/deepseek-v4-pro` (reasoning OFF por default)
- `moonshotai/kimi-k2`
- `anthropic/claude-haiku-4.5`

### Rate limits observados
- DeepInfra (sirve DeepSeek en OpenRouter free credits): saturado, hits 429 después de 30 entries
- BYOK (DeepSeek API key directo via OpenRouter): solución para rate limits, no testeado aún

## Replicación

```powershell
uv run autosubs "https://www.youtube.com/watch?v=Ndlo93lI0xI" --benchmark-translators --start-time 1h
```

Requiere: `OPENAI_API_KEY` y `OPENROUTER_API_KEY` en `.env`.

Reusa transcripciones cacheadas, no necesita re-descargar audio. Costo total: ~$0.13-0.20 (solo translation, varía según modelos).
