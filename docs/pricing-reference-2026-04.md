# Pricing Reference — Abril 2026

Snapshot de precios al momento del benchmark. Los precios cambian, validar antes de re-comparar.

## Transcripción (ASR)

| Modelo | Audio input/M | Notas | $/20min audio | $/2h |
|--------|---------------|-------|---------------|------|
| OpenAI gpt-4o-transcribe | $0.36/min | Tarifa fija por minuto, $0.006 | $0.12 | $0.72 |
| OpenAI whisper-1 | $0.36/min | Igual, deprecation soon | $0.12 | $0.72 |
| Gemini 2.5 Flash | $1.00 | Free tier disponible | $0.03 | $0.18 |
| Gemini 2.5 Pro | $1.00 | Output text adicional | $0.03 | $0.18 |
| Gemini 3.1 Pro Preview | ~$1.00 | Output más caro ($12/M) | ~$0.03 | ~$0.18 |
| Gemini 3.1 Flash-Lite Preview | $0.50 | Más barato pero ASR no validado | ~$0.015 | ~$0.09 |
| faster-whisper-* (local GPU) | $0 | Free, requiere GPU | $0 | $0 |
| Cohere Transcribe (local GPU) | $0 | Free, requiere GPU + 4GB VRAM | $0 | $0 |

**Asunción**: 25 tokens/segundo de audio para Gemini = 1500 tokens/min = 30k tokens/20min.

## Traducción (LLM)

Per million tokens (input / output):

| Modelo | Provider | Input/M | Output/M | $/2h¹ | CLP/hora |
|--------|----------|---------|----------|-------|----------|
| **Claude Haiku 4.5** ⭐ | OpenRouter / Anthropic | $1.00 | $5.00 | $0.25 | 125 |
| gpt-4.1 | OpenAI | $2.00 | $8.00 | $0.36 | 180 |
| gpt-4.1-mini | OpenAI | $0.40 | $1.60 | $0.18 | 90 |
| gpt-4.1-nano | OpenAI | $0.10 | $0.40 | $0.03 | 15 |
| Gemini 2.5 Pro (output) | Google | $1.25 | $10.00 | ~$0.30 | 150 |
| Gemini 2.5 Flash (output) | Google | $0.30 | $2.50 | $0.21 | 105 |
| Gemini 3.1 Flash-Lite | Google | ~$0.10 | $1.50 | ~$0.11 | 55 |
| **DeepSeek V3** | OpenRouter | $0.32 | $0.89 | $0.05 | 25 |
| DeepSeek V4 Flash | OpenRouter (DeepInfra) | $0.14 | $0.28 | $0.04 | 20 |
| DeepSeek V4 Pro | OpenRouter | $1.74 | $3.48 | $0.21 | 105 |
| Kimi K2 | OpenRouter | $0.57 | $2.30 | $0.11 | 55 |
| GPT-5 Nano | OpenAI | $0.05 | $0.40 | $0.03 | 15 |
| Cohere Command Series | Cohere | varía | varía | n/a | n/a |

¹ Asumiendo ~75k input + 9.6k output tokens por 2h (basado en benchmarks reales)

## Setups completos (ASR + traducción)

### Por costo

| # | Setup | $/2h | CLP/hora | Calidad |
|---|-------|------|----------|---------|
| 1 | Cohere local + DeepSeek V3 | $0.05 | 25 | ⚠️ DeepSeek tiene errores |
| 2 | Cohere local + Gemini 3.1 Flash-Lite | $0.11 | 55 | Incógnita |
| 3 | Cohere local + Claude Haiku 4.5 | $0.25 | 125 | Alta |
| 4 | Gemini 2.5 Pro E2E | ~$0.30 | 150 | Pendiente validar |
| 5 | Gemini 2.5 Pro + Claude Haiku 4.5 | $0.43 | 215 | Alta confirmada |
| 6 | gpt-4o-transcribe + Claude Haiku 4.5 | $0.97 | 485 | Alta confirmada |
| 7 | gpt-4o-transcribe + gpt-4.1 (legacy) | $1.08 | 540 | Alta |
| 8 | turbo (local) + gpt-4.1 | $0.36 | 180 | Media (outros) |

### Recomendación según presupuesto

| Presupuesto/mes | Horas streaming/mes | Setup recomendado |
|-----------------|---------------------|-------------------|
| <$5 (~5K CLP) | 60h+ | Setup 1-2 (Cohere local + provider barato) |
| $5-15 (~5-15K) | 60h | Setup 3 (Cohere + Haiku) o Setup 4 (Gemini E2E) |
| $15-30 (~15-30K) | 60h | Setup 5 (Pro + Haiku) - calidad confirmada |
| Sin tope | cualquier | Setup 6-7 (premium APIs) |

## Fees adicionales

### OpenRouter
- **0% markup** en inferencia (passthrough exacto)
- **Stripe fee** ~3-4% al cargar créditos (una sola vez)
- **Tax** según país: Chile +20% IVA al cargar

### Pago a proveedores chinos directos
- Riesgos: validación CC en Chile, rate limits cuenta gratuita, posibles retenciones
- BYOK via OpenRouter es la opción limpia (mismo precio + 5% fee BYOK)

### Anthropic directo
- Mismo precio que via OpenRouter
- Beneficio: sin riesgo de markup futuro

## Notas para futuras revisiones

Cuando los precios cambien, actualizar esta página con fecha en el nombre:
`pricing-reference-YYYY-MM.md`

Cosas que tienden a cambiar:
1. Modelos chinos: precios bajan rápido (DeepSeek bajó 50% en 6 meses)
2. Anthropic: estables ($1/$5 hace 18 meses)
3. OpenAI: bajadas de tier en lugar de precio (gpt-5 nano apareció)
4. Google: precios estables pero tiers nuevos (Flash-Lite recién apareció)
