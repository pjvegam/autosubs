# Ideas y mejoras pendientes — AutoSubs

## Mejoras de calidad rapidas (low-hanging fruit)

### 1. Deduplicar repetidos consecutivos
Problema: turbo alucina el mismo texto N veces seguidas (ej: "ご視聴ありがとうございました" x24).
Solucion: en `_transcribe_worker`, si el texto es identico al anterior, descartarlo.
Impacto: elimina alucinaciones repetitivas, ahorra llamadas GPT innecesarias.

### 2. Agrupar chunks cortos antes de traducir
Problema: turbo devuelve entries muy cortas ("んー", "え?") que se traducen individualmente.
Solucion: si la transcripcion es < N caracteres, concatenar con el siguiente chunk antes de enviar a GPT.
Impacto: menos llamadas GPT, traducciones con mas contexto.

### 3. Ajustar segment_duration
Problema: chunks de 5s (min=3s, max=8s) pueden ser cortos para turbo.
Solucion: probar segment_duration=8 (min=4.8s, max=12.8s) para dar mas contexto a Whisper.
Impacto: menos fragmentacion, mejor transcripcion, mas latencia.

## Streaming ASR (whisper_streaming)

### Concepto
En vez de chunk → transcribe → translate → mostrar, usar transcripcion continua:
- Audio se envia al modelo cada ~1s
- El modelo devuelve resultados parciales (provisionales) que se muestran en tiempo real
- Cuando detecta frase completa, emite resultado final confirmado
- Solo el resultado final se envia a traducir

### Ventajas
- "Local agreement policy": compara 2 iteraciones, solo emite lo que ambas confirman → filtra alucinaciones naturalmente
- Buffer continuo vs chunks aislados → mejor contexto para el modelo
- Latencia percibida mucho menor (texto se forma en pantalla)
- Eliminaria la dependencia del VAD para segmentar

### Implementacion
- Libreria: [whisper_streaming](https://github.com/ufal/whisper_streaming) (usa faster-whisper como backend)
- Alternativa: [WhisperLive](https://github.com/collabora/WhisperLive)
- Requiere repensar el pipeline: audio continuo → texto parcial → traducir solo frases finales
- Compatible con nuestro stack (faster-whisper + asyncio)

### Riesgo
- Cambio arquitectural grande (reescribir transcribe worker + segmentacion)
- whisper_streaming puede tener sus propios bugs/limitaciones
- Hay que validar que funcione bien con japones coloquial (gaming/VTuber)

## Traduccion mas barata

### Opciones evaluadas
| Opcion | Costo/Mtok in | Contexto | Calidad ja→es |
|--------|--------------|----------|---------------|
| gpt-4.1 (actual) | $2.00 | Si (historial, metadata) | Excelente |
| gpt-4.1-nano | $0.10 | Si | Por probar |
| Gemini 2.0 Flash | $0.10 | Si | Por probar |
| TranslateGemma 4B local | $0 | No | Por probar |

### TranslateGemma 4B (local, $0)
- Modelo de Google especializado en traduccion, 55 idiomas
- ~3GB VRAM (cabe con faster-whisper turbo ~2GB en RTX 5060 8GB)
- Disponible en Ollama: `ollama run translategemma:4b`
- Limitacion: no soporta instrucciones de contexto (historial, genero, metadata)
- Paper reporta regresion en japones (errores de named entities)
