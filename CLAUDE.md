# AutoSubs

## Proyecto
Pipeline de subtitulos traducidos en tiempo real para YouTube.
Arquitectura: yt-dlp -> ffmpeg (audio chunks) -> Whisper API -> GPT API -> SRT -> MPV.
Stack: Python 3.13, asyncio, openai SDK. Gestion con uv.

## Reglas

### Idioma
- Comunicarse SIEMPRE en espanol con el usuario.

### Python Senior
- Python moderno (3.13+): type hints, dataclasses, match/case cuando aplique.
- asyncio para concurrencia. No threading salvo para bridge con codigo bloqueante (to_thread).
- pathlib para rutas. f-strings. Walrus operator donde mejore legibilidad.
- No over-engineer: si no se necesita, no se agrega.
- Imports organizados: stdlib, terceros, locales. Sin imports de estrella.

### Code Review
Antes de finalizar cualquier cambio, verificar:
- No hay API keys hardcodeadas ni expuestas.
- Los subprocesos (ffmpeg, mpv, yt-dlp) se limpian correctamente en finally/shutdown.
- Los archivos temporales se eliminan al terminar.
- El manejo de errores no silencia excepciones importantes.
- El codigo es idiomatico y legible sin comentarios excesivos.
- Las operaciones async no bloquean el event loop.
