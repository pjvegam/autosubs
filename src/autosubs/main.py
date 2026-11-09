import argparse
import asyncio
import logging
import sys

from autosubs.config import Config
from autosubs.stream import parse_start_time, parse_time_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="autosubs",
        description="Subtitulos traducidos en vivo para YouTube",
    )
    parser.add_argument(
        "url", nargs="?", default="",
        help="URL del video o stream de YouTube (no requerido con --system-audio)",
    )
    parser.add_argument(
        "--source-lang", default="",
        help="Idioma del audio (default: AUTOSUBS_SOURCE_LANG o ja)",
    )
    parser.add_argument(
        "--target-lang", default="",
        help="Idioma de los subtitulos (default: AUTOSUBS_TARGET_LANG o es)",
    )
    parser.add_argument(
        "--segment-duration", default="",
        help="Duracion de segmento. Acepta: 5, 5s, 30s (default: AUTOSUBS_SEGMENT_DURATION o 5)",
    )
    parser.add_argument(
        "--whisper-model", default="",
        help="Modelo de transcripcion: microsoft/mai-transcribe-2 (API), "
             "qwen/qwen3-asr-flash (API), local-turbo (GPU), local-v3 (GPU). "
             "Default: AUTOSUBS_WHISPER_MODEL, mai-transcribe-2 si OPENROUTER_API_KEY, "
             "sino local-turbo, sino gpt-4o-transcribe",
    )
    parser.add_argument(
        "--gpt-model", default="",
        help="Modelo de traduccion. OpenAI (gpt-4.1, gpt-4.1-mini) o "
             "OpenRouter con formato provider/modelo (anthropic/claude-haiku-5.5, "
             "deepseek/deepseek-chat, etc). Default: anthropic/claude-haiku-5.5 si "
             "OPENROUTER_API_KEY, sino gpt-4.1",
    )
    parser.add_argument(
        "--cookies-browser", default="",
        help="Browser para cookies de yt-dlp (default: AUTOSUBS_COOKIES_BROWSER o firefox). "
             "Usar 'none' para desactivar.",
    )
    parser.add_argument(
        "--system-audio", action="store_true",
        help="Capturar audio del sistema (loopback WASAPI) en vez de YouTube",
    )
    parser.add_argument(
        "--diarize", action="store_true",
        help="Activar diarizacion de speakers (requiere pip install autosubs[diarize])",
    )
    parser.add_argument(
        "--batch-vod", action="store_true",
        help="Pre-procesar VOD completo con chunks grandes (mejor calidad)",
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Compara whisper-1 vs gpt-4o-transcribe sobre el mismo audio",
    )
    parser.add_argument(
        "--benchmark-no-translate", action="store_true",
        help="En modo benchmark, saltar traduccion (solo comparar transcripcion)",
    )
    parser.add_argument(
        "--benchmark-translators", action="store_true",
        help="Comparar multiples traductores sobre transcripcion cacheada de gpt-4o",
    )
    parser.add_argument(
        "--benchmark-e2e", action="store_true",
        help="Test E2E: Gemini 2.5 Pro audio->ES en una sola call",
    )
    parser.add_argument(
        "--fill-gaps", action="store_true",
        help="Rellena gaps en SRT cacheado usando el video local y --whisper-model",
    )
    parser.add_argument(
        "--vad-aggressiveness", type=int, default=-1,
        help="Agresividad del VAD 0-3 (default: 2)",
    )
    parser.add_argument(
        "--start-time", default="",
        help="Tiempo de inicio. Acepta: 3600, 3600s, 1h, 1h30m, 1:30:00, 30:00 "
             "(override sobre ?t= en URL)",
    )
    parser.add_argument(
        "--end-time", default="",
        help="Tiempo del video en que parar. Acepta: 3600, 55m, 1:30:00, 55:00. "
             "Combinado con --start-time traduce solo ese rango "
             "(default: hasta el final)",
    )
    parser.add_argument(
        "--max-minutes", default="",
        help="Minuto del video en que parar (posicion, no duracion: con "
             "--start-time 45:00 usa 55 para llegar al 55:00, no 10). "
             "Acepta: 100, 100m, 1h40m, 1:40:00. Preferir --end-time",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Forzar re-traduccion aunque exista SRT en cache",
    )
    parser.add_argument(
        "--speaker-gender", default="", choices=["f", "m", ""],
        help="Genero del hablante principal para consistencia gramatical (f/m)",
    )
    parser.add_argument(
        "--speakers", type=int, default=0,
        help="Numero de speakers en el video (da contexto a GPT)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Activar logging de debug",
    )
    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Silenciar logs ruidosos de SDKs externos
    logging.getLogger("httpx").setLevel(logging.WARNING)
    # Gemini SDK: spam "AFC is enabled with max remote calls: 10" en cada call
    logging.getLogger("google_genai.models").setLevel(logging.WARNING)
    # OpenAI SDK: retries internos automaticos (nuestro with_retries ya loggea fallos reales)
    logging.getLogger("openai._base_client").setLevel(logging.WARNING)


async def async_main(config: Config) -> None:
    from autosubs.pipeline import run_pipeline
    await run_pipeline(config)


def _parse_time_arg(value: str, arg_name: str, *, unit: str = "seconds") -> int:
    """Parsea CLI time arg.

    Entero puro = se interpreta en la unidad natural del flag (backward compat):
      --start-time 100        → 100 segundos
      --max-minutes 100       → 100 minutos
      --segment-duration 5    → 5 segundos
    Con sufijo/colon = se parsea como tiempo y se convierte a la unidad:
      --max-minutes 1h40m     → 100 minutos
      --start-time 1:30:00    → 5400 segundos
    """
    if not value:
        return 0
    raw = value.strip()
    if raw.isdigit():
        return int(raw)
    seconds = parse_time_value(raw)
    if seconds == 0:
        print(f"Error: --{arg_name} invalido: '{value}'", file=sys.stderr)
        sys.exit(1)
    return seconds // 60 if unit == "minutes" else seconds


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    # Extraer ?t= de la URL; --start-time CLI tiene precedencia
    url = args.url or ""
    url_start_time = 0
    if url:
        url, url_start_time = parse_start_time(url)
    cli_start_time = _parse_time_arg(args.start_time, "start-time")
    start_time = cli_start_time or url_start_time

    end_time = _parse_time_arg(args.end_time, "end-time")
    max_minutes = _parse_time_arg(args.max_minutes, "max-minutes", unit="minutes")
    segment_duration = _parse_time_arg(args.segment_duration, "segment-duration")

    try:
        config = Config(
            url=url,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            segment_duration=segment_duration,
            whisper_model=args.whisper_model,
            gpt_model=args.gpt_model,
            cookies_browser=args.cookies_browser,
            system_audio=args.system_audio,
            diarize=args.diarize,
            batch_vod=args.batch_vod,
            benchmark=args.benchmark,
            benchmark_no_translate=args.benchmark_no_translate,
            benchmark_translators=args.benchmark_translators,
            benchmark_e2e=args.benchmark_e2e,
            fill_gaps=args.fill_gaps,
            vad_aggressiveness=args.vad_aggressiveness,
            start_time=start_time,
            end_time=end_time,
            max_minutes=max_minutes,
            no_cache=args.no_cache,
            speaker_gender=args.speaker_gender,
            speakers=args.speakers,
            verbose=args.verbose,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        asyncio.run(async_main(config))
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario.")


if __name__ == "__main__":
    main()
