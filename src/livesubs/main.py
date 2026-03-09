import argparse
import asyncio
import logging
import sys

from livesubs.config import Config
from livesubs.stream import parse_start_time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="livesubs",
        description="Subtitulos traducidos en vivo para YouTube",
    )
    parser.add_argument(
        "url", nargs="?", default="",
        help="URL del video o stream de YouTube (no requerido con --system-audio)",
    )
    parser.add_argument(
        "--source-lang", default="",
        help="Idioma del audio (default: LIVESUBS_SOURCE_LANG o ja)",
    )
    parser.add_argument(
        "--target-lang", default="",
        help="Idioma de los subtitulos (default: LIVESUBS_TARGET_LANG o es)",
    )
    parser.add_argument(
        "--segment-duration", type=int, default=0,
        help="Duracion de segmento en segundos (default: LIVESUBS_SEGMENT_DURATION o 5)",
    )
    parser.add_argument(
        "--whisper-model", default="",
        help="Modelo de transcripcion: gpt-4o-transcribe (API), local-turbo (GPU), "
             "local-v3 (GPU). Default: LIVESUBS_WHISPER_MODEL o gpt-4o-transcribe",
    )
    parser.add_argument(
        "--gpt-model", default="",
        help="Modelo de GPT (default: LIVESUBS_GPT_MODEL o gpt-4.1)",
    )
    parser.add_argument(
        "--cookies-browser", default="",
        help="Browser para cookies de yt-dlp (default: LIVESUBS_COOKIES_BROWSER o firefox). "
             "Usar 'none' para desactivar.",
    )
    parser.add_argument(
        "--system-audio", action="store_true",
        help="Capturar audio del sistema (loopback WASAPI) en vez de YouTube",
    )
    parser.add_argument(
        "--diarize", action="store_true",
        help="Activar diarizacion de speakers (requiere pip install livesubs[diarize])",
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
        "--vad-aggressiveness", type=int, default=0,
        help="Agresividad del VAD 0-3 (default: 2)",
    )
    parser.add_argument(
        "--start-time", type=int, default=0,
        help="Tiempo de inicio en segundos (override sobre ?t= en URL)",
    )
    parser.add_argument(
        "--max-minutes", type=int, default=0,
        help="Limite de minutos a traducir (default: 0 = sin limite)",
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
    # Silenciar logs de httpx (spam de cada HTTP request)
    logging.getLogger("httpx").setLevel(logging.WARNING)


async def async_main(config: Config) -> None:
    from livesubs.pipeline import run_pipeline
    await run_pipeline(config)


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    # Extraer ?t= de la URL; --start-time CLI tiene precedencia
    url = args.url or ""
    url_start_time = 0
    if url:
        url, url_start_time = parse_start_time(url)
    start_time = args.start_time if args.start_time else url_start_time

    try:
        config = Config(
            url=url,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            segment_duration=args.segment_duration,
            whisper_model=args.whisper_model,
            gpt_model=args.gpt_model,
            cookies_browser=args.cookies_browser,
            system_audio=args.system_audio,
            diarize=args.diarize,
            batch_vod=args.batch_vod,
            benchmark=args.benchmark,
            vad_aggressiveness=args.vad_aggressiveness,
            start_time=start_time,
            max_minutes=args.max_minutes,
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
