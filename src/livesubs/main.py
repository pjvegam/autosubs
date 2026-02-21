import argparse
import asyncio
import logging
import sys

from livesubs.config import Config


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
        help="Modelo de transcripcion (default: LIVESUBS_WHISPER_MODEL o gpt-4o-transcribe)",
    )
    parser.add_argument(
        "--gpt-model", default="",
        help="Modelo de GPT (default: LIVESUBS_GPT_MODEL o gpt-4o-mini)",
    )
    parser.add_argument(
        "--subs-only", action="store_true",
        help="Solo subtitulos en ventana overlay (sin MPV). Auto para lives.",
    )
    parser.add_argument(
        "--system-audio", action="store_true",
        help="Capturar audio del sistema (loopback WASAPI) en vez de YouTube",
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


async def async_main(config: Config) -> None:
    from livesubs.pipeline import run_pipeline
    await run_pipeline(config)


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    try:
        config = Config(
            url=args.url or "",
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            segment_duration=args.segment_duration,
            whisper_model=args.whisper_model,
            gpt_model=args.gpt_model,
            subs_only=args.subs_only,
            system_audio=args.system_audio,
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
