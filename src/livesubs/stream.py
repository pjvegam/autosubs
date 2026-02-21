import asyncio
import json
import logging
import shutil

logger = logging.getLogger(__name__)

_DENO_SEARCH_PATHS_WIN = [
    r"~\AppData\Local\Microsoft\WinGet\Packages",
    r"~\.deno\bin",
]

_DENO_SEARCH_PATHS_UNIX = [
    "~/.deno/bin",
]


def _find_deno() -> str | None:
    """Busca deno en PATH y ubicaciones comunes."""
    import sys
    from pathlib import Path

    if path := shutil.which("deno"):
        return path

    is_win = sys.platform == "win32"
    search_paths = _DENO_SEARCH_PATHS_WIN if is_win else _DENO_SEARCH_PATHS_UNIX
    exe_name = "deno.exe" if is_win else "deno"

    for base in search_paths:
        expanded = Path(base).expanduser()
        if not expanded.exists():
            continue
        for exe in expanded.rglob(exe_name):
            logger.info("deno encontrado: %s", exe)
            return str(exe)
    return None


def _ytdlp_base_args() -> list[str]:
    """Argumentos base para yt-dlp, incluyendo --js-runtimes si deno existe."""
    args = ["yt-dlp"]
    if deno_path := _find_deno():
        args.extend(["--js-runtimes", f"deno:{deno_path}"])
    return args


def ytdlp_pipe_audio_cmd(url: str) -> list[str]:
    """Comando yt-dlp para descargar audio a stdout (piping a ffmpeg).

    Usa wa* (audio mas ligero, incluso combinado) para minimizar bandwidth.
    yt-dlp maneja reconexiones y URL refreshing automaticamente.
    """
    return [*_ytdlp_base_args(), "-f", "wa*", "-o", "-", url]


async def get_stream_info(url: str) -> dict:
    """Extrae metadata del video/stream via yt-dlp -j."""
    cmd = [*_ytdlp_base_args(), "-j", url]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"yt-dlp fallo: {stderr.decode().strip()}")
    return json.loads(stdout.decode())


async def _extract_url(url: str, format_spec: str, label: str) -> str:
    """Extrae una URL directa via yt-dlp -g."""
    cmd = [*_ytdlp_base_args(), "-g", "-f", format_spec, url]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"yt-dlp {label} fallo: {stderr.decode().strip()}")
    return stdout.decode().strip().splitlines()[0]


async def get_stream_urls(url: str) -> tuple[str, str | None, str]:
    """Extrae URLs directas de video y audio.

    Retorna (video_url, mpv_audio_url, ffmpeg_audio_url).
    - DASH: video-only + audio-only separados. MPV necesita --audio-file.
    - HLS/combinado: video ya tiene audio. ffmpeg usa stream mas ligero.
    """
    try:
        video_url, audio_url = await asyncio.gather(
            _extract_url(url, "bv*", "video"),
            _extract_url(url, "ba", "audio"),
        )
        logger.info("Usando DASH separado (video + audio)")
        return video_url, audio_url, audio_url
    except RuntimeError:
        pass

    # Sin audio-only: HLS o stream combinado
    # MPV usa stream-record para grabar y ffmpeg lee local (sin doble descarga)
    # wa* solo se usa como fallback si stream-record no funciona
    try:
        video_url, light_url = await asyncio.gather(
            _extract_url(url, "bv*", "video"),
            _extract_url(url, "wa*", "audio ligero"),
        )
        logger.info("Usando HLS (stream-record disponible)")
        return video_url, None, light_url
    except RuntimeError:
        logger.warning("Fallback a stream unico combinado")
        combined = await _extract_url(url, "b*", "stream")
        return combined, None, combined


async def get_audio_url(url: str) -> str:
    """Extrae URL del stream de audio mas ligero (para modo subs-only).

    Intenta: wa* (peor audio, mas ligero) -> ba (audio-only) -> b* (combinado).
    """
    for fmt, label in [("wa*", "audio ligero"), ("ba", "audio"), ("b*", "stream")]:
        try:
            return await _extract_url(url, fmt, label)
        except RuntimeError:
            continue
    raise RuntimeError("No se pudo obtener URL de audio")


async def is_live(url: str) -> bool:
    """Detecta si la URL es un live stream."""
    try:
        info = await get_stream_info(url)
        return info.get("is_live", False)
    except Exception:
        logger.warning("No se pudo detectar si es live, asumiendo VOD")
        return False
