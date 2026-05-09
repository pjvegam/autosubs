import asyncio
import json
import logging
import re
import shutil
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

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


def parse_time_value(raw: str) -> int:
    """Parsea un valor de tiempo a segundos.

    Soporta: '195', '195s', '3m15s', '1h2m3s', '1:30:00', '3:15'.
    """
    raw = raw.strip()
    if not raw:
        return 0

    # Solo digitos: segundos directos
    if raw.isdigit():
        return int(raw)

    # Formato con sufijo simple: '195s'
    if re.fullmatch(r"\d+s", raw):
        return int(raw[:-1])

    # Formato colon: 'HH:MM:SS' o 'MM:SS'
    if match := re.fullmatch(r"(?:(\d+):)?(\d+):(\d+)", raw):
        h = int(match.group(1) or 0)
        m = int(match.group(2))
        s = int(match.group(3))
        return h * 3600 + m * 60 + s

    # Formato HMS: '1h2m3s', '3m15s', '1h30s', '1h2m', etc.
    match = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?", raw)
    if match and any(match.groups()):
        h = int(match.group(1) or 0)
        m = int(match.group(2) or 0)
        s = int(match.group(3) or 0)
        return h * 3600 + m * 60 + s

    return 0


def parse_start_time(url: str) -> tuple[str, int]:
    """Extrae el parametro ?t= de una URL de YouTube.

    Retorna (url_limpia, segundos). Preserva otros query params.
    """
    parsed = urlparse(url)
    params = parse_qs(parsed.query)

    raw_t = params.pop("t", [None])
    seconds = parse_time_value(raw_t[0]) if raw_t and raw_t[0] else 0

    # Reconstruir URL sin ?t=
    clean_query = urlencode(params, doseq=True)
    clean_url = urlunparse(parsed._replace(query=clean_query))

    return clean_url, seconds


_cookies_browser: str = ""


def set_cookies_browser(browser: str) -> None:
    """Configura el browser para --cookies-from-browser de yt-dlp."""
    global _cookies_browser
    _cookies_browser = browser


def build_ytdl_raw_options() -> str:
    """Construye string de opciones para --ytdl-raw-options de MPV."""
    parts: list[str] = []
    if _cookies_browser:
        parts.append(f"cookies-from-browser={_cookies_browser}")
    if deno_path := _find_deno():
        parts.append(f"js-runtimes=deno:{deno_path}")
    return ",".join(parts)


def _ytdlp_base_args() -> list[str]:
    """Argumentos base para yt-dlp, incluyendo cookies y deno si existen."""
    args = ["yt-dlp", "--remote-components", "ejs:github"]
    if deno_path := _find_deno():
        args.extend(["--js-runtimes", f"deno:{deno_path}"])
    if _cookies_browser:
        args.extend(["--cookies-from-browser", _cookies_browser])
    return args


async def download_audio(url: str, output: str) -> None:
    """Descarga solo audio con yt-dlp (evita throttling de YouTube)."""
    cmd = [
        *_ytdlp_base_args(),
        "-f", "ba",
        "--no-part",
        "-o", output,
        url,
    ]
    logger.info("Descargando audio...")
    proc = await asyncio.create_subprocess_exec(*cmd)
    returncode = await proc.wait()

    if returncode != 0:
        raise RuntimeError(f"yt-dlp audio download fallo (exit {returncode})")
    out_path = Path(output)
    if not out_path.exists():
        raise RuntimeError(f"Archivo no encontrado: {output}")
    size_mb = out_path.stat().st_size / (1024 * 1024)
    logger.info("Audio descargado: %.1f MB", size_mb)


async def split_audio(
    input_path: str, output_dir: str, chunk_seconds: int = 1200,
) -> list[Path]:
    """Splitea audio en chunks de duracion fija con ffmpeg."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pattern = str(out / "chunk_%03d.mp3")

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-f", "segment",
        "-segment_time", str(chunk_seconds),
        "-c:a", "libmp3lame", "-q:a", "5",
        pattern,
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    returncode = await proc.wait()
    if returncode != 0:
        raise RuntimeError(f"ffmpeg split fallo (exit {returncode})")

    chunks = sorted(out.glob("chunk_*.mp3"))
    logger.info("Audio spliteado en %d chunks de %ds", len(chunks), chunk_seconds)
    return chunks


async def download_vod(url: str, output: str) -> None:
    """Descarga un VOD completo con yt-dlp.

    El progreso se muestra en consola (stderr heredado).
    yt-dlp maneja anti-throttling de YouTube automaticamente.
    """
    cmd = [
        *_ytdlp_base_args(),
        "-f", "bv*+ba/b*",
        "--merge-output-format", "mkv",
        "--no-part",
        "-o", output,
        url,
    ]
    logger.info("Descargando VOD completo...")
    proc = await asyncio.create_subprocess_exec(*cmd)
    returncode = await proc.wait()

    if returncode != 0:
        raise RuntimeError(f"yt-dlp download fallo (exit {returncode})")
    out_path = Path(output)
    if not out_path.exists():
        raise RuntimeError(f"Archivo no encontrado: {output}")
    size_mb = out_path.stat().st_size / (1024 * 1024)
    logger.info("Descarga completada: %.1f MB", size_mb)


async def cut_video(input_path: str, output: str, start_time: int) -> None:
    """Corta un video desde start_time con ffmpeg stream copy (instantaneo)."""
    hh, rem = divmod(start_time, 3600)
    mm, ss = divmod(rem, 60)
    logger.info("Cortando video desde %02d:%02d:%02d...", hh, mm, ss)

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_time),
        "-i", input_path,
        "-c", "copy",
        output,
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    returncode = await proc.wait()

    if returncode != 0:
        raise RuntimeError(f"ffmpeg cut fallo (exit {returncode})")
    logger.info("Corte completado")


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
    """Extrae URL del stream de audio mas ligero.

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
