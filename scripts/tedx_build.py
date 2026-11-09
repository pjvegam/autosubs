"""Reconstruye un subset de TEDxJP-10K como ground truth local.

TEDxJP-10K no se distribuye como archivos: hay que bajar los videos de YouTube
y reconstruirlo. Este script lo hace end-to-end y deja un manifest listo para
`tedx_eval.py`.

    uv run --extra tedx scripts/tedx_build.py --videos 10

Salida en tedx_data/: audio completo (wav 16k mono), un clip por utterance y
manifest.json con las transcripciones de referencia.
"""

import argparse
import asyncio
import functools
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

print = functools.partial(print, flush=True)  # type: ignore[assignment]

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from autosubs.stream import _find_deno
from autosubs.tedx import Utterance, build_utterances

TEDX_REPO = "https://github.com/laboroai/TEDxJP-10K.git"
DATASET_VERSION = "1.1"
SAMPLE_RATE = 16000
DOWNLOAD_CONCURRENCY = 2
CLIP_CONCURRENCY = 8

# En estos VODs de 2013-2019 el resto de los clientes exige un GVS PO token y
# se queda sin formatos ("Only images are available") o cae en HTTP 403 al
# descargar. android_vr es el unico que entrega audio y subtitulos sin token.
# No se usa el web_safari de stream.py: ese sirve para streams en vivo.
_PLAYER_CLIENT = "android_vr"

_cookies_browser = ""


def _ytdlp_args() -> list[str]:
    """Argumentos de yt-dlp para bajar los VODs del dataset."""
    args = [
        "yt-dlp",
        "--remote-components", "ejs:github",
        "--extractor-args", f"youtube:player_client={_PLAYER_CLIENT}",
        "--retries", "5",
        "--sleep-requests", "1",  # YouTube devuelve 429 si se le pide muy seguido
    ]
    if deno_path := _find_deno():
        args.extend(["--js-runtimes", f"deno:{deno_path}"])
    if _cookies_browser:
        args.extend(["--cookies-from-browser", _cookies_browser])
    return args


async def _run(cmd: list[str], what: str) -> None:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        detail = stderr.decode(errors="replace").strip().splitlines()[-3:]
        raise RuntimeError(f"{what} fallo (exit {proc.returncode}): {' | '.join(detail)}")


async def clone_repo(repo_dir: Path) -> None:
    """Clona el repo oficial (solo trae la tabla de hashes y los diffs)."""
    if (repo_dir / "data" / DATASET_VERSION / "utt_id_table.csv").exists():
        print(f"Repo TEDxJP-10K ya presente: {repo_dir}")
        return
    print(f"Clonando {TEDX_REPO}...")
    await _run(["git", "clone", "--depth", "1", TEDX_REPO, str(repo_dir)], "git clone")


async def fetch_video(video_id: str, audio_dir: Path, vtt_dir: Path) -> bool:
    """Baja subtitulo manual japones + audio, y normaliza a wav 16k mono."""
    wav_path = audio_dir / f"{video_id}.wav"
    vtt_path = vtt_dir / f"{video_id}.ja.vtt"
    url = f"https://www.youtube.com/watch?v={video_id}"

    if not vtt_path.exists():
        # --write-subs (no --write-auto-subs): solo subtitulos humanos.
        await _run([
            *_ytdlp_args(),
            "--skip-download", "--write-subs",
            "--sub-langs", "ja", "--sub-format", "vtt",
            "-o", str(vtt_dir / "%(id)s.%(ext)s"),
            url,
        ], f"yt-dlp subs {video_id}")

        if not vtt_path.exists():
            print(f"  {video_id}: sin subtitulos manuales en japones, se omite")
            return False

    if not wav_path.exists():
        raw = audio_dir / f"{video_id}.raw"
        try:
            # ba/b: si el video no tiene pista de solo-audio, se baja el mejor
            # formato completo y ffmpeg extrae el audio igual (-vn).
            await _run([
                *_ytdlp_args(), "-f", "ba/b", "--no-part", "-o", str(raw), url,
            ], f"yt-dlp audio {video_id}")
            await _run([
                "ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-vn",
                "-ar", str(SAMPLE_RATE), "-ac", "1", str(wav_path),
            ], f"ffmpeg wav {video_id}")
        finally:
            raw.unlink(missing_ok=True)

    print(f"  {video_id}: ok")
    return True


async def cut_clip(audio_path: Path, clip_path: Path, start: float, end: float) -> None:
    """Extrae un clip. -ss antes de -i hace seek sin decodificar todo el archivo."""
    if clip_path.exists():
        return
    await _run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start:.2f}", "-i", str(audio_path), "-t", f"{end - start:.2f}",
        "-ar", str(SAMPLE_RATE), "-ac", "1", str(clip_path),
    ], f"ffmpeg clip {clip_path.name}")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=int, default=10, help="cuantos videos bajar")
    parser.add_argument("--out", type=Path, default=_PROJECT_ROOT / "tedx_data")
    parser.add_argument("--repo", type=Path, default=None, help="repo TEDxJP-10K ya clonado")
    parser.add_argument(
        "--cookies-from-browser", default="",
        help="browser del que sacar cookies si YouTube pide verificacion",
    )
    args = parser.parse_args()

    global _cookies_browser
    _cookies_browser = args.cookies_from_browser

    out: Path = args.out
    repo_dir: Path = args.repo or (out / "TEDxJP-10K")
    audio_dir, vtt_dir, clip_dir = out / "audio", out / "vtt", out / "clips"
    for d in (audio_dir, vtt_dir, clip_dir):
        d.mkdir(parents=True, exist_ok=True)

    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg no esta en PATH")

    await clone_repo(repo_dir)

    urls = (repo_dir / "data" / "tedx-jp_urls.txt").read_text().split()
    video_ids = [u.rsplit("=", 1)[-1] for u in urls][: args.videos]

    print(f"\nDescargando {len(video_ids)} videos...")
    sem = asyncio.Semaphore(DOWNLOAD_CONCURRENCY)

    async def _fetch(video_id: str) -> str | None:
        async with sem:
            try:
                return video_id if await fetch_video(video_id, audio_dir, vtt_dir) else None
            except RuntimeError as e:
                print(f"  {video_id}: {e}")
                return None

    downloaded = {v for v in await asyncio.gather(*(_fetch(v) for v in video_ids)) if v}
    if not downloaded:
        sys.exit("No se pudo descargar ningun video")
    if failed := len(video_ids) - len(downloaded):
        print(f"  ({failed} videos sin audio o sin subtitulos, se excluyen)")

    print("\nReconstruyendo utterances desde los subtitulos...")
    utterances, missing = build_utterances(
        vtt_dir, repo_dir / "data" / DATASET_VERSION, video_ids=downloaded,
    )
    if not utterances:
        sys.exit("Ninguna utterance reconstruida: revisa los .vtt descargados")

    print(f"  {len(utterances)} utterances reconstruidas")
    if missing:
        pct = missing / (missing + len(utterances))
        print(f"  {missing} descartadas ({pct:.1%}): subtitulo editado en YouTube desde 2020")

    print("\nCortando clips...")
    clip_sem = asyncio.Semaphore(CLIP_CONCURRENCY)

    async def _cut(utt: Utterance) -> Utterance | None:
        async with clip_sem:
            try:
                await cut_clip(
                    audio_dir / f"{utt.video_id}.wav",
                    clip_dir / f"{utt.utt_id}.wav",
                    utt.start, utt.end,
                )
            except RuntimeError as e:
                print(f"  {utt.utt_id}: {e}")
                return None
            return utt

    cut = await asyncio.gather(*(_cut(u) for u in utterances))
    dropped = sum(1 for u in cut if u is None)
    utterances = [u for u in cut if u is not None]
    if dropped:
        print(f"  {dropped} utterances sin clip, se excluyen del manifest")
    if not utterances:
        sys.exit("No se pudo cortar ningun clip")

    total_seconds = sum(u.duration for u in utterances)
    manifest = {
        "dataset": "TEDxJP-10K",
        "version": DATASET_VERSION,
        "license": "Apache-2.0 (Laboro.AI)",
        "videos": sorted({u.video_id for u in utterances}),
        "utterance_count": len(utterances),
        "discarded_count": missing,
        "total_audio_seconds": round(total_seconds, 2),
        "utterances": [
            {**asdict(u), "clip": f"clips/{u.utt_id}.wav"} for u in utterances
        ],
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    print(
        f"\nListo: {len(utterances)} utterances / "
        f"{total_seconds / 60:.1f} min de audio evaluable\n"
        f"Manifest: {manifest_path}",
    )


if __name__ == "__main__":
    asyncio.run(main())
