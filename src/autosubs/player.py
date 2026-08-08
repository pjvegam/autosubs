import asyncio
import contextlib
import json
import logging
import subprocess
from pathlib import Path
from typing import IO

logger = logging.getLogger(__name__)


class MPVPlayer:
    """Lanza MPV y se comunica via IPC (named pipe en Windows)."""

    def __init__(self, ipc_pipe: str) -> None:
        self.ipc_pipe = ipc_pipe
        self.process: subprocess.Popen | None = None
        self._pipe_handle: IO[bytes] | None = None
        self._request_id = 0
        self._connected = False

    def _next_request_id(self) -> int:
        self._request_id += 1
        return self._request_id

    async def start(
        self,
        video_url: str,
        audio_url: str | None = None,
        live: bool = False,
        start_time: int = 0,
    ) -> None:
        """Lanza MPV con la URL del video (y audio separado si es DASH).

        start_time se pasa como --start en vez de mandar un seek por IPC:
        contra un stream HLS el seek llega antes de que MPV tenga el indice
        cargado y se pierde sin avisar.
        """
        cmd = [
            "mpv",
            f"--input-ipc-server={self.ipc_pipe}",
            "--force-window=yes",
            "--sub-visibility=yes",
        ]
        if start_time > 0:
            cmd.append(f"--start={start_time}")
        if live:
            cmd.extend([
                "--cache=yes",
                "--demuxer-max-bytes=150MiB",
                "--demuxer-readahead-secs=30",
            ])
        if audio_url:
            cmd.append(f"--audio-file={audio_url}")
        cmd.append(video_url)

        logger.info("Lanzando MPV...")
        self.process = await asyncio.to_thread(
            subprocess.Popen,
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    async def _connect_pipe(self, max_retries: int = 15) -> None:
        """Conecta al named pipe de MPV con retry."""
        for attempt in range(max_retries):
            try:
                self._pipe_handle = await asyncio.to_thread(
                    open, self.ipc_pipe, "r+b", 0
                )
                logger.info("IPC conectado a MPV (intento %d)", attempt + 1)
                return
            except OSError:
                wait = min(0.5 * (attempt + 1), 3.0)
                if attempt % 3 == 2:
                    logger.info(
                        "Esperando IPC de MPV... (intento %d/%d)", attempt + 1, max_retries,
                    )
                await asyncio.sleep(wait)
        raise ConnectionError(
            f"No se pudo conectar al pipe de MPV despues de {max_retries} intentos"
        )

    async def connect(self) -> None:
        """Conecta al IPC de MPV y verifica que funcione."""
        await self._connect_pipe()
        resp = await self._send_command(["get_property", "mpv-version"])
        if resp and resp.get("error") == "success":
            logger.info("MPV version: %s", resp.get("data"))
            self._connected = True
        else:
            logger.warning("IPC conectado pero no responde correctamente: %s", resp)
            self._connected = True

    async def setup_osd_for_subs(self) -> None:
        """Configura OSD para mostrar texto como subtitulos (abajo, centrado)."""
        await self._send_command(["set_property", "osd-align-y", "bottom"])
        await self._send_command(["set_property", "osd-align-x", "center"])
        await self._send_command(["set_property", "osd-margin-y", "50"])
        await self._send_command(["set_property", "osd-font-size", "40"])

    async def _send_command(self, command: list[str]) -> dict | None:
        """Envia un comando JSON por IPC y lee la respuesta.

        MPV envia eventos async (audio-reconfig, etc.) por el mismo pipe.
        Leemos lineas hasta encontrar la respuesta con nuestro request_id.
        """
        if self._pipe_handle is None:
            return None
        req_id = self._next_request_id()
        msg = json.dumps({"command": command, "request_id": req_id}) + "\n"

        def _write_read() -> dict | None:
            self._pipe_handle.write(msg.encode())  # type: ignore[union-attr]
            self._pipe_handle.flush()  # type: ignore[union-attr]
            # Leer hasta encontrar nuestra respuesta (skip eventos)
            for _ in range(20):
                raw = self._pipe_handle.readline()  # type: ignore[union-attr]
                if not raw:
                    return None
                try:
                    data = json.loads(raw.decode())
                except json.JSONDecodeError:
                    continue
                if data.get("request_id") == req_id:
                    return data
                # Es un evento async de MPV, ignorar
            return None

        try:
            return await asyncio.to_thread(_write_read)
        except OSError as e:
            logger.warning("Error IPC: %s", e)
            self._connected = False
        return None

    async def command(self, *args: str) -> dict | None:
        """Ejecuta un comando generico de MPV."""
        return await self._send_command(list(args))

    async def get_time_pos(self) -> float | None:
        """Obtiene la posicion actual de reproduccion en segundos."""
        resp = await self._send_command(["get_property", "time-pos"])
        if resp and resp.get("error") == "success":
            return resp.get("data")
        return None

    async def show_text(self, text: str, duration_ms: int = 5000) -> None:
        """Muestra texto en pantalla por N milisegundos (OSD overlay)."""
        await self._send_command(["expand-properties", "show-text", text, str(duration_ms)])

    async def sub_add(self, path: Path) -> None:
        """Agrega un archivo de subtitulos."""
        resp = await self._send_command(["sub-add", str(path)])
        if resp and resp.get("error") == "success":
            logger.info("Subtitulo cargado en MPV: %s", path.name)
        else:
            logger.warning("Fallo al cargar subtitulo: %s", resp)

    async def sub_reload(self, path: Path) -> None:
        """Recarga subtitulos: remove + add para forzar relectura del archivo."""
        await self._send_command(["sub-remove"])
        await self._send_command(["sub-add", str(path)])

    def is_running(self) -> bool:
        """Verifica si MPV sigue corriendo."""
        if self.process is None:
            return False
        return self.process.poll() is None

    async def wait(self) -> int:
        """Espera a que MPV termine."""
        if self.process is None:
            return -1
        return await asyncio.to_thread(self.process.wait)

    async def close(self) -> None:
        """Cierra la conexion IPC y termina MPV."""
        if self._pipe_handle is not None:
            with contextlib.suppress(OSError):
                await asyncio.to_thread(self._pipe_handle.close)
            self._pipe_handle = None
        if self.process is not None and self.is_running():
            self.process.terminate()
            try:
                await asyncio.to_thread(self.process.wait, timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
