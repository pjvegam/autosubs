import asyncio
import contextlib
import logging
import tkinter as tk
from collections.abc import Callable

logger = logging.getLogger(__name__)

# Constantes de estilo
_BG_COLOR = "#1a1a1a"
_FG_COLOR = "#ffffff"
_FONT_FAMILY = "Segoe UI"
_FONT_SIZE = 28
_PADDING_X = 40
_PADDING_Y = 20
_OPACITY = 0.80
_BOTTOM_MARGIN = 60
_WIDTH_RATIO = 0.6  # 60% del ancho de pantalla


class SubtitleOverlay:
    """Ventana semi-transparente always-on-top para mostrar subtitulos."""

    def __init__(self) -> None:
        self._root: tk.Tk | None = None
        self._label: tk.Label | None = None
        self._on_close: Callable[[], None] | None = None

    def start(self, on_close: Callable[[], None] | None = None) -> None:
        """Crea y muestra la ventana overlay.

        Debe llamarse desde el hilo principal (tkinter lo requiere).
        """
        self._on_close = on_close
        self._root = tk.Tk()
        self._root.title("livesubs")
        self._root.configure(bg=_BG_COLOR)

        # Sin bordes, siempre encima
        self._root.overrideredirect(True)
        self._root.attributes("-topmost", True)
        self._root.attributes("-alpha", _OPACITY)

        # Dimensiones: 60% del ancho, posicion centrada abajo
        screen_w = self._root.winfo_screenwidth()
        screen_h = self._root.winfo_screenheight()
        win_w = int(screen_w * _WIDTH_RATIO)

        self._label = tk.Label(
            self._root,
            text="",
            font=(_FONT_FAMILY, _FONT_SIZE),
            fg=_FG_COLOR,
            bg=_BG_COLOR,
            wraplength=win_w - _PADDING_X * 2,
            justify="center",
            padx=_PADDING_X,
            pady=_PADDING_Y,
        )
        self._label.pack(fill="both", expand=True)

        # Posicionar: centrado horizontalmente, pegado abajo
        self._root.update_idletasks()
        win_h = self._root.winfo_reqheight()
        x = (screen_w - win_w) // 2
        y = screen_h - win_h - _BOTTOM_MARGIN
        self._root.geometry(f"{win_w}x{win_h}+{x}+{y}")

        # Click derecho o Escape para cerrar
        self._root.bind("<Escape>", lambda _: self._handle_close())
        self._root.bind("<Button-3>", lambda _: self._handle_close())

        # Permitir arrastrar la ventana
        self._root.bind("<Button-1>", self._start_drag)
        self._root.bind("<B1-Motion>", self._on_drag)

        self._root.protocol("WM_DELETE_WINDOW", self._handle_close)
        logger.info("Overlay de subtitulos iniciado (%dx en +%d+%d)", win_w, x, y)

    def _start_drag(self, event: tk.Event) -> None:
        self._drag_x = event.x
        self._drag_y = event.y

    def _on_drag(self, event: tk.Event) -> None:
        if self._root is None:
            return
        x = self._root.winfo_x() + event.x - self._drag_x
        y = self._root.winfo_y() + event.y - self._drag_y
        self._root.geometry(f"+{x}+{y}")

    def _handle_close(self) -> None:
        if self._on_close:
            self._on_close()
        self.close()

    def update_text(self, text: str) -> None:
        """Actualiza el texto del subtitulo.

        Thread-safe: usa after() para encolar en el hilo de tkinter.
        """
        if self._root is None or self._label is None:
            return
        self._root.after(0, self._set_text, text)

    def _set_text(self, text: str) -> None:
        if self._label is None or self._root is None:
            return
        self._label.configure(text=text)
        # Reajustar altura tras cambio de texto
        self._root.update_idletasks()
        screen_h = self._root.winfo_screenheight()
        win_h = self._root.winfo_reqheight()
        x = self._root.winfo_x()
        y = screen_h - win_h - _BOTTOM_MARGIN
        win_w = self._root.winfo_width()
        self._root.geometry(f"{win_w}x{win_h}+{x}+{y}")

    def is_running(self) -> bool:
        return self._root is not None

    def close(self) -> None:
        """Cierra la ventana overlay."""
        if self._root is not None:
            with contextlib.suppress(tk.TclError):
                self._root.destroy()
            self._root = None
            self._label = None
            logger.info("Overlay cerrado")

    def mainloop_step(self, interval_ms: int = 50) -> None:
        """Procesa eventos de tkinter sin bloquear.

        Llamar periodicamente desde el event loop de asyncio.
        """
        if self._root is None:
            return
        try:
            self._root.update()
        except tk.TclError:
            self._root = None
            self._label = None


async def run_overlay_loop(overlay: SubtitleOverlay) -> None:
    """Mantiene el overlay vivo procesando eventos de tkinter en async."""
    while overlay.is_running():
        overlay.mainloop_step()
        await asyncio.sleep(0.05)
