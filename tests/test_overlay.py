from unittest.mock import MagicMock, patch

from autosubs.overlay import SubtitleOverlay


class TestSubtitleOverlay:
    def test_initial_state(self):
        overlay = SubtitleOverlay()
        assert overlay._root is None
        assert overlay._label is None
        assert overlay.is_running() is False

    def test_close_without_start(self):
        overlay = SubtitleOverlay()
        overlay.close()  # No deberia lanzar error

    def test_update_text_without_start(self):
        overlay = SubtitleOverlay()
        overlay.update_text("texto")  # No deberia lanzar error

    def test_mainloop_step_without_start(self):
        overlay = SubtitleOverlay()
        overlay.mainloop_step()  # No deberia lanzar error

    def test_is_running_after_close(self):
        overlay = SubtitleOverlay()
        overlay._root = MagicMock()
        assert overlay.is_running() is True
        overlay._root = None
        assert overlay.is_running() is False

    @patch("autosubs.overlay.tk.Tk")
    def test_start_creates_window(self, mock_tk_class):
        mock_root = MagicMock()
        mock_root.winfo_screenwidth.return_value = 1920
        mock_root.winfo_screenheight.return_value = 1080
        mock_root.winfo_reqheight.return_value = 80
        mock_tk_class.return_value = mock_root

        overlay = SubtitleOverlay()
        overlay.start()

        assert overlay._root is mock_root
        mock_root.overrideredirect.assert_called_once_with(True)
        mock_root.attributes.assert_any_call("-topmost", True)
        assert overlay.is_running() is True

        overlay.close()

    @patch("autosubs.overlay.tk.Tk")
    def test_update_text_calls_after(self, mock_tk_class):
        mock_root = MagicMock()
        mock_root.winfo_screenwidth.return_value = 1920
        mock_root.winfo_screenheight.return_value = 1080
        mock_root.winfo_reqheight.return_value = 80
        mock_tk_class.return_value = mock_root

        overlay = SubtitleOverlay()
        overlay.start()
        overlay.update_text("nuevo texto")

        mock_root.after.assert_called()
        overlay.close()

    @patch("autosubs.overlay.tk.Tk")
    def test_on_close_callback(self, mock_tk_class):
        mock_root = MagicMock()
        mock_root.winfo_screenwidth.return_value = 1920
        mock_root.winfo_screenheight.return_value = 1080
        mock_root.winfo_reqheight.return_value = 80
        mock_tk_class.return_value = mock_root

        callback = MagicMock()
        overlay = SubtitleOverlay()
        overlay.start(on_close=callback)
        overlay._handle_close()

        callback.assert_called_once()
