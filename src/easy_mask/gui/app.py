"""Create the Qt application and show the main window."""

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from easy_mask.gui.main_window import MainWindow, Settings
from easy_mask.gui.theme import apply_theme


def run(path: Path | None, settings: Settings) -> int:
    """Start the GUI, optionally opening ``path``. Blocks until the window closes."""
    existing = QApplication.instance()
    app = existing if isinstance(existing, QApplication) else QApplication(sys.argv)
    app.setApplicationName("easy-mask")
    apply_theme(app)  # before the window, so icons are tinted for the right palette
    window = MainWindow(settings)
    window.show()
    if path is not None:
        window.open_path(path)
    return app.exec()
