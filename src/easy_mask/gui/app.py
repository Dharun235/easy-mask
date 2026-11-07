"""Create the Qt application and show the main window."""

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from easy_mask.gui.main_window import MainWindow, Settings


def run(path: Path | None, settings: Settings) -> int:
    """Start the GUI, optionally opening ``path``. Blocks until the window closes."""
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("easy-mask")
    window = MainWindow(settings)
    window.show()
    if path is not None:
        window.open_path(path)
    return app.exec()
