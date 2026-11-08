"""Toolbar icons: small SVGs shipped in ``gui/icons/``."""

from pathlib import Path

from PySide6.QtGui import QIcon

ICON_DIR = Path(__file__).parent / "icons"


def icon(name: str) -> QIcon:
    """The icon ``gui/icons/<name>.svg`` (an empty icon if the file is missing)."""
    path = ICON_DIR / f"{name}.svg"
    return QIcon(str(path)) if path.is_file() else QIcon()
