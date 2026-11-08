"""Toolbar icons: Lucide line icons (ISC licence, see ``icons/LICENSE-lucide.txt``).

Lucide draws with ``currentColor``; we render each icon in the palette's text colour, so
icons stay crisp and readable in both light and dark mode.
"""

from pathlib import Path

from PySide6.QtCore import QByteArray, Qt
from PySide6.QtGui import QColor, QGuiApplication, QIcon, QPainter, QPalette, QPixmap
from PySide6.QtSvg import QSvgRenderer

ICON_DIR = Path(__file__).parent / "icons"
SIZES = (16, 24, 32, 48)  # rendered once each so the icon is sharp at any toolbar size


def icon(name: str, color: QColor | None = None) -> QIcon:
    """The icon ``icons/<name>.svg`` tinted ``color`` (default: the palette's text colour).

    Returns an empty icon if the file is missing.
    """
    path = ICON_DIR / f"{name}.svg"
    if not path.is_file():
        return QIcon()
    if color is None:
        color = QGuiApplication.palette().color(QPalette.ColorRole.WindowText)
    svg = path.read_text(encoding="utf-8").replace("currentColor", color.name())
    renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    result = QIcon()
    for size in SIZES:
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        renderer.render(painter)
        painter.end()
        result.addPixmap(pixmap)
    return result
