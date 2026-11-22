"""Base class for canvas tools, plus helpers between numpy arrays and Qt images."""

from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QInputEvent, QKeyEvent, QPainter, QPixmap

from easy_mask.core.geometry import Point

if TYPE_CHECKING:  # avoid a runtime import cycle: the canvas imports this module
    from easy_mask.gui.canvas import Canvas


class CanvasTool:
    """A pixel tool. Override the events you need; every default does nothing.

    Points are in image pixel coordinates, already clamped to the image.
    """

    cursor = Qt.CursorShape.CrossCursor
    right_button = False  # True: the canvas also forwards right-button press/release

    def activate(self, canvas: "Canvas") -> None:
        """The tool was switched on, or a new image was shown while it is on."""

    def press(self, canvas: "Canvas", p: Point, event: QInputEvent) -> None:
        """Left mouse button pressed."""

    def move(self, canvas: "Canvas", p: Point, event: QInputEvent) -> None:
        """Mouse moved (with or without a button held)."""

    def release(self, canvas: "Canvas", p: Point, event: QInputEvent) -> None:
        """Left mouse button released."""

    def double_click(self, canvas: "Canvas", p: Point, event: QInputEvent) -> None:
        """Left button double-clicked (after the press/release of the first click)."""

    def key(self, canvas: "Canvas", event: QKeyEvent) -> bool:
        """Key pressed. Return True if handled (the canvas then ignores it)."""
        return False

    def paint(self, canvas: "Canvas", painter: QPainter) -> None:
        """Draw an overlay. The painter is already in image coordinates."""

    def cancel(self, canvas: "Canvas") -> None:
        """The tool is being switched off; drop any half-finished work."""


def mask_image(mask: np.ndarray, rgb: tuple[int, int, int], alpha: int) -> QImage:
    """A boolean mask as a translucent coloured image (transparent where False)."""
    height, width = mask.shape
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[mask] = (*rgb, alpha)
    image = QImage(rgba.data, width, height, 4 * width, QImage.Format.Format_RGBA8888)
    return image.copy()  # detach from the numpy buffer, which is freed on return


def pixmap_to_rgb(pixmap: QPixmap) -> np.ndarray:
    """A pixmap's pixels as an ``(height, width, 3)`` uint8 RGB array."""
    image = pixmap.toImage().convertToFormat(QImage.Format.Format_RGB888)
    height, width = image.height(), image.width()
    rows = np.frombuffer(image.constBits(), np.uint8).reshape(height, image.bytesPerLine())
    return rows[:, : 3 * width].reshape(height, width, 3).copy()


def rgb_to_pixmap(rgb: np.ndarray) -> QPixmap:
    """An ``(height, width, 3)`` uint8 RGB array as a pixmap."""
    height, width = rgb.shape[:2]
    data = np.ascontiguousarray(rgb)
    image = QImage(data.data, width, height, 3 * width, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(image.copy())
