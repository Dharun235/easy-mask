"""Base class for canvas tools, plus a helper to draw boolean masks as overlays."""

from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QInputEvent, QKeyEvent, QPainter

from easy_mask.core.geometry import Point

if TYPE_CHECKING:  # avoid a runtime import cycle: the canvas imports this module
    from easy_mask.gui.canvas import Canvas


class CanvasTool:
    """A pixel tool. Override the events you need; every default does nothing.

    Points are in image pixel coordinates, already clamped to the image.
    """

    cursor = Qt.CursorShape.CrossCursor

    def press(self, canvas: "Canvas", p: Point, event: QInputEvent) -> None:
        """Left mouse button pressed."""

    def move(self, canvas: "Canvas", p: Point, event: QInputEvent) -> None:
        """Mouse moved (with or without a button held)."""

    def release(self, canvas: "Canvas", p: Point, event: QInputEvent) -> None:
        """Left mouse button released."""

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
