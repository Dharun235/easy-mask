"""Watershed tool: scribble on the object (and optionally the background), press Enter.

* Drag: object marker (green). Alt+drag: background marker (red). Without background
  markers the image border is used.
* Enter: run watershed and make a new mask. Esc: clear the markers.
* ``[`` / ``]``: thinner / thicker markers.
"""

from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QInputEvent, QKeyEvent, QPainter

from easy_mask.core import masks, pixel_ops
from easy_mask.core.geometry import Point
from easy_mask.gui.tools.base import CanvasTool, mask_image
from easy_mask.gui.tools.brush import StrokeCallback

if TYPE_CHECKING:
    from easy_mask.gui.canvas import Canvas

OBJECT_RGB, BACKGROUND_RGB = (0, 220, 0), (230, 40, 40)


class WatershedTool(CanvasTool):
    """Collects object / background marker strokes, then floods them with watershed."""

    def __init__(self, on_result: StrokeCallback, radius: int = 3):
        self.on_result = on_result
        self.radius = radius
        self.foreground: np.ndarray | None = None
        self.background: np.ndarray | None = None
        self._last: Point | None = None
        self._paint_background = False

    def press(self, canvas: "Canvas", p: Point, event: QInputEvent) -> None:
        """Start a marker stroke (background with Alt)."""
        if self.foreground is None or self.background is None:
            size = (canvas.pixmap.height(), canvas.pixmap.width())
            self.foreground = np.zeros(size, dtype=bool)
            self.background = np.zeros(size, dtype=bool)
        self._paint_background = bool(event.modifiers() & Qt.KeyboardModifier.AltModifier)
        self._last = p
        self.move(canvas, p, event)

    def move(self, canvas: "Canvas", p: Point, event: QInputEvent) -> None:
        """Extend the marker stroke while the button is held."""
        if self._last is None or self.foreground is None or self.background is None:
            return
        target = self.background if self._paint_background else self.foreground
        masks.paint_stroke(target, self._last, p, self.radius, value=True)
        self._last = p

    def release(self, canvas: "Canvas", p: Point, event: QInputEvent) -> None:
        """End the marker stroke."""
        self._last = None

    def key(self, canvas: "Canvas", event: QKeyEvent) -> bool:
        """Enter runs watershed, Esc clears markers, ``[`` / ``]`` resize the marker."""
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.run(canvas)
        elif event.key() == Qt.Key.Key_Escape and self.foreground is not None:
            self.cancel(canvas)
        elif event.text() in ("[", "]"):
            self.radius = min(max(self.radius + (1 if event.text() == "]" else -1), 1), 50)
            canvas.status_changed.emit(f"marker size {self.radius}px")
        else:
            return False
        return True

    def run(self, canvas: "Canvas") -> None:
        """Flood the markers and hand a new mask to the main window."""
        if self.foreground is None or not self.foreground.any():
            canvas.status_changed.emit("draw object markers first")
            return
        region = pixel_ops.watershed(canvas.image_rgb(), self.foreground, self.background)
        self.cancel(canvas)
        self.on_result(region, None)

    def paint(self, canvas: "Canvas", painter: QPainter) -> None:
        """Draw the markers."""
        if self.foreground is not None and self.background is not None:
            painter.drawImage(QPointF(0, 0), mask_image(self.foreground, OBJECT_RGB, 200))
            painter.drawImage(QPointF(0, 0), mask_image(self.background, BACKGROUND_RGB, 200))

    def cancel(self, canvas: "Canvas") -> None:
        """Clear all markers."""
        self.foreground = self.background = None
        self._last = None
