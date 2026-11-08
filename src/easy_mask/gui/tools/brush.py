"""Brush and eraser: paint pixels into a mask shape.

* Brush with a shape selected paints into it (a polygon/rectangle/circle becomes a mask).
  Brush with nothing selected starts a new mask; its label is asked when the stroke ends.
* Eraser paints the area to remove; when the stroke ends it is cut from every mask under
  it (only from the selected shapes, if any), exactly like a cut, so Propagate carries it.

``[`` and ``]`` change the brush size.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QInputEvent, QKeyEvent, QPainter, QPen

from easy_mask.core import masks
from easy_mask.core.geometry import Point
from easy_mask.core.labels import label_color
from easy_mask.core.rasterize import shape_to_mask
from easy_mask.gui.tools.base import CanvasTool, mask_image

if TYPE_CHECKING:
    from easy_mask.gui.canvas import Canvas

MIN_RADIUS, MAX_RADIUS = 1, 200
NEW_MASK_RGB = (0, 200, 255)
ERASE_RGB = (255, 80, 60)

# Called when a stroke ends with the edited full-size mask and the index of the shape it
# replaces (None for a brand-new mask). The eraser passes the painted area and None.
StrokeCallback = Callable[[np.ndarray, int | None], None]


class BrushTool(CanvasTool):
    """Round brush (or eraser) working on a full-size boolean mask during a stroke."""

    cursor = Qt.CursorShape.BlankCursor  # we draw our own circle cursor

    def __init__(self, on_stroke: StrokeCallback, erase: bool = False, radius: int = 8):
        self.on_stroke = on_stroke
        self.erase = erase
        self.radius = radius
        self.hidden_index: int | None = None  # shape being edited; canvas skips drawing it
        self._full: np.ndarray | None = None
        self._last: Point | None = None
        self._cursor: Point | None = None
        self._rgb = NEW_MASK_RGB
        self._box: list[int] = [0, 0, 0, 0]  # x0, y0, x1, y1 (exclusive) of changed pixels

    def press(self, canvas: "Canvas", p: Point, event: QInputEvent) -> None:
        """Start a stroke on the selected shape, or on a new empty mask."""
        height, width = canvas.pixmap.height(), canvas.pixmap.width()
        index = canvas.selected
        if self.erase:  # paint the area to remove; the main window cuts it out
            self._full = np.zeros((height, width), dtype=bool)
            self._rgb = ERASE_RGB
            self._box = [int(p[0]), int(p[1]), int(p[0]) + 1, int(p[1]) + 1]
            self.hidden_index = None
        elif index is not None and canvas.shapes[index].shape_type.is_area:
            shape = canvas.shapes[index]
            self._full = shape_to_mask(shape, height, width)
            self._rgb = label_color(shape.label, canvas.labels)
            x0, y0, x1, y1 = shape.bounding_box()
            self._box = [int(x0), int(y0), int(x1) + 2, int(y1) + 2]
            self.hidden_index = index
        else:
            self._full = np.zeros((height, width), dtype=bool)
            self._rgb = NEW_MASK_RGB
            self._box = [int(p[0]), int(p[1]), int(p[0]) + 1, int(p[1]) + 1]
            self.hidden_index = None
        self._last = p
        self._stroke(p)

    def move(self, canvas: "Canvas", p: Point, event: QInputEvent) -> None:
        """Extend the stroke while the button is held; always track the cursor."""
        self._cursor = p
        if self._full is not None:
            self._stroke(p)

    def release(self, canvas: "Canvas", p: Point, event: QInputEvent) -> None:
        """Hand the finished mask to the main window."""
        if self._full is None:
            return
        full, index = self._full, self.hidden_index
        self.cancel(canvas)
        self.on_stroke(full, index)

    def key(self, canvas: "Canvas", event: QKeyEvent) -> bool:
        """``[`` / ``]`` shrink / grow the brush."""
        step = max(1, self.radius // 5)  # ~20% steps, at least one pixel
        if event.text() == "[":
            self.radius = max(MIN_RADIUS, self.radius - step)
        elif event.text() == "]":
            self.radius = min(MAX_RADIUS, self.radius + step)
        else:
            return False
        canvas.status_changed.emit(f"brush size {self.radius}px")
        return True

    def paint(self, canvas: "Canvas", painter: QPainter) -> None:
        """Draw the stroke in progress and a circle showing the brush size."""
        if self._full is not None:
            x0, y0, x1, y1 = self._clipped_box()
            if x1 > x0 and y1 > y0:
                crop = self._full[y0:y1, x0:x1]
                painter.drawImage(QPointF(x0, y0), mask_image(crop, self._rgb, 150))
        if self._cursor is not None:
            pen = QPen(QColor(255, 80, 80) if self.erase else QColor(255, 255, 255), 1)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            cx, cy = self._cursor
            r = self.radius
            painter.drawEllipse(QRectF(cx - r, cy - r, 2 * r, 2 * r))

    def cancel(self, canvas: "Canvas") -> None:
        """Drop the stroke in progress."""
        self._full = None
        self._last = None
        self.hidden_index = None

    def _stroke(self, p: Point) -> None:
        assert self._full is not None and self._last is not None
        masks.paint_stroke(self._full, self._last, p, self.radius, value=True)
        r = self.radius + 1
        box = self._box
        box[0] = min(box[0], int(min(p[0], self._last[0])) - r)
        box[1] = min(box[1], int(min(p[1], self._last[1])) - r)
        box[2] = max(box[2], int(max(p[0], self._last[0])) + r + 1)
        box[3] = max(box[3], int(max(p[1], self._last[1])) + r + 1)
        self._last = p

    def _clipped_box(self) -> tuple[int, int, int, int]:
        assert self._full is not None
        height, width = self._full.shape
        x0, y0, x1, y1 = self._box
        return max(x0, 0), max(y0, 0), min(x1, width), min(y1, height)
