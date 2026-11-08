"""Cut: drag a lasso around a part to remove it from the masks underneath.

No label is needed: the outlined pixels are cut from every mask under the lasso, whatever
its label (with shapes selected, only from those objects). The last cut can then be carried
through the next frames (Video > Cut Through Frames).
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QInputEvent, QKeyEvent, QPainter, QPen, QPolygonF

from easy_mask.core.geometry import Point
from easy_mask.core.rasterize import shape_to_mask
from easy_mask.core.shapes import Shape, ShapeType
from easy_mask.gui.tools.base import CanvasTool

if TYPE_CHECKING:
    from easy_mask.gui.canvas import Canvas

MIN_POINTS = 3
LASSO_RGB = (255, 80, 60)

CutCallback = Callable[[np.ndarray], None]  # called with the full-size region to cut


def lasso_region(points: list[Point], height: int, width: int) -> np.ndarray:
    """The pixels inside the closed outline ``points`` as a full-size bool mask."""
    return shape_to_mask(Shape("cut", ShapeType.POLYGON, points), height, width)


class LassoCutTool(CanvasTool):
    """Press, drag around the part to cut, release. ``Esc`` drops the lasso."""

    def __init__(self, on_cut: CutCallback):
        self.on_cut = on_cut
        self.points: list[Point] = []

    def press(self, canvas: "Canvas", p: Point, event: QInputEvent) -> None:
        """Start a new lasso."""
        self.points = [p]

    def move(self, canvas: "Canvas", p: Point, event: QInputEvent) -> None:
        """Extend the lasso while the button is held."""
        if self.points and p != self.points[-1]:
            self.points.append(p)
            canvas.update()

    def release(self, canvas: "Canvas", p: Point, event: QInputEvent) -> None:
        """Close the lasso and cut what it encloses."""
        points, self.points = self.points, []
        canvas.update()
        if len(points) < MIN_POINTS:
            return
        region = lasso_region(points, canvas.pixmap.height(), canvas.pixmap.width())
        if region.any():
            self.on_cut(region)

    def key(self, canvas: "Canvas", event: QKeyEvent) -> bool:
        """``Esc`` drops the lasso being drawn."""
        if event.key() == Qt.Key.Key_Escape and self.points:
            self.cancel(canvas)
            return True
        return False

    def paint(self, canvas: "Canvas", painter: QPainter) -> None:
        """The lasso outline, closed back to its start."""
        if len(self.points) < 2:
            return
        pen = QPen(QColor(*LASSO_RGB), 0, Qt.PenStyle.DashLine)  # cosmetic: 1 px at any zoom
        painter.setPen(pen)
        painter.setBrush(QColor(*LASSO_RGB, 50))
        painter.drawPolygon(QPolygonF([QPointF(x, y) for x, y in self.points]))

    def cancel(self, canvas: "Canvas") -> None:
        """Drop the half-drawn lasso."""
        self.points = []
        canvas.update()
