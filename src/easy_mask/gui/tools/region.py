"""Click and box tools that select a region of similar pixels: magic wand, GrabCut, superpixels.

All three follow the same convention (see :func:`combine_with_selection`): a plain click or
drag makes a new mask, Shift adds to the selected shape, Alt subtracts from it.
"""

from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QInputEvent, QKeyEvent, QPainter, QPen

from easy_mask.core import pixel_ops
from easy_mask.core.geometry import Point
from easy_mask.gui.tools.base import CanvasTool, combine_with_selection, mask_image
from easy_mask.gui.tools.brush import StrokeCallback

if TYPE_CHECKING:
    from easy_mask.gui.canvas import Canvas

PREVIEW_RGB = (0, 200, 255)


class MagicWandTool(CanvasTool):
    """Click to flood-fill a region of similar colour. ``[`` / ``]`` change the tolerance."""

    def __init__(self, on_result: StrokeCallback, tolerance: int = 32):
        self.on_result = on_result
        self.tolerance = tolerance

    def press(self, canvas: "Canvas", p: Point, event: QInputEvent) -> None:
        """Select the region around the clicked pixel."""
        region = pixel_ops.magic_wand(canvas.image_rgb(), p, self.tolerance)
        edit = combine_with_selection(canvas, region, event.modifiers())
        if edit is not None:
            self.on_result(*edit)

    def key(self, canvas: "Canvas", event: QKeyEvent) -> bool:
        """``[`` / ``]`` lower / raise the colour tolerance."""
        if event.text() not in ("[", "]"):
            return False
        step = 4 if event.text() == "]" else -4
        self.tolerance = min(max(self.tolerance + step, 0), 255)
        canvas.status_changed.emit(f"wand tolerance {self.tolerance}")
        return True


class GrabCutTool(CanvasTool):
    """Drag a box around an object; GrabCut separates it from the background."""

    def __init__(self, on_result: StrokeCallback, iterations: int = 5):
        self.on_result = on_result
        self.iterations = iterations
        self._start: Point | None = None
        self._end: Point | None = None
        self._modifiers = Qt.KeyboardModifier.NoModifier

    def press(self, canvas: "Canvas", p: Point, event: QInputEvent) -> None:
        """Start the box."""
        self._start = self._end = p
        self._modifiers = event.modifiers()

    def move(self, canvas: "Canvas", p: Point, event: QInputEvent) -> None:
        """Stretch the box while dragging."""
        if self._start is not None:
            self._end = p

    def release(self, canvas: "Canvas", p: Point, event: QInputEvent) -> None:
        """Run GrabCut in the finished box."""
        if self._start is None:
            return
        (x0, y0), (x1, y1) = self._start, p
        self.cancel(canvas)
        box = (round(min(x0, x1)), round(min(y0, y1)), round(max(x0, x1)), round(max(y0, y1)))
        canvas.setCursor(Qt.CursorShape.WaitCursor)
        try:
            region = pixel_ops.grabcut(canvas.image_rgb(), box, iterations=self.iterations)
        finally:
            canvas.setCursor(self.cursor)
        edit = combine_with_selection(canvas, region, self._modifiers)
        if edit is not None:
            self.on_result(*edit)

    def paint(self, canvas: "Canvas", painter: QPainter) -> None:
        """Draw the box being dragged."""
        if self._start is None or self._end is None:
            return
        pen = QPen(QColor(*PREVIEW_RGB), 2, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(QRectF(QPointF(*self._start), QPointF(*self._end)).normalized())

    def cancel(self, canvas: "Canvas") -> None:
        """Drop the box in progress."""
        self._start = self._end = None


class SuperpixelTool(CanvasTool):
    """Click or drag across superpixels to pick them. ``[`` / ``]`` change their size.

    Superpixels are computed once per image (and size) and outlined while the tool is on.
    """

    def __init__(self, on_result: StrokeCallback, size: int = 20):
        self.on_result = on_result
        self.size = size
        self._labels: np.ndarray | None = None
        self._labels_for: tuple[int, int] = (0, 0)  # (id of image array, size) of the cache
        self._outline: QImage | None = None
        self._points: list[Point] = []
        self._modifiers = Qt.KeyboardModifier.NoModifier

    def labels(self, canvas: "Canvas") -> np.ndarray:
        """Superpixel labels of the canvas image, computed on first use."""
        rgb = canvas.image_rgb()
        if self._labels is None or self._labels_for != (id(rgb), self.size):
            self._labels = pixel_ops.superpixels(rgb, self.size)
            self._labels_for = (id(rgb), self.size)
            self._outline = None
        return self._labels

    def press(self, canvas: "Canvas", p: Point, event: QInputEvent) -> None:
        """Start picking superpixels."""
        self._points = [p]
        self._modifiers = event.modifiers()

    def move(self, canvas: "Canvas", p: Point, event: QInputEvent) -> None:
        """Pick every superpixel the cursor passes over (gaps between events filled in)."""
        if self._points:
            (x0, y0), (x1, y1) = self._points[-1], p
            steps = max(1, int(max(abs(x1 - x0), abs(y1 - y0)) / 2))
            self._points += [
                (x0 + (x1 - x0) * i / steps, y0 + (y1 - y0) * i / steps)
                for i in range(1, steps + 1)
            ]

    def release(self, canvas: "Canvas", p: Point, event: QInputEvent) -> None:
        """Turn the picked superpixels into a mask edit."""
        if not self._points:
            return
        picked = pixel_ops.pick_superpixels(self.labels(canvas), self._points)
        self._points = []
        edit = combine_with_selection(canvas, picked, self._modifiers)
        if edit is not None:
            self.on_result(*edit)

    def key(self, canvas: "Canvas", event: QKeyEvent) -> bool:
        """``[`` / ``]`` make superpixels smaller / bigger."""
        if event.text() not in ("[", "]"):
            return False
        step = 5 if event.text() == "]" else -5
        self.size = min(max(self.size + step, 5), 200)
        canvas.status_changed.emit(f"superpixel size {self.size}px")
        return True

    def paint(self, canvas: "Canvas", painter: QPainter) -> None:
        """Outline the superpixels and shade the ones picked so far."""
        labels = self.labels(canvas)
        if self._outline is None:
            self._outline = mask_image(pixel_ops.label_boundaries(labels), (255, 255, 0), 110)
        painter.drawImage(QPointF(0, 0), self._outline)
        if self._points:
            picked = pixel_ops.pick_superpixels(labels, self._points)
            painter.drawImage(QPointF(0, 0), mask_image(picked, PREVIEW_RGB, 120))

    def cancel(self, canvas: "Canvas") -> None:
        """Drop the pick in progress."""
        self._points = []
