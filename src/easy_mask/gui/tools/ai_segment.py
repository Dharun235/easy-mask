"""AI Segment: click an object and a model (EfficientSAM) outlines it.

* Left-click: a point on the object. Right-click or Alt+click: a point on the background.
* Drag: a box around the object (replaces the previous box).
* The mask preview updates after every prompt. Enter or double-click accepts it as a new
  object; prompts made with Shift held add it to the selected shape instead.
* Esc clears the prompts, Backspace removes the last one.

The image is encoded in the background (see :mod:`~easy_mask.gui.tools.ai_worker`);
clicks made meanwhile are kept and the preview appears when encoding finishes.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QThreadPool, Signal
from PySide6.QtGui import (
    QColor,
    QImage,
    QInputEvent,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
)

from easy_mask.ai.segmenter import NEGATIVE, POSITIVE, Box
from easy_mask.core.geometry import Point
from easy_mask.core.rasterize import shape_to_mask
from easy_mask.gui.tools.ai_worker import EncodeJob, LazySegmenter, PrefetchJob, SegmenterLoader
from easy_mask.gui.tools.base import CanvasTool, mask_image
from easy_mask.gui.tools.brush import StrokeCallback

if TYPE_CHECKING:
    from easy_mask.gui.canvas import Canvas

PREVIEW_RGB = (30, 144, 255)
DRAG_THRESHOLD_PX = 4  # screen pixels a press must move to become a box
DOT_RADIUS_PX = 5
READY_HINT = (
    "AI Segment: click the object (right/Alt-click = background, drag = box), "
    "Enter = accept, Esc = clear"
)


def pixmap_to_rgb(pixmap: QPixmap) -> np.ndarray:
    """A pixmap as an ``(H, W, 3)`` uint8 RGB array (copied, safe to use on any thread)."""
    image = pixmap.toImage().convertToFormat(QImage.Format.Format_RGB888)
    height, width = image.height(), image.width()
    rows = np.frombuffer(image.constBits(), np.uint8).reshape(height, image.bytesPerLine())
    return rows[:, : 3 * width].reshape(height, width, 3).copy()


VISIBLE_PRIORITY = 1  # QThreadPool runs higher priorities first
PREFETCH_PRIORITY = 0


class AiSegmentTool(QObject, CanvasTool):
    """Collects click/box prompts, shows the predicted mask, hands it over on accept.

    Signals:
        status(str): Progress and hints for the status bar.
        error(str): Loading or encoding failed (e.g. onnxruntime missing, offline).
        encoded(): The shown image is ready (a good moment to prefetch the next ones).
        busy(str, int, int): Background work for the shown image: text, done, total
            (total 0 = unknown); drives the progress bar.
        idle(): That work finished or failed.
    """

    status = Signal(str)
    error = Signal(str)
    encoded = Signal()
    busy = Signal(str, int, int)
    idle = Signal()
    right_button = True

    def __init__(
        self,
        on_accept: StrokeCallback,
        load: SegmenterLoader,
        image_key: Callable[[], str | None],
    ):
        """Create the tool.

        Args:
            on_accept: Receives the accepted full-size mask and the index of the shape it
                replaces (None for a new mask), like the brush.
            load: Builds the segmenter; called once, on a worker thread.
            image_key: Identifies the current image (its path) for the embedding cache.
        """
        super().__init__()
        self.on_accept = on_accept
        self.image_key = image_key
        self.lazy = LazySegmenter(load)
        self.points: list[Point] = []
        self.labels: list[int] = []
        self.box: Box | None = None
        self.mask: np.ndarray | None = None  # current preview, full image size
        self.ready = False  # the current image is encoded
        self._order: list[str] = []  # "point" / "box", so Backspace undoes the latest
        self.add_to_selection = False  # a Shift+prompt: Enter adds to the selected shape
        self._canvas: Canvas | None = None
        self._generation = 0  # bumped per encode request; only the newest result counts
        self._jobs: dict[int, EncodeJob] = {}  # keeps running jobs (and signals) alive
        self._pool = QThreadPool()
        self._pool.setMaxThreadCount(1)
        self._press: Point | None = None
        self._press_negative = False
        self._drag_end: Point | None = None
        self._preview: QImage | None = None

    # ------------------------------------------------------------------ encoding

    def activate(self, canvas: "Canvas") -> None:
        """Start encoding the shown image in the background; prompts start empty."""
        self._canvas = canvas
        self.clear()
        self.ready = False
        if canvas.pixmap.isNull():
            return
        self._generation += 1
        key = self.image_key() or f"pixmap-{canvas.pixmap.cacheKey()}"
        job = EncodeJob(self.lazy, pixmap_to_rgb(canvas.pixmap), key, self._generation)
        job.signals.progress.connect(self.status.emit)
        job.signals.download.connect(
            lambda name, done, total: self.busy.emit(f"Downloading AI model ({name})", done, total)
        )
        job.signals.done.connect(self._on_encoded)
        job.signals.failed.connect(lambda message, g=self._generation: self._on_failed(g, message))
        self._jobs[self._generation] = job
        if self.lazy.segmenter is None:
            self.status.emit("AI Segment: loading model...")
            self.busy.emit("Loading AI model", 0, 0)
        elif not self.lazy.segmenter.is_cached(key):
            self.busy.emit("Preparing image for AI", 0, 0)
        self._pool.start(job, VISIBLE_PRIORITY)  # ahead of any queued prefetch

    def prefetch(self, paths: list[str]) -> None:
        """Encode these images in the background so AI Segment is instant on them.

        Does nothing until the model has been loaded once (no surprise downloads).
        """
        segmenter = self.lazy.segmenter
        if segmenter is None:
            return
        for path in paths:
            if not segmenter.is_cached(path):
                self._pool.start(PrefetchJob(segmenter, path), PREFETCH_PRIORITY)

    def _on_encoded(self, generation: int) -> None:
        self._jobs.pop(generation, None)
        if generation != self._generation:
            return  # an older image; a newer job is queued behind it
        self.idle.emit()
        self.ready = True
        self.status.emit(READY_HINT)
        self._predict()
        self.encoded.emit()

    def _on_failed(self, generation: int, message: str) -> None:
        self._jobs.pop(generation, None)
        if generation == self._generation:
            self.idle.emit()
            self.error.emit(message)

    def wait(self, timeout_ms: int = -1) -> bool:
        """Block until background jobs finish (for tests and shutdown)."""
        return self._pool.waitForDone(timeout_ms)

    # ------------------------------------------------------------------ prompts

    def press(self, canvas: "Canvas", p: Point, event: QInputEvent) -> None:
        """Remember where a click or box drag starts and whether it is negative."""
        alt = bool(event.modifiers() & Qt.KeyboardModifier.AltModifier)
        right = isinstance(event, QMouseEvent) and event.button() == Qt.MouseButton.RightButton
        self._press, self._press_negative, self._drag_end = p, alt or right, None
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self.add_to_selection = True

    def move(self, canvas: "Canvas", p: Point, event: QInputEvent) -> None:
        """Show a box once a positive press has been dragged far enough."""
        if self._press is None or self._press_negative:
            return
        dx, dy = p[0] - self._press[0], p[1] - self._press[1]
        if max(abs(dx), abs(dy)) * canvas.scale > DRAG_THRESHOLD_PX:
            self._drag_end = p

    def release(self, canvas: "Canvas", p: Point, event: QInputEvent) -> None:
        """Finish the prompt: a box if dragged, otherwise a point."""
        if self._press is None:
            return
        start, end = self._press, self._drag_end
        self._press = self._drag_end = None
        if end is not None:
            self.box = (start[0], start[1], end[0], end[1])
            self._order = [kind for kind in self._order if kind != "box"] + ["box"]
        else:
            self.points.append(start)
            self.labels.append(NEGATIVE if self._press_negative else POSITIVE)
            self._order.append("point")
        self._predict()

    def double_click(self, canvas: "Canvas", p: Point, event: QInputEvent) -> None:
        """Accept the mask.

        The double-click's first click already added a point; drop it unless it is the
        only prompt (a lone double-click segments and accepts in one go).
        """
        if len(self._order) > 1 and self._order[-1] == "point" and self.points[-1] == p:
            self.remove_last()
        self.accept()

    def key(self, canvas: "Canvas", event: QKeyEvent) -> bool:
        """Enter accepts, Backspace removes the last prompt, Esc clears all prompts."""
        key = event.key()
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.accept()
            return True
        if not self._order:
            return False  # nothing to undo: let the canvas deselect / delete as usual
        if key == Qt.Key.Key_Backspace:
            self.remove_last()
            return True
        if key == Qt.Key.Key_Escape:
            self.clear()
            return True
        return False

    def remove_last(self) -> None:
        """Undo the most recent point or box."""
        if self._order.pop() == "box":
            self.box = None
        else:
            self.points.pop()
            self.labels.pop()
        self._predict()

    def clear(self) -> None:
        """Drop every prompt and the preview."""
        self.points, self.labels, self.box, self._order = [], [], None, []
        self._press = self._drag_end = None
        self.add_to_selection = False
        self._set_mask(None)

    def cancel(self, canvas: "Canvas") -> None:
        """The tool is switched off: drop prompts (the embedding stays cached)."""
        self.clear()

    # ------------------------------------------------------------------ results

    def _predict(self) -> None:
        segmenter = self.lazy.segmenter
        if not self.ready or segmenter is None or not self._order:
            self._set_mask(None)
            return
        predictions = segmenter.predict(self.points, self.labels, self.box)
        self._set_mask(predictions[0][0] if predictions else None)

    def _set_mask(self, mask: np.ndarray | None) -> None:
        self.mask = mask
        self._preview = None if mask is None else mask_image(mask, PREVIEW_RGB, 120)
        if self._canvas is not None:
            self._canvas.update()

    def accept(self) -> None:
        """Hand the preview mask to the main window: a new mask, or added to the selection."""
        canvas = self._canvas
        if canvas is None or self.mask is None or not self.mask.any():
            return
        full = self.mask
        index = canvas.selected if self.add_to_selection else None
        if index is not None and canvas.shapes[index].shape_type.is_area:
            full = full | shape_to_mask(canvas.shapes[index], *full.shape)
        else:
            index = None
        self.clear()
        self.on_accept(full, index)
        if index is None:
            canvas.select(None)  # the next object starts a new mask, not added to this one

    # ------------------------------------------------------------------ drawing

    def paint(self, canvas: "Canvas", painter: QPainter) -> None:
        """Draw the mask preview, the box and the clicked points."""
        if self._preview is not None:
            painter.drawImage(QPointF(0, 0), self._preview)
        pen = QPen(QColor(255, 255, 0), 2, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        boxes = [self.box] if self.box else []
        if self._press is not None and self._drag_end is not None:
            boxes.append((*self._press, *self._drag_end))
        for x0, y0, x1, y1 in boxes:
            painter.drawRect(QRectF(QPointF(x0, y0), QPointF(x1, y1)).normalized())
        outline = QPen(QColor(255, 255, 255), 1)
        outline.setCosmetic(True)
        painter.setPen(outline)
        r = DOT_RADIUS_PX / canvas.scale
        for (x, y), label in zip(self.points, self.labels, strict=True):
            painter.setBrush(QColor(0, 200, 0) if label == POSITIVE else QColor(230, 0, 0))
            painter.drawEllipse(QPointF(x, y), r, r)
