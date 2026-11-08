"""The drawing surface: shows the image, paints shapes, turns mouse input into edits.

The canvas owns a *view transform* (``scale`` + ``offset``) mapping image pixels to widget
pixels. Everything stored in shapes is in image pixels; widget coordinates never leave
this module.

Responsibilities are split with the main window like this:

* The canvas handles drawing a new shape point by point, and dragging existing shapes or
  vertices. It emits :attr:`Canvas.shape_drawn` with an *unlabelled* shape when drawing
  finishes, and :attr:`Canvas.shapes_edited` after a drag.
* The main window asks for a label, appends the shape, records undo history, and
  refreshes the side panels.
"""

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QPixmap,
    QPolygonF,
    QResizeEvent,
    QWheelEvent,
)
from PySide6.QtWidgets import QWidget

from easy_mask.core.geometry import Point
from easy_mask.core.labels import label_color
from easy_mask.core.shapes import Shape, ShapeType
from easy_mask.gui.tools.base import CanvasTool, mask_image

HIT_TOLERANCE_PX = 6  # how close (in screen pixels) a click must be to grab something
VERTEX_SIZE_PX = 7
DRAG_THRESHOLD_PX = 4  # press-drag-release shorter than this counts as a click
ZOOM_STEP = 1.15
MIN_SCALE, MAX_SCALE = 0.02, 64.0
DRAFT_RGB = (0, 200, 255)
BACKGROUND = QColor(40, 40, 44)


class Canvas(QWidget):
    """Image view with shape drawing and editing.

    Signals:
        shape_drawn(Shape): A new shape was completed; its label is empty.
        shapes_edited(): Existing shapes were moved or reshaped by dragging.
        selection_changed(list): Selected shape indices, in the order they were picked.
        double_clicked_shape(int): A shape was double-clicked in edit mode.
        delete_requested(): User pressed Backspace while not drawing.
        status_changed(str): Cursor position / zoom text for the status bar.
    """

    shape_drawn = Signal(object)
    shapes_edited = Signal()
    selection_changed = Signal(list)
    double_clicked_shape = Signal(int)
    delete_requested = Signal()
    status_changed = Signal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(200, 200)

        self.pixmap = QPixmap()
        self.shapes: list[Shape] = []
        self.hidden: set[int] = set()  # id() of shapes the user hid
        self.selection: list[int] = []  # selected indices, in pick order
        self.fill = True
        self.mode: ShapeType | None = None  # None means edit (select/move) mode
        self.tool: CanvasTool | None = None  # pixel tool (brush, eraser, ...) if active
        self.labels: list[str] = []  # label order decides colours; set by the main window

        self.scale = 1.0
        self.offset = QPointF(0, 0)
        self._auto_fit = True  # keep fitting on resize until the user zooms or pans

        self._draft: list[Point] = []  # points of the shape being drawn
        self._draft_press: QPointF | None = None  # screen pos where a 2-point draft began
        self._cursor: Point | None = None  # last mouse position, image coords

        self._drag_vertex: int | None = None
        self._drag_last: Point | None = None
        self._dragged = False
        self._pan_last: QPointF | None = None

    # ------------------------------------------------------------------ public API

    def set_image(self, pixmap: QPixmap) -> None:
        """Show a new image and fit it to the window."""
        self.pixmap = pixmap
        self.cancel_draft()
        self.fit_to_window()

    def set_shapes(self, shapes: list[Shape]) -> None:
        """Replace all shapes (e.g. after loading or undo). Clears selection."""
        self.shapes = shapes
        self.hidden.clear()
        self.select(None)
        self.update()

    def set_mode(self, mode: ShapeType | None) -> None:
        """Switch to drawing ``mode``, or to edit mode with ``None``."""
        self._drop_tool()
        self.cancel_draft()
        self.mode = mode
        if mode is not None:
            self.select(None)
        self.setCursor(Qt.CursorShape.CrossCursor if mode else Qt.CursorShape.ArrowCursor)

    def set_tool(self, tool: CanvasTool) -> None:
        """Activate a pixel tool (brush, eraser, ...). Selection is kept: tools act on it."""
        self._drop_tool()
        self.cancel_draft()
        self.mode = None
        self.tool = tool
        self.setCursor(tool.cursor)
        self.update()

    def _drop_tool(self) -> None:
        if self.tool is not None:
            self.tool.cancel(self)
            self.tool = None

    @property
    def selected(self) -> int | None:
        """The most recently picked selected shape (the one edits apply to), if any."""
        return self.selection[-1] if self.selection else None

    def select(self, index: int | None) -> None:
        """Select only the shape at ``index`` (or nothing) and notify listeners."""
        self.set_selection([] if index is None else [index])

    def set_selection(self, indices: list[int]) -> None:
        """Select several shapes; the first is the target of combine operations."""
        if indices == self.selection:
            return
        self.selection = list(indices)
        self.selection_changed.emit(list(self.selection))
        self.update()

    def toggle_selected(self, index: int) -> None:
        """Add ``index`` to the selection, or remove it if already selected."""
        if index in self.selection:
            self.set_selection([i for i in self.selection if i != index])
        else:
            self.set_selection([*self.selection, index])

    def set_shape_visible(self, index: int, visible: bool) -> None:
        """Show or hide one shape without deleting it."""
        key = id(self.shapes[index])
        if visible:
            self.hidden.discard(key)
        else:
            self.hidden.add(key)
        self.update()

    def is_drawing(self) -> bool:
        """True while a shape is half-drawn."""
        return bool(self._draft)

    def cancel_draft(self) -> None:
        """Throw away the half-drawn shape."""
        self._draft = []
        self._draft_press = None
        self.update()

    def finish_draft(self) -> None:
        """Complete the half-drawn polygon / line strip (bound to Enter)."""
        if self.mode is None or len(self._draft) < self.mode.min_points:
            return
        shape = Shape(label="", shape_type=self.mode, points=list(self._draft))
        self.cancel_draft()
        if shape.shape_type.fixed_points == 2 and shape.points[0] == shape.points[1]:
            return  # zero-size rectangle / circle / line: ignore
        self.shape_drawn.emit(shape)

    def fit_to_window(self) -> None:
        """Scale and centre the image so it fits the widget."""
        if self.pixmap.isNull():
            return
        iw, ih = self.pixmap.width(), self.pixmap.height()
        self.scale = 0.98 * min(self.width() / iw, self.height() / ih)
        self.offset = QPointF(
            (self.width() - iw * self.scale) / 2, (self.height() - ih * self.scale) / 2
        )
        self._auto_fit = True
        self._emit_status()
        self.update()

    def zoom_by(self, factor: float, anchor: QPointF | None = None) -> None:
        """Zoom by ``factor`` keeping the image point under ``anchor`` fixed."""
        if self.pixmap.isNull():
            return
        anchor = anchor if anchor is not None else QPointF(self.width() / 2, self.height() / 2)
        ix, iy = self.to_image(anchor)
        self.scale = min(max(self.scale * factor, MIN_SCALE), MAX_SCALE)
        self.offset = QPointF(anchor.x() - ix * self.scale, anchor.y() - iy * self.scale)
        self._auto_fit = False
        self._emit_status()
        self.update()

    def zoom_to_actual_size(self) -> None:
        """Show the image at 100% (one image pixel per screen pixel)."""
        self.zoom_by(1.0 / self.scale)

    # ------------------------------------------------------------------ coordinates

    def to_image(self, pos: QPointF) -> Point:
        """Convert a widget position to image pixel coordinates."""
        return (pos.x() - self.offset.x()) / self.scale, (pos.y() - self.offset.y()) / self.scale

    def _clamp(self, p: Point) -> Point:
        """Keep a point inside the image rectangle."""
        return (
            min(max(p[0], 0.0), float(self.pixmap.width())),
            min(max(p[1], 0.0), float(self.pixmap.height())),
        )

    def _tolerance(self) -> float:
        """Hit tolerance converted to image pixels at the current zoom."""
        return HIT_TOLERANCE_PX / self.scale

    # ------------------------------------------------------------------ mouse input

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Start panning, add a draft point, or grab a shape/vertex."""
        if self.pixmap.isNull():
            return
        pos = event.position()
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_last = pos
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        p = self._clamp(self.to_image(pos))
        if self.tool is not None:
            self.tool.press(self, p, event)
            self.update()
        elif self.mode is None:
            add = event.modifiers() & (
                Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier
            )
            self._press_edit(p, add_to_selection=bool(add))
        else:
            self._press_draw(p, pos)

    def _press_draw(self, p: Point, pos: QPointF) -> None:
        mode = self.mode
        if mode == ShapeType.POINT:
            self._draft = [p]
            self.finish_draft()
        elif mode in (ShapeType.POLYGON, ShapeType.LINESTRIP):
            near_start = (
                mode == ShapeType.POLYGON
                and len(self._draft) >= 3
                and Shape("", ShapeType.POINT, [self._draft[0]]).hit(p, self._tolerance())
            )
            if near_start:
                self.finish_draft()  # clicking the first vertex closes the polygon
            else:
                self._draft.append(p)
        elif not self._draft:  # rectangle / circle / line: first corner
            self._draft = [p]
            self._draft_press = pos
        else:  # second click completes it
            self._draft.append(p)
            self.finish_draft()
        self.update()

    def _press_edit(self, p: Point, add_to_selection: bool = False) -> None:
        tol = self._tolerance()
        if add_to_selection:  # Shift/Ctrl+click: toggle membership, never drag
            for index in self._hit_order():
                if self.shapes[index].hit(p, tol):
                    self.toggle_selected(index)
                    return
            return
        # Vertices win over bodies so small shapes stay editable; selected shape first.
        for index in self._hit_order():
            vertex = self.shapes[index].nearest_vertex(p, tol)
            if vertex is not None:
                self.select(index)
                self._drag_vertex, self._drag_last = vertex, p
                return
        for index in self._hit_order():
            if self.shapes[index].hit(p, tol):
                if index not in self.selection:
                    self.select(index)  # clicking inside a multi-selection drags all of it
                self._drag_last = p
                return
        self.select(None)

    def _hit_order(self) -> list[int]:
        """Visible shape indices, selected first, then topmost (last drawn) first."""
        order = [
            i for i in reversed(range(len(self.shapes))) if id(self.shapes[i]) not in self.hidden
        ]
        for index in reversed(self.selection):
            if index in order:
                order.remove(index)
                order.insert(0, index)
        return order

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Pan, drag the grabbed shape/vertex, or update the draft preview."""
        if self.pixmap.isNull():
            return
        pos = event.position()
        if self._pan_last is not None:
            self.offset += pos - self._pan_last
            self._pan_last = pos
            self._auto_fit = False
            self.update()
            return

        p = self._clamp(self.to_image(pos))
        self._cursor = p
        if self.tool is not None:
            self.tool.move(self, p, event)
        elif self._drag_last is not None and self.selected is not None:
            if self._drag_vertex is not None:
                self.shapes[self.selected].points[self._drag_vertex] = p
            else:
                dx, dy = p[0] - self._drag_last[0], p[1] - self._drag_last[1]
                for index in self.selection:
                    self.shapes[index].translate(dx, dy)
            self._drag_last = p
            self._dragged = True
        self._emit_status()
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Finish a pan, a drag edit, or a press-drag-release rectangle/circle/line."""
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_last = None
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self.tool is not None:
            self.tool.release(self, self._clamp(self.to_image(event.position())), event)
            self.update()
            return
        if self._dragged:
            self.shapes_edited.emit()
        self._drag_vertex, self._drag_last, self._dragged = None, None, False

        # Dragging out a 2-point shape finishes it on release; a plain click waits for a
        # second click instead. Both feel natural, so we support both.
        if self._draft_press is not None and len(self._draft) == 1:
            moved = (event.position() - self._draft_press).manhattanLength()
            if moved > DRAG_THRESHOLD_PX:
                self._draft.append(self._clamp(self.to_image(event.position())))
                self.finish_draft()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        """Finish a polygon / line strip, or request a label edit in edit mode."""
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self.mode in (ShapeType.POLYGON, ShapeType.LINESTRIP):
            self.finish_draft()
        elif self.mode is None and not self.pixmap.isNull():
            self._press_edit(self._clamp(self.to_image(event.position())))
            self._drag_last = self._drag_vertex = None  # a double-click never drags
            if self.selected is not None:
                self.double_clicked_shape.emit(self.selected)

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Ctrl+wheel zooms at the cursor; plain wheel / trackpad scroll pans."""
        if self.pixmap.isNull():
            return
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            steps = event.angleDelta().y() / 120.0
            self.zoom_by(ZOOM_STEP**steps, event.position())
        else:
            delta = (
                event.pixelDelta() if not event.pixelDelta().isNull() else event.angleDelta() / 2
            )
            self.offset += QPointF(delta)
            self._auto_fit = False
            self.update()
        event.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Enter finishes, Backspace removes a point (or deletes), Esc cancels."""
        if self.tool is not None and self.tool.key(self, event):
            self.update()
            return
        key = event.key()
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.finish_draft()
        elif key == Qt.Key.Key_Backspace:
            if self._draft:
                self._draft.pop()
                self.update()
            else:
                self.delete_requested.emit()
        elif key == Qt.Key.Key_Escape:
            if self._draft:
                self.cancel_draft()
            else:
                self.select(None)
        else:
            super().keyPressEvent(event)

    def resizeEvent(self, event: QResizeEvent) -> None:
        """Re-fit the image on resize unless the user has zoomed or panned."""
        if self._auto_fit:
            self.fit_to_window()
        super().resizeEvent(event)

    # ------------------------------------------------------------------ painting

    def paintEvent(self, event: QPaintEvent) -> None:
        """Draw the image, every visible shape, and the draft preview."""
        painter = QPainter(self)
        painter.fillRect(self.rect(), BACKGROUND)
        if self.pixmap.isNull():
            painter.setPen(QColor(170, 170, 170))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Open an image (Ctrl+O) or a folder (Ctrl+Shift+O)",
            )
            return

        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        # Smooth when shrinking; crisp pixels when zoomed in so edges are visible.
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self.scale < 1.0)
        painter.translate(self.offset)
        painter.scale(self.scale, self.scale)
        painter.drawPixmap(0, 0, self.pixmap)

        editing = getattr(self.tool, "hidden_index", None)  # shape a tool is redrawing
        for index, shape in enumerate(self.shapes):
            if id(shape) not in self.hidden and index != editing:
                self._paint_shape(painter, shape, selected=index in self.selection)
        self._paint_draft(painter)
        if self.tool is not None:
            self.tool.paint(self, painter)

    def _paint_shape(
        self,
        painter: QPainter,
        shape: Shape,
        selected: bool,
        rgb: tuple[int, int, int] | None = None,
    ) -> None:
        r, g, b = rgb or label_color(shape.label, self.labels)
        pen = QPen(QColor(r, g, b), 3 if selected else 2)
        pen.setCosmetic(True)  # constant on-screen width at any zoom
        painter.setPen(pen)
        if self.fill and shape.shape_type.is_area:
            painter.setBrush(QColor(r, g, b, 120 if selected else 60))
        else:
            painter.setBrush(Qt.BrushStyle.NoBrush)

        if shape.shape_type == ShapeType.MASK:
            self._paint_mask(painter, shape, (r, g, b), selected)
            return
        pts = [QPointF(x, y) for x, y in shape.points]
        match shape.shape_type:
            case ShapeType.POLYGON:
                painter.drawPolygon(QPolygonF(pts))
            case ShapeType.RECTANGLE:
                painter.drawRect(QRectF(pts[0], pts[1]).normalized())
            case ShapeType.CIRCLE:
                radius = shape.radius()
                painter.drawEllipse(pts[0], radius, radius)
            case ShapeType.LINE | ShapeType.LINESTRIP:
                painter.drawPolyline(QPolygonF(pts))
        self._paint_vertices(painter, pts, QColor(r, g, b), selected)

    def _paint_mask(
        self, painter: QPainter, shape: Shape, rgb: tuple[int, int, int], selected: bool
    ) -> None:
        if shape.mask is None:
            return
        x1, y1 = round(shape.points[0][0]), round(shape.points[0][1])
        painter.drawImage(QPointF(x1, y1), mask_image(shape.mask, rgb, 150 if selected else 100))
        if selected:  # dashed bounding box so a selected mask is easy to spot
            pen = QPen(QColor(255, 255, 255), 1, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            height, width = shape.mask.shape
            painter.drawRect(QRectF(x1, y1, width, height))

    def _paint_vertices(
        self, painter: QPainter, pts: list[QPointF], color: QColor, selected: bool
    ) -> None:
        size = VERTEX_SIZE_PX / self.scale * (1.4 if selected else 1.0)
        pen = QPen(QColor(255, 255, 255) if selected else color, 1)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(color if selected else QColor(255, 255, 255, 200))
        for pt in pts:
            painter.drawRect(QRectF(pt.x() - size / 2, pt.y() - size / 2, size, size))

    def _paint_draft(self, painter: QPainter) -> None:
        if not self._draft or self.mode is None:
            return
        points = list(self._draft)
        if self._cursor is not None:
            points.append(self._cursor)
        if self.mode in (ShapeType.POLYGON, ShapeType.LINESTRIP):
            preview = Shape("", ShapeType.LINESTRIP, points)  # open outline while drawing
        else:
            preview = Shape("", self.mode, points[:2])
        if len(preview.points) >= preview.shape_type.min_points:
            self._paint_shape(painter, preview, selected=False, rgb=DRAFT_RGB)
        else:
            self._paint_vertices(painter, [QPointF(*points[0])], QColor(*DRAFT_RGB), False)

    def _emit_status(self) -> None:
        text = f"zoom {self.scale * 100:.0f}%"
        if self._cursor is not None:
            text = f"x {self._cursor[0]:.0f}, y {self._cursor[1]:.0f}   {text}"
        self.status_changed.emit(text)
