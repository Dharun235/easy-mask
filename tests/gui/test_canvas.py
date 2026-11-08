"""Canvas interaction details: zoom, pan, keys and drawing edge cases."""

import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QPixmap

from easy_mask.core.shapes import ShapeType
from easy_mask.gui.canvas import MAX_SCALE, MIN_SCALE, Canvas


def test_empty_canvas_paints_and_ignores_input(qtbot):
    canvas = Canvas()
    qtbot.addWidget(canvas)
    canvas.resize(300, 200)
    canvas.show()
    qtbot.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=QPoint(50, 50))
    canvas.zoom_by(2.0)
    canvas.fit_to_window()
    assert not canvas.grab().isNull()  # placeholder text painted without errors


def test_ctrl_wheel_zooms_plain_wheel_pans(qtbot, window, ui):
    canvas = window.canvas
    canvas.zoom_by(1.0)  # any zoom stops auto-fit, so a late resize can't change the scale
    scale, offset = canvas.scale, QPointF(canvas.offset)
    pos = ui.screen(50, 40)
    from PySide6.QtGui import QWheelEvent

    def wheel(dy, modifiers):
        event = QWheelEvent(
            QPointF(pos),
            QPointF(canvas.mapToGlobal(pos)),
            QPoint(0, 0),
            QPoint(0, dy),
            Qt.MouseButton.NoButton,
            modifiers,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        )
        canvas.wheelEvent(event)

    wheel(120, Qt.KeyboardModifier.ControlModifier)
    assert canvas.scale == pytest.approx(scale * 1.15)
    zoomed_offset = QPointF(canvas.offset)
    wheel(120, Qt.KeyboardModifier.NoModifier)
    assert canvas.scale == pytest.approx(scale * 1.15)
    assert canvas.offset.y() == pytest.approx(zoomed_offset.y() + 60)
    assert offset != canvas.offset


def test_middle_drag_pans(window, ui):
    before = QPointF(window.canvas.offset)
    ui.drag((50, 40), (60, 40), button=Qt.MouseButton.MiddleButton)
    assert window.canvas.offset.x() > before.x()


def test_zoom_is_clamped(window):
    window.canvas.zoom_by(1e9)
    assert window.canvas.scale == MAX_SCALE
    window.canvas.zoom_by(1e-9)
    assert window.canvas.scale == MIN_SCALE


def test_actual_size_and_fit(window):
    window.canvas.zoom_to_actual_size()
    assert window.canvas.scale == pytest.approx(1.0)
    window.canvas.fit_to_window()
    assert window.canvas.scale > 1.0  # 100x80 image in a large window


def test_points_are_clamped_to_image(window, ui):
    window.set_tool(ShapeType.POINT)
    ui.click(-20, 500)
    assert window.canvas.shapes[0].points == [(0.0, 80.0)]


def test_escape_cancels_draft_then_deselects(window, ui):
    window.set_tool(ShapeType.POLYGON)
    ui.click(10, 10)
    ui.click(20, 10)
    assert window.canvas.is_drawing()
    ui.key(Qt.Key.Key_Escape)
    assert not window.canvas.is_drawing() and window.canvas.shapes == []


def test_enter_with_too_few_points_does_nothing(window, ui):
    window.set_tool(ShapeType.POLYGON)
    ui.click(10, 10)
    ui.click(20, 10)
    ui.key(Qt.Key.Key_Return)
    assert window.canvas.shapes == [] and window.canvas.is_drawing()


def test_double_click_finishes_line_strip(window, ui):
    window.set_tool(ShapeType.LINESTRIP)
    ui.click(10, 10)
    ui.click(30, 30)
    # A real double-click arrives as press/release then a double-click event.
    ui.click(50, 10)
    ui.double_click(50, 10)
    assert window.canvas.shapes[0].shape_type == ShapeType.LINESTRIP
    assert len(window.canvas.shapes[0].points) == 3


def test_zero_size_rectangle_is_ignored(window, ui):
    window.set_tool(ShapeType.RECTANGLE)
    ui.click(20, 20)
    ui.click(20, 20)
    assert window.canvas.shapes == []


def test_switching_tool_discards_draft(window, ui):
    window.set_tool(ShapeType.POLYGON)
    ui.click(10, 10)
    window.set_tool(ShapeType.RECTANGLE)
    assert not window.canvas.is_drawing()


@pytest.mark.parametrize("mode", [t for t in ShapeType if t != ShapeType.MASK])
def test_draft_preview_paints_for_every_tool(window, ui, mode):
    window.set_tool(mode)
    if mode == ShapeType.POINT:
        return  # a point completes on the first click; nothing to preview
    ui.click(10, 10)
    ui.qtbot.mouseMove(window.canvas, ui.screen(40, 30))
    assert not window.canvas.grab().isNull()


def test_fill_toggle_and_new_image_refits(window):
    window._toggle_fill(False)
    assert window.canvas.fill is False
    window.canvas.zoom_by(3.0)
    window.canvas.set_image(QPixmap(window.image_paths[1].as_posix()))
    assert window.canvas.scale == pytest.approx(
        0.98 * min(window.canvas.width() / 100, window.canvas.height() / 80)
    )
