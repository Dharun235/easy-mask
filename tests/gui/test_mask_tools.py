"""Brush, eraser, multi-select and the Mask menu, driven through the real window."""

import numpy as np
from PySide6.QtCore import Qt

from easy_mask.core import masks
from easy_mask.core.shapes import ShapeType
from easy_mask.gui.label_dialog import LabelInfo


def draw_rect(window, ui, a, b):
    window.set_tool(ShapeType.RECTANGLE)
    ui.drag(a, b)
    window.set_tool(None)


def full(window, index=0):
    return masks.to_full(window.canvas.shapes[index], 80, 100)


def test_brush_with_nothing_selected_creates_labelled_mask(window, ui):
    window.ask_label = lambda known, initial, parent: LabelInfo("dog")
    window.set_tool("brush")
    ui.drag((20, 40), (60, 40))
    (shape,) = window.canvas.shapes
    assert (shape.label, shape.shape_type) == ("dog", ShapeType.MASK)
    assert full(window)[40, 20:61].all()
    assert window.canvas.selected == 0 and window.dirty


def test_brush_cancelled_label_adds_nothing(window, ui):
    window.ask_label = lambda known, initial, parent: None
    window.set_tool("brush")
    ui.drag((20, 40), (60, 40))
    assert window.canvas.shapes == []


def test_brush_adds_to_selected_rectangle_and_converts_it(window, ui):
    draw_rect(window, ui, (10, 10), (30, 30))
    ui.click(20, 20)
    window.set_tool("brush")
    ui.drag((30, 20), (60, 20))
    shape = window.canvas.shapes[0]
    assert shape.shape_type == ShapeType.MASK and shape.label == "cat"
    assert full(window)[20, 15] and full(window)[20, 55]


def test_eraser_removes_pixels_and_whole_shape(window, ui):
    draw_rect(window, ui, (10, 10), (30, 30))
    ui.click(20, 20)
    window.set_tool("eraser")
    window.pixel_tools["eraser"].radius = 3
    ui.drag((5, 20), (35, 20))
    assert not full(window)[20, 20] and full(window)[12, 20]
    window.pixel_tools["eraser"].radius = 50
    ui.click(20, 20)
    assert window.canvas.shapes == []


def test_eraser_without_selection_does_nothing(window, ui):
    window.set_tool("eraser")
    ui.drag((5, 20), (35, 20))
    assert window.canvas.shapes == [] and not window.dirty


def test_brush_size_keys(window, ui):
    window.set_tool("brush")
    start = window.pixel_tools["brush"].radius
    ui.qtbot.keyClicks(window.canvas, "]")
    assert window.pixel_tools["brush"].radius > start
    ui.qtbot.keyClicks(window.canvas, "[[[[[[[[[[[[[[")
    assert window.pixel_tools["brush"].radius == 1


def test_stroke_is_undoable(window, ui):
    window.set_tool("brush")
    ui.drag((20, 40), (60, 40))
    window.undo()
    assert window.canvas.shapes == []


def test_shift_click_multi_select_and_union(window, ui):
    draw_rect(window, ui, (10, 10), (30, 30))
    draw_rect(window, ui, (25, 10), (50, 30))
    ui.click(20, 20)
    ui.qtbot.mouseClick(
        window.canvas,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.ShiftModifier,
        ui.screen(45, 20),
    )
    assert window.canvas.selection == [0, 1]
    assert window.shape_list.selected_rows() == [0, 1]
    window.combine("union")
    (merged,) = window.canvas.shapes
    assert merged.shape_type == ShapeType.MASK and full(window)[20, 12] and full(window)[20, 48]
    window.undo()
    assert len(window.canvas.shapes) == 2


def test_combine_needs_two_shapes(window, ui):
    draw_rect(window, ui, (10, 10), (30, 30))
    ui.click(20, 20)
    window.combine("subtract")
    assert window.canvas.shapes[0].shape_type == ShapeType.RECTANGLE


def test_dragging_multi_selection_moves_all(window, ui):
    draw_rect(window, ui, (10, 10), (20, 20))
    draw_rect(window, ui, (40, 10), (50, 20))
    window.canvas.set_selection([0, 1])
    ui.drag((15, 15), (15, 35))
    assert window.canvas.shapes[0].points[0][1] > 25 and window.canvas.shapes[1].points[0][1] > 25


def test_shape_list_extended_selection_drives_canvas(window, ui):
    draw_rect(window, ui, (10, 10), (20, 20))
    draw_rect(window, ui, (40, 10), (50, 20))
    window.shape_list.item(0).setSelected(True)
    window.shape_list.item(1).setSelected(True)
    assert window.canvas.selection == [0, 1]


def test_delete_removes_whole_selection(window, ui):
    draw_rect(window, ui, (10, 10), (20, 20))
    draw_rect(window, ui, (40, 10), (50, 20))
    window.canvas.set_selection([0, 1])
    window.delete_selected()
    assert window.canvas.shapes == []


def test_cleanup_and_conversion_menu_actions(window, ui):
    draw_rect(window, ui, (10, 10), (40, 40))
    window.canvas.select(0)
    window.convert_to_masks()
    assert window.canvas.shapes[0].shape_type == ShapeType.MASK
    window.canvas.select(0)
    before = full(window).sum()
    window.map_selected(lambda f: f | np.roll(f, 2, axis=1))
    assert full(window).sum() > before
    window.canvas.select(0)
    window.ask_number = lambda *a: 2.0
    window.convert_to_polygons()
    assert window.canvas.shapes[0].shape_type == ShapeType.POLYGON


def test_remove_specks_and_cancel(window, ui):
    window.set_tool("brush")
    window.pixel_tools["brush"].radius = 1
    ui.drag((10, 10), (11, 10))
    window.set_tool(None)
    window.canvas.select(0)
    window.ask_number = lambda *a: None  # cancelled
    window.remove_specks()
    assert len(window.canvas.shapes) == 1
    window.ask_number = lambda *a: 1000.0
    window.remove_specks()
    assert window.canvas.shapes == []


def test_mask_shapes_save_reload_and_paint(window, ui, image_dir):
    window.set_tool("brush")
    ui.drag((20, 40), (60, 40))
    window.set_tool(None)
    window.save()
    window.next_image()
    window.prev_image()
    assert window.canvas.shapes[0].shape_type == ShapeType.MASK
    window.canvas.select(0)
    image = window.canvas.grab().toImage()
    assert not image.isNull()
    assert np.array_equal(full(window)[40, 20:61], np.ones(41, bool))
