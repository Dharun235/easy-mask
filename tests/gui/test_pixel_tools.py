"""Magic wand, GrabCut, superpixels, watershed, threshold panel and display adjustments,
driven through the real window."""

import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialogButtonBox

from easy_mask.core import masks
from easy_mask.core.display import DisplaySettings
from easy_mask.core.shapes import Shape, ShapeType
from easy_mask.gui.label_dialog import LabelInfo
from easy_mask.gui.tools.base import pixmap_to_rgb, rgb_to_pixmap

SHIFT, ALT = Qt.KeyboardModifier.ShiftModifier, Qt.KeyboardModifier.AltModifier
H, W = 80, 100


def scene() -> np.ndarray:
    """Gray image with a red block (x 30-59, y 20-49) and a blue block (x 70-89, y 5-19)."""
    rgb = np.full((H, W, 3), 60, np.uint8)
    rgb[20:50, 30:60] = (220, 30, 30)
    rgb[5:20, 70:90] = (30, 30, 220)
    return rgb


RED = np.zeros((H, W), bool)
RED[20:50, 30:60] = True
BLUE = np.zeros((H, W), bool)
BLUE[5:20, 70:90] = True


@pytest.fixture
def window(make_window, tmp_path):
    folder = tmp_path / "scene"
    folder.mkdir()
    Image.fromarray(scene()).save(folder / "a.png")
    Image.fromarray(scene()[:, ::-1].copy()).save(folder / "b.png")
    win = make_window(label=LabelInfo("obj"))
    win.open_path(folder)
    return win


@pytest.fixture
def ui(driver, window):
    return driver(window)


def full(window, index=0):
    return masks.to_full(window.canvas.shapes[index], H, W)


def click(ui, x, y, modifier=Qt.KeyboardModifier.NoModifier):
    ui.qtbot.mouseClick(ui.window.canvas, Qt.MouseButton.LeftButton, modifier, ui.screen(x, y))


def test_pixmap_rgb_round_trip(qtbot):
    rgb = scene()[:, :97]  # odd width: rows are padded inside QImage
    assert np.array_equal(pixmap_to_rgb(rgb_to_pixmap(rgb)), rgb)


def test_canvas_image_rgb_is_the_loaded_image(window):
    assert np.array_equal(window.canvas.image_rgb(), scene())


# ---------------------------------------------------------------------- magic wand


def test_wand_click_creates_labelled_mask(window, ui):
    window.set_tool("wand")
    click(ui, 40, 30)
    (shape,) = window.canvas.shapes
    assert (shape.label, shape.shape_type) == ("obj", ShapeType.MASK)
    assert np.array_equal(full(window), RED)
    assert window.canvas.selected == 0 and window.dirty


def test_wand_shift_adds_and_alt_subtracts(window, ui):
    window.set_tool("wand")
    click(ui, 40, 30)
    click(ui, 80, 10, SHIFT)
    assert len(window.canvas.shapes) == 1
    assert np.array_equal(full(window), RED | BLUE)
    click(ui, 80, 10, ALT)
    assert np.array_equal(full(window), RED)
    window.undo()
    assert np.array_equal(full(window), RED | BLUE)


def test_wand_plain_click_with_selection_makes_second_mask(window, ui):
    window.set_tool("wand")
    click(ui, 40, 30)
    click(ui, 80, 10)
    assert len(window.canvas.shapes) == 2 and np.array_equal(full(window, 1), BLUE)


def test_wand_alt_without_selection_does_nothing(window, ui):
    window.set_tool("wand")
    click(ui, 40, 30, ALT)
    assert window.canvas.shapes == [] and not window.dirty


def test_wand_shift_on_non_area_shape_makes_new_mask(window, ui):
    window.canvas.set_shapes([Shape("line", ShapeType.LINE, [(0, 0), (5, 5)])])
    window.canvas.select(0)
    window.set_tool("wand")
    click(ui, 40, 30, SHIFT)
    assert len(window.canvas.shapes) == 2 and np.array_equal(full(window, 1), RED)


def test_wand_cancelled_label_adds_nothing(window, ui):
    window.ask_label = lambda known, initial, parent: None
    window.set_tool("wand")
    click(ui, 40, 30)
    assert window.canvas.shapes == []


def test_wand_tolerance_keys(window, ui):
    window.set_tool("wand")
    wand = window.pixel_tools["wand"]
    start = wand.tolerance
    ui.qtbot.keyClicks(window.canvas, "]")
    assert wand.tolerance == start + 4
    ui.qtbot.keyClicks(window.canvas, "[" * 100)
    assert wand.tolerance == 0
    assert "tolerance 0" in window.status_label.text()
    ui.key(Qt.Key.Key_Escape)  # not handled by the wand: canvas deselects


# ---------------------------------------------------------------------- GrabCut


def test_grabcut_box_selects_object(window, ui):
    window.set_tool("grabcut")
    ui.drag((22, 12), (68, 58))
    assert len(window.canvas.shapes) == 1
    assert masks.iou(full(window), RED) > 0.95


def test_grabcut_shift_box_adds_to_selection(window, ui):
    window.set_tool("grabcut")
    ui.drag((22, 12), (68, 58))
    c = window.canvas
    ui.qtbot.mousePress(c, Qt.MouseButton.LeftButton, SHIFT, ui.screen(64, 1))
    ui.qtbot.mouseMove(c, ui.screen(80, 12))
    ui.qtbot.mouseMove(c, ui.screen(96, 24))
    ui.qtbot.mouseRelease(c, Qt.MouseButton.LeftButton, SHIFT, ui.screen(96, 24))
    assert len(c.shapes) == 1 and masks.iou(full(window), RED | BLUE) > 0.9


def test_grabcut_click_without_box_does_nothing(window, ui):
    window.set_tool("grabcut")
    click(ui, 40, 30)
    assert window.canvas.shapes == []


def test_grabcut_draws_box_while_dragging(window, ui):
    window.set_tool("grabcut")
    tool = window.pixel_tools["grabcut"]
    ui.qtbot.mousePress(window.canvas, Qt.MouseButton.LeftButton, pos=ui.screen(10, 10))
    ui.qtbot.mouseMove(window.canvas, ui.screen(50, 50))
    assert tool._end == pytest.approx((50, 50), abs=1)
    assert not window.canvas.grab().isNull()
    window.set_tool(None)  # switching tools drops the box
    assert tool._start is None


# ---------------------------------------------------------------------- superpixels


def test_superpixel_click_picks_region_inside_object(window, ui):
    window.set_tool("superpixel")
    click(ui, 45, 35)
    picked = full(window)
    assert picked[35, 45] and not (picked & ~RED).any()


def test_superpixel_drag_picks_several(window, ui):
    window.set_tool("superpixel")
    window.pixel_tools["superpixel"].size = 10
    ui.drag((32, 22), (57, 47))
    labels = window.pixel_tools["superpixel"].labels(window.canvas)
    diagonal = {labels[i + 22, i + 32] for i in range(26)}  # every label along the drag
    assert np.array_equal(full(window), np.isin(labels, list(diagonal)))


def test_superpixel_size_keys_recompute_labels(window, ui):
    window.set_tool("superpixel")
    tool = window.pixel_tools["superpixel"]
    before = len(np.unique(tool.labels(window.canvas)))
    ui.qtbot.keyClicks(window.canvas, "[")
    assert tool.size == 15 and "15px" in window.status_label.text()
    assert len(np.unique(tool.labels(window.canvas))) > before
    ui.qtbot.keyClicks(window.canvas, "]" * 60)
    assert tool.size == 200
    ui.key(Qt.Key.Key_Escape)


def test_superpixel_paint_and_cancel(window, ui):
    window.set_tool("superpixel")
    tool = window.pixel_tools["superpixel"]
    ui.qtbot.mousePress(window.canvas, Qt.MouseButton.LeftButton, pos=ui.screen(40, 30))
    assert not window.canvas.grab().isNull()  # outlines + pick preview
    window.set_tool(None)
    assert tool._points == []


def test_superpixels_recomputed_for_next_image(window, ui):
    window.set_tool("superpixel")
    tool = window.pixel_tools["superpixel"]
    first = tool.labels(window.canvas)
    window.next_image()
    assert tool.labels(window.canvas) is not first


def test_tool_release_without_press_is_ignored(window, ui):
    for name in ("superpixel", "grabcut", "watershed"):
        window.set_tool(name)
        ui.qtbot.mouseRelease(window.canvas, Qt.MouseButton.LeftButton, pos=ui.screen(5, 5))
        ui.qtbot.mouseMove(window.canvas, ui.screen(6, 6))
    assert window.canvas.shapes == []


# ---------------------------------------------------------------------- watershed


def test_watershed_markers_then_enter(window, ui):
    window.set_tool("watershed")
    ui.drag((40, 30), (50, 40))
    assert window.canvas.shapes == []  # nothing until Enter
    ui.key(Qt.Key.Key_Return)
    assert masks.iou(full(window), RED) > 0.85  # flat synthetic background leaks a little


def test_watershed_alt_paints_background_markers(window, ui):
    window.set_tool("watershed")
    tool = window.pixel_tools["watershed"]
    ui.drag((40, 30), (50, 40))
    c = window.canvas
    ui.qtbot.mousePress(c, Qt.MouseButton.LeftButton, ALT, ui.screen(5, 70))
    ui.qtbot.mouseMove(c, ui.screen(95, 70))
    ui.qtbot.mouseRelease(c, Qt.MouseButton.LeftButton, ALT, ui.screen(95, 70))
    assert tool.background[70, 50] and not tool.foreground[70, 50]
    assert not window.canvas.grab().isNull()
    ui.key(Qt.Key.Key_Enter)
    assert full(window)[35, 45] and not full(window)[70, 50]


def test_watershed_enter_without_markers_only_warns(window, ui):
    window.set_tool("watershed")
    ui.key(Qt.Key.Key_Return)
    assert window.canvas.shapes == [] and "markers" in window.status_label.text()


def test_watershed_escape_clears_markers_then_deselects(window, ui):
    window.set_tool("watershed")
    tool = window.pixel_tools["watershed"]
    ui.drag((40, 30), (50, 40))
    ui.key(Qt.Key.Key_Escape)
    assert tool.foreground is None
    ui.key(Qt.Key.Key_Escape)  # no markers: falls through to the canvas


def test_watershed_marker_size_keys(window, ui):
    window.set_tool("watershed")
    tool = window.pixel_tools["watershed"]
    ui.qtbot.keyClicks(window.canvas, "]]")
    assert tool.radius == 5
    ui.qtbot.keyClicks(window.canvas, "[" * 10)
    assert tool.radius == 1
    ui.key(Qt.Key.Key_A)  # unrelated key goes to the canvas / window


# ---------------------------------------------------------------------- threshold panel


@pytest.fixture
def panel(window):
    window.open_threshold()
    return window.threshold_dialog


def test_threshold_panel_previews_otsu_and_adds_mask(window, panel):
    assert window.canvas.tool is panel.tool and panel.isVisible()
    panel.channel.setCurrentText("red")
    assert np.array_equal(panel.preview(), RED)
    assert not window.canvas.grab().isNull()
    add = next(
        b for b in panel.findChildren(QDialogButtonBox)[0].buttons() if b.text() == "Add mask"
    )
    add.click()
    (shape,) = window.canvas.shapes
    assert shape.label == "obj" and np.array_equal(full(window), RED)


def test_threshold_manual_level_and_invert(window, panel):
    panel.method.setCurrentText("Manual")
    panel.level.setValue(250)
    assert not panel.preview().any()
    panel.apply()  # empty: nothing added
    assert window.canvas.shapes == []
    panel.invert.setChecked(True)
    assert panel.preview().all()


def test_threshold_adaptive(window, panel):
    panel.method.setCurrentText("Adaptive")
    panel.block.setValue(5)
    assert panel.preview()[0, 0] and not panel.preview()[35, 28]


def test_colour_range_picks_colour_by_clicking_image(window, ui, panel):
    panel.method.setCurrentText("Colour range")
    click(ui, 80, 10)
    assert panel.color == (30, 30, 220)
    assert np.array_equal(panel.preview(), BLUE)
    panel.invert.setChecked(True)
    assert np.array_equal(panel.preview(), ~BLUE)


def test_threshold_limited_to_selected_shape(window, panel):
    window.canvas.set_shapes([Shape("box", ShapeType.RECTANGLE, [(0, 0), (45, 79)])])
    window.canvas.select(0)
    panel.channel.setCurrentText("red")
    panel.inside_selection.setChecked(True)
    result = panel.preview()
    assert result[30, 40] and not result[30, 50]
    panel.method.setCurrentText("Colour range")
    panel.set_color((220, 30, 30))
    assert panel.preview()[30, 40] and not panel.preview()[30, 50]


def test_threshold_preview_follows_image_change(window, panel):
    panel.channel.setCurrentText("red")
    first = panel.preview()
    window.next_image()
    window.canvas.set_tool(panel.tool)
    assert np.array_equal(panel.preview(), first[:, ::-1])


def test_closing_threshold_panel_returns_to_edit_mode(window, panel):
    panel.reject()
    assert window.canvas.tool is None and window.tool_actions[None].isChecked()


def test_closing_panel_after_switching_tool_keeps_tool(window, panel):
    window.set_tool("wand")
    panel.reject()
    assert window.canvas.tool is window.pixel_tools["wand"]


def test_reopening_threshold_panel_reuses_it(window, panel):
    panel.reject()
    window.open_threshold()
    assert window.threshold_dialog is panel and window.canvas.tool is panel.tool


def test_threshold_panel_needs_an_image(make_window):
    win = make_window()
    win.open_threshold()
    assert win.threshold_dialog is None


# ---------------------------------------------------------------------- display


def test_display_adjustments_change_view_not_data(window):
    window.open_display()
    dialog = window.display_dialog
    dialog.brightness.setValue(80)
    dialog.clahe.setChecked(True)
    dialog.view_channel.setCurrentText("red")
    assert window.canvas.display == DisplaySettings(80, 1.0, 1.0, True, "red", 1.0)
    shown = pixmap_to_rgb(window.canvas._shown)
    assert not np.array_equal(shown, scene())
    assert np.array_equal(window.canvas.image_rgb(), scene())  # tools see the original


def test_display_settings_survive_image_change_and_reset(window):
    window.open_display()
    dialog = window.display_dialog
    dialog.contrast.setValue(250)
    dialog.gamma.setValue(50)
    window.next_image()
    assert window.canvas.display.contrast == 2.5
    assert not np.array_equal(pixmap_to_rgb(window.canvas._shown), window.canvas.image_rgb())
    dialog.reset()
    assert window.canvas.display == DisplaySettings()
    assert window.canvas._shown is window.canvas.pixmap
    window.open_display()
    assert window.display_dialog is dialog


def test_mask_opacity_fades_masks(window, ui):
    window.set_tool("wand")
    click(ui, 40, 30)
    window.canvas.select(None)
    window.open_display()
    canvas = window.canvas
    centre = ui.screen(45, 35)
    opaque = canvas.grab().toImage().pixelColor(centre)
    window.display_dialog.opacity.setValue(0)
    clear = canvas.grab().toImage().pixelColor(centre)
    assert canvas.display.mask_opacity == 0
    assert (clear.red(), clear.green(), clear.blue()) == (220, 30, 30)
    assert opaque != clear


def test_menu_actions_open_panels(window):
    actions = {a.text(): a for a in window.actions()}
    actions["Threshold / Colour Range..."].trigger()
    actions["Adjust Display..."].trigger()
    assert window.threshold_dialog.isVisible() and window.display_dialog.isVisible()


@pytest.mark.parametrize(
    ("key", "name"), [("W", "wand"), ("G", "grabcut"), ("S", "superpixel"), ("M", "watershed")]
)
def test_tool_shortcuts(window, ui, key, name):
    ui.key(getattr(Qt.Key, f"Key_{key}"))
    assert window.canvas.tool is window.pixel_tools[name]
