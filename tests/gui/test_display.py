"""Image <-> pixmap helpers and display adjustments, driven through the real window."""

import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import Qt

from easy_mask.core import masks
from easy_mask.core.display import DisplaySettings
from easy_mask.core.shapes import ShapeType
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
    window.set_tool(ShapeType.RECTANGLE)
    ui.drag((30, 20), (59, 49))
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
    actions["Adjust Display..."].trigger()
    assert window.display_dialog.isVisible()
