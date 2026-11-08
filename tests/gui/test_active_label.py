"""Active label: new shapes take it without the label dialog; auto-save on by default."""

from PySide6.QtCore import Qt

from easy_mask.cli import build_gui_parser
from easy_mask.core.shapes import ShapeType
from easy_mask.gui.main_window import Settings


def no_dialog(*args):
    raise AssertionError("label dialog must not open with an active label")


def test_number_key_activates_label_and_skips_dialog(window, ui):
    ui.qtbot.keyClick(window, Qt.Key.Key_2)
    assert window.active_label == "dog"
    window.ask_label = no_dialog
    window.set_tool(ShapeType.RECTANGLE)
    ui.drag((10, 10), (30, 30))
    assert window.canvas.shapes[0].label == "dog"
    assert "dog" in window.active_label_status.text()


def test_zero_goes_back_to_asking(window, ui):
    window.set_active_label("cat")
    ui.qtbot.keyClick(window, Qt.Key.Key_0)
    assert window.active_label is None
    asked = []
    window.ask_label = lambda known, initial, parent: asked.append(initial) or None
    window.set_tool(ShapeType.POINT)
    ui.click(20, 20)
    assert asked and window.canvas.shapes == []


def test_number_beyond_known_labels_is_ignored(window):
    window.set_active_label_number(9)
    assert window.active_label is None


def test_clicking_label_toggles_it(window):
    item = window.label_list.item(1)  # row 0 is Erase
    window.label_list.itemClicked.emit(item)
    assert window.active_label == "cat"
    assert window.label_list.item(1).font().bold()
    window.label_list.itemClicked.emit(window.label_list.item(1))
    assert window.active_label is None


def test_labels_panel_starts_with_erase(window):
    texts = [window.label_list.item(i).text() for i in range(window.label_list.count())]
    assert texts[0].startswith("✂") and texts[1:] == ["cat", "dog"]


def test_polygon_with_erase_label_cuts_instead_of_being_kept(window, ui):
    from easy_mask.core.rasterize import shape_to_mask
    from easy_mask.core.shapes import Shape
    from easy_mask.core.video import ERASE_LABEL

    window.canvas.set_shapes([Shape("dog", ShapeType.RECTANGLE, [(10, 10), (60, 60)])])
    window._commit()
    window.set_active_label(ERASE_LABEL)
    assert "Erase" in window.active_label_status.text()
    assert ERASE_LABEL not in window.known_labels
    window.set_tool(ShapeType.RECTANGLE)
    ui.drag((40, 5), (70, 70))
    (dog,) = window.canvas.shapes
    full = shape_to_mask(dog, 80, 100)
    assert full[30, 20] and not full[30, 50]
    assert window.video.last_cut is not None  # Propagate will carry it


def test_label_dialog_erase_button(qtbot):
    from PySide6.QtWidgets import QPushButton

    from easy_mask.core.video import ERASE_LABEL
    from easy_mask.gui.label_dialog import LabelDialog, LabelInfo

    dialog = LabelDialog(["cat"], LabelInfo("cat"))
    qtbot.addWidget(dialog)
    erase = next(b for b in dialog.findChildren(QPushButton) if b.text() == "Erase With It")
    erase.click()
    assert dialog.result() and dialog.result_info().label == ERASE_LABEL


def test_brush_mask_uses_active_label(window, ui):
    window.set_active_label("zebra")  # new labels are remembered
    window.ask_label = no_dialog
    window.set_tool("brush")
    ui.drag((20, 40), (60, 40))
    assert window.canvas.shapes[0].label == "zebra"
    assert "zebra" in window.known_labels


def test_edit_label_still_opens_dialog(window, ui):
    window.set_active_label("cat")
    window.set_tool(ShapeType.RECTANGLE)
    ui.drag((10, 10), (30, 30))
    window.set_tool(None)
    window.canvas.select(0)
    from easy_mask.gui.label_dialog import LabelInfo

    window.ask_label = lambda known, initial, parent: LabelInfo("dog")
    window.edit_label()
    assert window.canvas.shapes[0].label == "dog"


def test_autosave_is_on_by_default(make_window, image_dir, driver):
    win = make_window(Settings())
    assert win.settings.autosave
    win.open_path(image_dir)
    win.set_active_label("cat")
    win.set_tool(ShapeType.POINT)
    driver(win).click(10, 10)
    win.next_image()  # no "save changes?" question: the dialogs fixture would fail
    assert (image_dir / "img1.json").exists()


def test_cli_autosave_flags():
    parser = build_gui_parser()
    assert parser.parse_args([]).autosave is True
    assert parser.parse_args(["--no-autosave"]).autosave is False


def test_second_object_with_the_same_label_gets_its_own_group_id(window, ui):
    window.set_active_label("dog")
    window.set_tool(ShapeType.RECTANGLE)
    ui.drag((10, 10), (30, 30))
    ui.drag((50, 10), (70, 30))
    first, second = window.canvas.shapes
    assert (first.label, first.group_id) == ("dog", None)
    assert (second.label, second.group_id) == ("dog", 1)  # a separate object
