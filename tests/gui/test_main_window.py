"""End-to-end GUI tests: real mouse/key events on a headless window."""

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QMessageBox

from easy_mask.core import annotation as ann_io
from easy_mask.core.shapes import ShapeType
from easy_mask.gui.label_dialog import LabelInfo
from easy_mask.gui.main_window import MainWindow, Settings


@pytest.fixture(autouse=True)
def no_modal_dialogs(monkeypatch):
    """A modal dialog would block a headless test forever; fail loudly instead."""

    def fail(*args, **kwargs):
        raise AssertionError(f"unexpected modal dialog: {args[1:3]}")

    monkeypatch.setattr(QMessageBox, "question", fail)
    monkeypatch.setattr(QMessageBox, "warning", fail)


@pytest.fixture
def make_window(qtbot):
    """Factory for shown windows; closes them without the unsaved-changes prompt."""
    windows = []

    def make(settings: Settings, label: LabelInfo) -> MainWindow:
        win = MainWindow(settings)
        win.ask_label = lambda known, initial, parent: label
        win.show()
        qtbot.waitExposed(win)
        windows.append(win)
        return win

    yield make
    for win in windows:
        win.dirty = False
        win.close()


@pytest.fixture
def window(make_window, image_dir):
    win = make_window(Settings(labels=["cat", "dog"], flags=["blurry"]), LabelInfo("cat"))
    win.open_path(image_dir)
    return win


def screen(win, x, y) -> QPoint:
    """Widget position of image pixel (x, y)."""
    c = win.canvas
    return QPoint(round(c.offset.x() + x * c.scale), round(c.offset.y() + y * c.scale))


def click(qtbot, win, x, y):
    qtbot.mouseClick(win.canvas, Qt.MouseButton.LeftButton, pos=screen(win, x, y))


def drag(qtbot, win, start, end):
    c = win.canvas
    qtbot.mousePress(c, Qt.MouseButton.LeftButton, pos=screen(win, *start))
    qtbot.mouseMove(c, screen(win, (start[0] + end[0]) / 2, (start[1] + end[1]) / 2))
    qtbot.mouseMove(c, screen(win, *end))
    qtbot.mouseRelease(c, Qt.MouseButton.LeftButton, pos=screen(win, *end))


def test_opens_folder(window):
    assert window.file_list.count() == 3
    assert window.image_path.name == "img1.png"
    assert window.flag_list.flags() == {"blurry": False}


def test_draw_polygon_then_undo_redo(qtbot, window):
    window.set_tool(ShapeType.POLYGON)
    for x, y in [(10, 10), (40, 10), (40, 40)]:
        click(qtbot, window, x, y)
    qtbot.keyClick(window.canvas, Qt.Key.Key_Return)

    assert len(window.canvas.shapes) == 1
    shape = window.canvas.shapes[0]
    assert shape.label == "cat" and shape.shape_type == ShapeType.POLYGON
    assert shape.points[1] == pytest.approx((40, 10), abs=1)
    assert window.dirty and window.shape_list.count() == 1

    window.undo()
    assert window.canvas.shapes == []
    window.redo()
    assert len(window.canvas.shapes) == 1


def test_clicking_first_vertex_closes_polygon(qtbot, window):
    window.set_tool(ShapeType.POLYGON)
    for x, y in [(10, 10), (40, 10), (40, 40), (10, 10)]:
        click(qtbot, window, x, y)
    assert len(window.canvas.shapes) == 1
    assert len(window.canvas.shapes[0].points) == 3


def test_backspace_removes_last_draft_point(qtbot, window):
    window.set_tool(ShapeType.POLYGON)
    for x, y in [(10, 10), (40, 10), (60, 60), (40, 40)]:
        click(qtbot, window, x, y)
    qtbot.keyClick(window.canvas, Qt.Key.Key_Backspace)
    qtbot.keyClick(window.canvas, Qt.Key.Key_Return)
    assert len(window.canvas.shapes[0].points) == 3


def test_rectangle_by_drag_and_by_two_clicks(qtbot, window):
    window.set_tool(ShapeType.RECTANGLE)
    drag(qtbot, window, (10, 10), (30, 30))
    click(qtbot, window, 50, 50)
    click(qtbot, window, 70, 70)
    assert [s.shape_type for s in window.canvas.shapes] == [ShapeType.RECTANGLE] * 2


def test_cancelled_label_dialog_discards_shape(qtbot, window):
    window.ask_label = lambda known, initial, parent: None
    window.set_tool(ShapeType.POINT)
    click(qtbot, window, 20, 20)
    assert window.canvas.shapes == []


def test_move_shape_and_vertex_in_edit_mode(qtbot, window):
    window.set_tool(ShapeType.RECTANGLE)
    drag(qtbot, window, (10, 10), (30, 30))
    window.set_tool(None)

    drag(qtbot, window, (20, 20), (40, 20))  # grab the body
    x1, y1 = window.canvas.shapes[0].points[0]
    assert x1 == pytest.approx(30, abs=1.5) and y1 == pytest.approx(10, abs=1.5)

    drag(qtbot, window, (50, 30), (60, 50))  # grab the bottom-right vertex
    assert window.canvas.shapes[0].points[1] == pytest.approx((60, 50), abs=1.5)
    window.undo()
    assert window.canvas.shapes[0].points[1] == pytest.approx((50, 30), abs=1.5)


def test_delete_duplicate_copy_paste(qtbot, window, image_dir):
    window.set_tool(ShapeType.CIRCLE)
    drag(qtbot, window, (50, 40), (60, 40))
    window.set_tool(None)
    click(qtbot, window, 50, 40)
    assert window.canvas.selected == 0

    window.duplicate_selected()
    assert len(window.canvas.shapes) == 2
    window.delete_selected()
    assert len(window.canvas.shapes) == 1

    window.canvas.select(0)
    window.copy_selected()
    window.settings.autosave = True
    window.next_image()
    assert window.image_path.name == "img2.png"
    window.paste()
    assert window.canvas.shapes[0].shape_type == ShapeType.CIRCLE


def test_save_and_reload(qtbot, window, image_dir):
    window.set_tool(ShapeType.LINE)
    drag(qtbot, window, (5, 5), (50, 50))
    window.flag_list.item(0).setCheckState(Qt.CheckState.Checked)
    assert window.save()

    saved = ann_io.load(image_dir / "img1.json")
    assert saved.image_path == "img1.png" and saved.flags == {"blurry": True}
    assert saved.shapes[0].shape_type == ShapeType.LINE
    assert window.file_list.item(0).checkState() == Qt.CheckState.Checked
    assert not window.dirty

    window.next_image()
    window.prev_image()
    assert len(window.canvas.shapes) == 1


def test_autosave_on_navigation_with_output_dir(qtbot, make_window, image_dir, tmp_path):
    out = tmp_path / "labels"
    win = make_window(Settings(output_dir=out, autosave=True), LabelInfo("dog", group_id=2))
    win.open_path(image_dir / "img2.png")
    win.set_tool(ShapeType.POINT)
    click(qtbot, win, 10, 10)
    win.next_image()

    saved = ann_io.load(out / "img2.json")
    assert saved.image_path == "../images/img2.png"
    assert saved.shapes[0].group_id == 2


def test_hidden_shape_cannot_be_selected(qtbot, window):
    window.set_tool(ShapeType.RECTANGLE)
    drag(qtbot, window, (10, 10), (30, 30))
    window.set_tool(None)
    window.shape_list.item(0).setCheckState(Qt.CheckState.Unchecked)
    window.canvas.select(None)
    click(qtbot, window, 20, 20)
    assert window.canvas.selected is None


def test_zoom_keeps_anchor_fixed(window):
    c = window.canvas
    anchor = screen(window, 50, 40)
    before = c.to_image(anchor.toPointF())
    c.zoom_by(2.0, anchor.toPointF())
    assert c.to_image(anchor.toPointF()) == pytest.approx(before, abs=0.01)


def test_double_click_relabels_unselected_shape(qtbot, window):
    window.set_tool(ShapeType.RECTANGLE)
    drag(qtbot, window, (10, 10), (30, 30))
    window.set_tool(None)
    window.canvas.select(None)
    window.ask_label = lambda known, initial, parent: LabelInfo("dog", description="note")
    qtbot.mouseDClick(window.canvas, Qt.MouseButton.LeftButton, pos=screen(window, 20, 20))
    shape = window.canvas.shapes[0]
    assert (shape.label, shape.description) == ("dog", "note")
    assert shape.points[0] == pytest.approx((10, 10), abs=1)  # not dragged


def test_gui_export_keeps_labels_added_in_session(qtbot, window, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    window.ask_label = lambda known, initial, parent: LabelInfo("zebra")  # not in --labels
    window.set_tool(ShapeType.RECTANGLE)
    drag(qtbot, window, (10, 10), (30, 30))
    window.save()
    out = tmp_path / "exp"
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: str(out))
    window.export_dataset("masks")
    assert (out / "class_names.txt").read_text().split() == ["_background_", "cat", "dog", "zebra"]
