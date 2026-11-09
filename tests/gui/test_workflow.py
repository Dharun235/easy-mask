"""Review mode, copy-from-previous, statistics and strict labels, driven through the window."""

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog

from easy_mask.core import annotation as ann_io
from easy_mask.core.annotation import REVIEWED_FLAG, Annotation
from easy_mask.core.shapes import Shape, ShapeType
from easy_mask.gui import workflow
from easy_mask.gui.label_dialog import LabelInfo
from easy_mask.gui.main_window import Settings


def rect(label, x, y, size=10):
    return Shape(label, ShapeType.RECTANGLE, [(x, y), (x + size, y + size)])


@pytest.fixture
def folder(image_dir):
    """img1: two objects; img2: one object; img10: nothing."""
    ann_io.save(
        Annotation("img1.png", 100, 80, [rect("cat", 5, 5), rect("dog", 60, 40)]),
        image_dir / "img1.json",
    )
    ann_io.save(Annotation("img2.png", 100, 80, [rect("cat", 30, 30)]), image_dir / "img2.json")
    return image_dir


@pytest.fixture
def win(make_window, folder):
    window = make_window(Settings(labels=["cat", "dog"]))
    window.open_path(folder)
    return window


def menu_action(window, text):
    return next(a for a in window.actions() if a.text() == text)


# ---------------------------------------------------------------------- review mode


def test_n_steps_through_objects_and_zooms_to_each(win, driver):
    fitted = win.canvas.scale
    driver(win).key(Qt.Key.Key_N)
    assert win.canvas.selection == [0]
    assert win.canvas.scale > fitted
    assert "Object 1/2: cat" in win.statusBar().currentMessage()

    driver(win).key(Qt.Key.Key_N)
    assert win.canvas.selection == [1]
    centre = win.canvas.to_image(win.canvas.rect().center().toPointF())
    assert centre == pytest.approx((65, 45), abs=1)


def test_stepping_past_last_object_marks_reviewed_and_moves_on(win, folder):
    for _ in range(3):
        win.workflow.next_object()
    assert ann_io.load(folder / "img1.json").flags == {REVIEWED_FLAG: True}
    assert win.image_path.name == "img2.png"
    assert win.canvas.selection == [0]
    assert not win.dirty


def test_empty_image_is_reviewed_with_one_step(win, folder):
    win.file_list.setCurrentRow(2)  # img10, no annotation
    win.workflow.next_object()
    assert "No objects" not in win.statusBar().currentMessage()
    assert ann_io.load(folder / "img10.json").flags[REVIEWED_FLAG]
    assert "Last image reviewed" in win.statusBar().currentMessage()


def test_landing_on_empty_image_says_so(win, folder):
    win.file_list.setCurrentRow(1)  # img2
    win.workflow.next_object()
    win.workflow.next_object()
    assert win.image_path.name == "img10.png"
    assert "No objects" in win.statusBar().currentMessage()


def test_shift_n_goes_back_across_images(win, driver):
    win.file_list.setCurrentRow(1)  # img2
    driver(win).key(Qt.Key.Key_N, Qt.KeyboardModifier.ShiftModifier)
    assert win.canvas.selection == [0]
    win.workflow.prev_object()
    assert win.image_path.name == "img1.png"
    assert win.canvas.selection == [1]  # the previous image's last object


def test_prev_on_first_image_stays(win):
    win.workflow.prev_object()
    win.workflow.prev_object()
    win.workflow.prev_object()
    assert win.image_path.name == "img1.png" and win.canvas.selection == [0]


def test_review_relabel_and_delete_use_normal_edits(win):
    win.workflow.next_object()
    win.ask_label = lambda known, initial, parent: LabelInfo("dog")
    win.edit_label()
    win.workflow.next_object()
    win.delete_selected()
    assert [s.label for s in win.canvas.shapes] == ["dog"]
    win.undo()
    assert len(win.canvas.shapes) == 2


def test_review_without_image_does_nothing(make_window):
    window = make_window()
    window.workflow.next_object()
    window.workflow.prev_object()
    assert window.image_path is None


def test_reviewed_flag_is_not_set_twice(win):
    win.workflow.mark_reviewed()
    win.dirty = False
    win.workflow.mark_reviewed()
    assert not win.dirty
    assert win.flag_list.flags() == {REVIEWED_FLAG: True}


# ---------------------------------------------------------------------- copy previous


# ---------------------------------------------------------------------- statistics


@pytest.fixture
def reports(win):
    shown = []
    win.workflow.show_report = lambda parent, title, text: shown.append(text)
    return shown


def test_statistics_report(win, reports):
    menu_action(win, "Statistics...").trigger()
    (text,) = reports
    assert "Images   3  (annotated 2, unannotated 1, reviewed 0)" in text
    assert "All labels are in the label list." in text


def test_statistics_mentions_unsaved_changes_and_unknown_labels(win, reports, folder):
    ann_io.save(Annotation("img2.png", 100, 80, [rect("zebra", 1, 1)]), folder / "img2.json")
    win.dirty = True
    win.workflow.show_statistics()
    assert reports[0].startswith("Unsaved changes")
    assert "zebra: img2.png" in reports[0]


def test_statistics_without_folder_warns(make_window, dialogs):
    dialogs.allow_warnings = True
    make_window().workflow.show_statistics()
    assert dialogs.warnings == ["Open a folder first."]


def test_statistics_with_corrupt_file_warns(win, folder, dialogs):
    dialogs.allow_warnings = True
    (folder / "img2.json").write_text("{ broken")
    win.workflow.show_statistics()
    assert "Could not read every annotation" in dialogs.warnings[0]


def test_show_report_dialog(qtbot, monkeypatch):
    seen = []
    monkeypatch.setattr(QDialog, "exec", lambda self: seen.append(self.windowTitle()) or 0)
    workflow.show_report(None, "Statistics", "a table")
    assert seen == ["Statistics"]


# ---------------------------------------------------------------------- strict labels


def test_strict_labels_reject_unknown_label(make_window, image_dir, driver):
    window = make_window(Settings(labels=["cat"], strict_labels=True), LabelInfo("zebra"))
    window.open_path(image_dir)
    window.set_tool(ShapeType.POINT)
    driver(window).click(20, 20)
    assert window.canvas.shapes == []
    assert "'zebra' is not in the label list" in window.statusBar().currentMessage()


def test_strict_labels_accept_listed_label(make_window, image_dir, driver):
    window = make_window(Settings(labels=["cat"], strict_labels=True), LabelInfo("cat"))
    window.open_path(image_dir)
    window.set_tool(ShapeType.POINT)
    driver(window).click(20, 20)
    assert [s.label for s in window.canvas.shapes] == ["cat"]


def test_labels_without_strict_allow_anything(make_window, image_dir, driver):
    window = make_window(Settings(labels=["cat"]), LabelInfo("zebra"))
    window.open_path(image_dir)
    window.set_tool(ShapeType.POINT)
    driver(window).click(20, 20)
    assert [s.label for s in window.canvas.shapes] == ["zebra"]


def test_review_visits_automatic_shapes_first(win):
    from easy_mask.core.video import PROPAGATED_FLAG

    def box(x, **flags):
        return Shape("cat", ShapeType.RECTANGLE, [(x, 10), (x + 5, 15)], flags=flags)

    win.canvas.set_shapes([box(5), box(30, **{PROPAGATED_FLAG: True}), box(60)])
    assert win.workflow.review_order() == [1, 0, 2]
    win.workflow.next_object()
    assert win.canvas.selected == 1
    assert "auto: check it" in win.statusBar().currentMessage()
    win.workflow.next_object()
    assert win.canvas.selected == 0 and "auto" not in win.statusBar().currentMessage()
    win.workflow.prev_object()
    assert win.canvas.selected == 1
