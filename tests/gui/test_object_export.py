"""The Export window: format, parts, frames and folder; remembering the choices."""

import json

from easy_mask.core.shapes import ShapeType
from easy_mask.gui.export_dialog import ExportChoice, ExportDialog, ask_export
from easy_mask.gui.main_window import Settings
from easy_mask.io.project import PROJECT_FILE


def dialog_for(qtbot, tmp_path, **changes):
    choice = ExportChoice("objects", tmp_path / "out", ["masks", "coordinates"], 0, 9)
    for key, value in changes.items():
        setattr(choice, key, value)
    dialog = ExportDialog(choice, frame_count=10, current=3)
    qtbot.addWidget(dialog)
    return dialog


def test_parts_include_centroid_and_corners_as_separate_choices(qtbot, tmp_path):
    from PySide6.QtWidgets import QDialogButtonBox

    dialog = dialog_for(qtbot, tmp_path)
    assert list(dialog.part_checks) == [
        "masks",
        "cutouts",
        "coordinates",
        "centroid",
        "corners",
        "points",
        "overlays",
    ]
    dialog.part_checks["centroid"].setChecked(True)
    assert dialog.choice().parts == ["masks", "coordinates", "centroid"]
    ok = dialog.buttons.button(QDialogButtonBox.StandardButton.Ok)
    for check in dialog.part_checks.values():
        check.setChecked(False)
    assert not ok.isEnabled()  # Objects needs at least one part
    dialog.format_box.setCurrentIndex(dialog.format_box.findData("coco"))
    assert ok.isEnabled()  # other formats have no parts


def test_frames_all_this_or_a_range(qtbot, tmp_path):
    dialog = dialog_for(qtbot, tmp_path)
    assert dialog.all_frames.isChecked()
    assert (dialog.choice().first, dialog.choice().last) == (0, 9)
    dialog.this_frame.setChecked(True)
    assert (dialog.choice().first, dialog.choice().last) == (3, 3)
    dialog.first_spin.setValue(8)
    dialog.last_spin.setValue(5)  # typed backwards: still a valid range
    assert dialog.range_frames.isChecked()
    assert (dialog.choice().first, dialog.choice().last) == (4, 7)


def test_a_saved_range_reopens_as_a_range(qtbot, tmp_path):
    dialog = dialog_for(qtbot, tmp_path, first=2, last=4)
    assert dialog.range_frames.isChecked()
    assert (dialog.first_spin.value(), dialog.last_spin.value()) == (3, 5)


def test_ask_export(qtbot, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QDialog

    choice = ExportChoice("coco", tmp_path, [], 0, 0)
    monkeypatch.setattr(ExportDialog, "exec", lambda self: QDialog.DialogCode.Rejected)
    assert ask_export(choice, 1, 0) is None
    monkeypatch.setattr(ExportDialog, "exec", lambda self: QDialog.DialogCode.Accepted)
    assert ask_export(choice, 1, 0).fmt == "coco"


def test_export_window_exports_chosen_parts_and_frames_and_remembers(
    make_window, image_dir, tmp_path, driver, wait_export, export_as
):
    win = make_window(Settings())
    win.open_path(image_dir)
    win.set_active_label("cat")
    win.set_tool(ShapeType.RECTANGLE)
    ui = driver(win)
    ui.drag((10, 10), (30, 30))
    win.next_image()
    ui.drag((40, 40), (60, 60))
    win.save()
    out = tmp_path / "objects"
    export_as(win, "objects", out, parts=["centroid", "corners"], first=1, last=1)
    wait_export(win)
    records = json.loads((out / "objects.json").read_text())
    assert [r["image"] for r in records] == ["img2.png"]  # only the chosen frame
    assert "centroid" in records[0] and "rotated_corners" in records[0]
    assert not (out / "masks").exists()
    project = json.loads((image_dir / PROJECT_FILE).read_text())
    assert project["object_parts"] == ["centroid", "corners"]
    seen = []
    win.ask_export = lambda choice, count, current, parent: seen.append(choice)
    win.export_dialog()  # opens again with the last choices
    assert seen[0].fmt == "objects" and seen[0].out == out
    assert seen[0].parts == ["centroid", "corners"]


def test_cancelling_the_export_window_exports_nothing(make_window, image_dir):
    win = make_window(Settings())
    win.open_path(image_dir)
    win.ask_export = lambda choice, count, current, parent: None
    win.export_dialog()
    assert win.export_worker is None


def test_toolbar_export_opens_the_window(make_window, image_dir):
    win = make_window(Settings())
    win.open_path(image_dir)
    seen = []
    win.ask_export = lambda choice, count, current, parent: seen.append((count, current))
    win._export_action.trigger()
    assert seen == [(3, 0)]
