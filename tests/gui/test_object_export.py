"""File > Export > Objects: the parts dialog, the export and remembering the choice."""

import json

from easy_mask.core.shapes import ShapeType
from easy_mask.gui.main_window import Settings
from easy_mask.gui.object_export_dialog import ObjectExportDialog, ask_object_parts
from easy_mask.io.project import PROJECT_FILE


def test_dialog_needs_at_least_one_part(qtbot):
    from PySide6.QtWidgets import QDialogButtonBox

    dialog = ObjectExportDialog(["masks", "coordinates"])
    qtbot.addWidget(dialog)
    ok = dialog.buttons.button(QDialogButtonBox.StandardButton.Ok)
    assert dialog.parts() == ["masks", "coordinates"] and ok.isEnabled()
    dialog.boxes["masks"].setChecked(False)
    dialog.boxes["coordinates"].setChecked(False)
    assert dialog.parts() == [] and not ok.isEnabled()


def test_ask_object_parts(qtbot, monkeypatch):
    from PySide6.QtWidgets import QDialog

    monkeypatch.setattr(ObjectExportDialog, "exec", lambda self: QDialog.DialogCode.Rejected)
    assert ask_object_parts(["masks"]) is None
    monkeypatch.setattr(ObjectExportDialog, "exec", lambda self: QDialog.DialogCode.Accepted)
    assert ask_object_parts(["cutouts"]) == ["cutouts"]


def test_export_objects_from_menu_and_remember_parts(
    make_window, image_dir, tmp_path, driver, wait_export, monkeypatch
):
    from PySide6.QtWidgets import QFileDialog

    win = make_window(Settings())
    win.open_path(image_dir)
    win.set_active_label("cat")
    win.set_tool(ShapeType.RECTANGLE)
    driver(win).drag((10, 10), (30, 30))
    win.save()
    win.ask_object_parts = lambda current, parent: ["cutouts", "coordinates"]
    out = tmp_path / "objects"
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: str(out))
    win.export_dataset("objects")
    wait_export(win)
    assert (out / "cutouts" / "img1_01_cat.png").exists() and not (out / "masks").exists()
    assert json.loads((image_dir / PROJECT_FILE).read_text())["object_parts"] == [
        "cutouts",
        "coordinates",
    ]
    again = make_window(Settings())
    again.open_path(image_dir)
    assert again.object_parts == ["cutouts", "coordinates"]


def test_cancelling_parts_dialog_exports_nothing(make_window, image_dir):
    win = make_window(Settings())
    win.open_path(image_dir)
    win.ask_object_parts = lambda current, parent: None
    win.export_dataset("objects")
    assert win.export_worker is None


def test_export_menu_uses_readable_names(make_window):
    win = make_window(Settings())
    file_menu = next(a.menu() for a in win.menuBar().actions() if a.text() == "&File")
    export_menu = next(a.menu() for a in file_menu.actions() if a.text() == "Export Folder As")
    names = [a.text() for a in export_menu.actions()]
    assert names[0].startswith("Objects: binary masks, cutouts, coordinates")
    assert "COCO (exact RLE masks)" in names
