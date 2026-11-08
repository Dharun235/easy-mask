"""Remembered state: per-folder project file and app-wide settings."""

import json

from easy_mask.core.shapes import ShapeType
from easy_mask.gui.label_dialog import LabelInfo
from easy_mask.gui.main_window import Settings
from easy_mask.io.project import PROJECT_FILE


def test_labels_and_active_label_are_restored_per_folder(make_window, image_dir, driver):
    first = make_window(Settings(), LabelInfo("zebra"))
    first.open_path(image_dir)
    first.set_tool(ShapeType.POINT)
    driver(first).click(10, 10)  # labels 'zebra' via the dialog
    first.set_active_label("zebra")
    first.close()
    saved = json.loads((image_dir / PROJECT_FILE).read_text())
    assert saved["labels"] == ["zebra"] and saved["active_label"] == "zebra"

    second = make_window(Settings())
    second.open_path(image_dir)
    assert second.known_labels == ["zebra"] and second.active_label == "zebra"


def test_command_line_labels_come_first(make_window, image_dir):
    (image_dir / PROJECT_FILE).write_text(json.dumps({"labels": ["b", "a"], "flags": ["dark"]}))
    win = make_window(Settings(labels=["a", "c"]))
    win.open_path(image_dir)
    assert win.known_labels == ["a", "c", "b"]
    assert win.flag_list.flags() == {"dark": False}


def test_output_dir_is_remembered_relative(make_window, image_dir, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    labels = image_dir / "labels"
    labels.mkdir()
    win = make_window(Settings())
    win.open_path(image_dir)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: str(labels))
    win.change_output_dir()
    assert json.loads((image_dir / PROJECT_FILE).read_text())["output_dir"] == "labels"
    again = make_window(Settings())
    again.open_path(image_dir)
    assert again.settings.output_dir == labels


def test_recent_folders_and_menu(make_window, image_dir, tmp_path):
    win = make_window(Settings())
    win.open_path(image_dir)
    win.open_path(image_dir / "img2.png")  # same folder: not duplicated
    assert win.session.recent() == [image_dir.resolve()]
    win.session.recent_menu.aboutToShow.emit()
    (action,) = win.session.recent_menu.actions()
    assert action.text() == str(image_dir.resolve())


def test_empty_recent_menu_shows_placeholder(make_window):
    win = make_window(Settings())
    win.session.recent_menu.aboutToShow.emit()
    (action,) = win.session.recent_menu.actions()
    assert not action.isEnabled()


def test_tool_sizes_and_layout_survive_restart(make_window):
    win = make_window(Settings())
    win.pixel_tools["brush"].radius = 23
    win.session.save()
    again = make_window(Settings())
    assert again.pixel_tools["brush"].radius == 23


def test_read_only_folder_is_not_an_error(make_window, image_dir, monkeypatch):
    import easy_mask.io.project as project_io

    def refuse(*args):
        raise OSError("read-only")

    monkeypatch.setattr(project_io, "save", refuse)
    win = make_window(Settings())
    win.open_path(image_dir)
    win.set_active_label("cat")  # would write the project file
    assert win.active_label == "cat"


# ---------------------------------------------------------------------- export settings


def export_once(win, out, monkeypatch, fmt="coco"):
    from PySide6.QtWidgets import QFileDialog

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: str(out))
    win.export_dataset(fmt)


def annotate(win, driver):
    win.set_active_label("cat")
    win.set_tool(ShapeType.RECTANGLE)
    driver(win).drag((10, 10), (30, 30))
    win.save()


def test_export_again_reuses_format_and_folder(
    make_window, image_dir, tmp_path, monkeypatch, driver
):
    win = make_window(Settings())
    win.open_path(image_dir)
    annotate(win, driver)
    out = tmp_path / "dataset"
    export_once(win, out, monkeypatch)
    (out / "annotations.json").unlink()
    from PySide6.QtWidgets import QFileDialog

    def must_not_ask(*args, **kwargs):
        raise AssertionError("Export Again must not ask for a folder")

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", must_not_ask)
    win.export_again()
    assert (out / "annotations.json").exists()


def test_export_settings_survive_reopening(make_window, image_dir, tmp_path, monkeypatch, driver):
    win = make_window(Settings())
    win.open_path(image_dir)
    annotate(win, driver)
    export_once(win, tmp_path / "yolo-out", monkeypatch, fmt="yolo")
    win.export_on_close_action.setChecked(True)
    win.close()
    again = make_window(Settings())
    again.open_path(image_dir)
    assert again.last_export == ("yolo", (tmp_path / "yolo-out").resolve())
    assert again.export_on_close_action.isChecked()


def test_export_on_close(make_window, image_dir, tmp_path, monkeypatch, driver):
    win = make_window(Settings())
    win.open_path(image_dir)
    annotate(win, driver)
    out = tmp_path / "masks-out"
    export_once(win, out, monkeypatch, fmt="masks")
    (out / "class_names.txt").unlink()
    win.export_on_close_action.setChecked(True)
    win.close()
    assert (out / "class_names.txt").exists()


def test_export_again_before_any_export_explains(make_window, image_dir):
    win = make_window(Settings())
    win.open_path(image_dir)
    win.export_again()
    assert "Export Folder As" in win.statusBar().currentMessage()


def test_unknown_export_format_in_project_is_ignored(make_window, image_dir):
    (image_dir / PROJECT_FILE).write_text(json.dumps({"export_format": "gone", "export_dir": "x"}))
    win = make_window(Settings())
    win.open_path(image_dir)
    assert win.last_export is None
