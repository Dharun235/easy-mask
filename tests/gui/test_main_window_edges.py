"""Main-window behaviour around errors, prompts and file dialogs."""

import pytest
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QFileDialog, QMessageBox

from easy_mask.core.shapes import ShapeType
from easy_mask.gui.main_window import Settings

Btn = QMessageBox.StandardButton


def make_dirty(window, ui):
    window.set_tool(ShapeType.POINT)
    ui.click(10, 10)
    assert window.dirty


def test_open_folder_without_images_warns(make_window, tmp_path, dialogs):
    dialogs.allow_warnings = True
    make_window().open_path(tmp_path)
    assert "No images found" in dialogs.warnings[0]


def test_open_non_image_warns(make_window, image_dir, dialogs):
    dialogs.allow_warnings = True
    make_window().open_path(image_dir / "notes.txt")
    assert "Not an image" in dialogs.warnings[0]


def test_corrupt_annotation_warns_and_is_not_overwritten(make_window, image_dir, dialogs):
    dialogs.allow_warnings = True
    bad = image_dir / "img1.json"
    bad.write_text("{ not json")
    win = make_window(Settings(autosave=True))
    win.open_path(image_dir)
    assert any("Could not read annotation" in w for w in dialogs.warnings)
    assert bad.read_text() == "{ not json"


def test_unreadable_image_warns(make_window, image_dir, dialogs):
    dialogs.allow_warnings = True
    (image_dir / "img1.png").write_bytes(b"not a png")
    make_window().open_path(image_dir)
    assert any("Could not read image" in w for w in dialogs.warnings)


@pytest.mark.parametrize(
    ("answer", "moves", "saved"),
    [(Btn.Save, True, True), (Btn.Discard, True, False), (Btn.Cancel, False, False)],
)
def test_unsaved_prompt_answers(window, ui, dialogs, image_dir, answer, moves, saved):
    make_dirty(window, ui)
    dialogs.answer = answer
    window.next_image()
    assert dialogs.questions == ["Save changes to img1.png?"]
    assert (window.image_path.name == "img2.png") is moves
    assert (image_dir / "img1.json").exists() is saved


def test_close_with_cancel_keeps_window_open(window, ui, dialogs):
    make_dirty(window, ui)
    dialogs.answer = Btn.Cancel
    event = QCloseEvent()
    window.closeEvent(event)
    assert not event.isAccepted()


def test_file_list_click_navigates_and_cancel_restores(window, ui, dialogs):
    window.file_list.setCurrentRow(2)
    assert window.image_path.name == "img10.png"
    make_dirty(window, ui)
    dialogs.answer = Btn.Cancel
    window.file_list.setCurrentRow(0)
    assert window.image_path.name == "img10.png"
    assert window.file_list.currentRow() == 2


def test_step_stops_at_ends(window):
    window.prev_image()
    assert window.image_path.name == "img1.png"
    for _ in range(5):
        window.next_image()
    assert window.image_path.name == "img10.png"


def test_open_dialogs(make_window, image_dir, monkeypatch):
    win = make_window()
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", lambda *a, **k: (str(image_dir / "img2.png"), "")
    )
    win.open_image_dialog()
    assert win.image_path.name == "img2.png"
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: str(image_dir))
    win.open_folder_dialog()
    assert win.image_path.name == "img1.png"
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: "")
    win.open_folder_dialog()  # cancelled: nothing changes
    assert win.image_path.name == "img1.png"


def test_export_without_image_warns(make_window, dialogs):
    dialogs.allow_warnings = True
    make_window().export_dataset("coco")
    assert dialogs.warnings == ["Open a folder first."]


def test_edit_commands_without_selection_are_noops(window):
    for command in (
        window.edit_label,
        window.delete_selected,
        window.duplicate_selected,
        window.copy_selected,
        window.paste,
        window.undo,
        window.redo,
    ):
        command()
    assert window.canvas.shapes == [] and not window.dirty


def test_save_failure_warns(window, ui, dialogs, monkeypatch):
    dialogs.allow_warnings = True
    make_dirty(window, ui)

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr("easy_mask.core.annotation.save", boom)
    assert window.save() is False
    assert window.dirty and "disk full" in dialogs.warnings[0]


def test_about_and_title(window, ui):
    window._about()
    assert window.windowTitle() == "easy-mask - img1.png"
    make_dirty(window, ui)
    assert window.windowTitle().endswith("*")
