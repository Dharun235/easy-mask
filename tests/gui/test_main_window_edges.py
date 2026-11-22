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
    assert window.statusBar().currentMessage() == "This is the first image"
    for _ in range(5):
        window.next_image()
    assert window.image_path.name == "img10.png"
    assert window.statusBar().currentMessage() == "This is the last image"


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
    make_window().export_dialog()
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
    folder = window.image_path.parent.name
    assert window.windowTitle() == f"{folder}  ·  frame 1 of 3 (img1.png)  —  easy-mask"
    make_dirty(window, ui)
    assert "(img1.png) *" in window.windowTitle()
    window.settings.autosave = True  # move on without the save question
    window.next_image()
    assert "frame 2 of 3" in window.windowTitle()


def test_title_names_the_video_of_a_frames_folder(make_window, tmp_path):
    import shutil

    from PIL import Image

    frames = tmp_path / "wingbeat_frames"
    frames.mkdir()
    for i in range(2):
        Image.new("RGB", (8, 8)).save(frames / f"frame_{i:06d}.png")
    win = make_window()
    win.open_path(frames)
    assert win.windowTitle().startswith("wingbeat  ·  frame 1 of 2")
    win.close()
    shutil.rmtree(frames)


def test_arrow_keys_move_between_frames(window, qtbot):
    from PySide6.QtCore import Qt

    window.activateWindow()
    qtbot.keyClick(window.canvas, Qt.Key.Key_Right)
    assert window.current_index == 1
    qtbot.keyClick(window.canvas, Qt.Key.Key_Down)
    assert window.current_index == 2
    qtbot.keyClick(window.canvas, Qt.Key.Key_Up)
    qtbot.keyClick(window.canvas, Qt.Key.Key_Left)
    assert window.current_index == 0
    window.label_panel.add_edit.setFocus()
    qtbot.keyClick(window.label_panel.add_edit, Qt.Key.Key_Right)  # typing keeps its arrows
    assert window.current_index == 0


def test_backspace_removes_a_point_while_drawing_else_deletes(window, driver, qtbot):
    from PySide6.QtCore import Qt

    from easy_mask.core.shapes import ShapeType

    ui = driver(window)
    window.set_active_label("cat")
    window.set_tool(ShapeType.POLYGON)
    ui.click(10, 10)
    ui.click(30, 10)
    qtbot.keyClick(window.canvas, Qt.Key.Key_Backspace)
    assert len(window.canvas._draft) == 1 and window.canvas.shapes == []
    window.canvas.cancel_draft()
    window.set_tool(ShapeType.RECTANGLE)
    ui.drag((10, 10), (30, 30))
    window.canvas.select(0)
    qtbot.keyClick(window.canvas, Qt.Key.Key_Backspace)
    assert window.canvas.shapes == []
