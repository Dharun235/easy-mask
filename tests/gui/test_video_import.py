"""Import a video as images: the dialog, the background worker and the flow."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from easy_mask.gui.main_window import Settings
from easy_mask.gui.video_import import (
    ImportOptions,
    VideoImportDialog,
    VideoImportWorker,
    default_options,
)
from easy_mask.io.images import list_images
from easy_mask.io.video import probe


@pytest.fixture
def clip(tmp_path) -> Path:
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    for k in range(8):
        writer.write(np.full((48, 64, 3), 25 * k, np.uint8))
    writer.release()
    return path


def test_dialog_shows_plan_and_returns_choices(qtbot, clip, tmp_path):
    dialog = VideoImportDialog(default_options(clip), probe(clip))
    qtbot.addWidget(dialog)
    assert dialog.summary.text() == "Will write 8 JPG images."
    dialog.format_box.setCurrentIndex(dialog.format_box.findData("png"))
    assert not dialog.quality_spin.isEnabled()  # PNG is lossless
    dialog.step_spin.setValue(3)
    dialog.max_spin.setValue(2)
    dialog.out_edit.setText(str(tmp_path / "pngs"))
    dialog.open_box.setChecked(False)
    assert dialog.summary.text() == "Will write 2 PNG images."
    options = dialog.options()
    assert (options.fmt, options.step, options.max_frames, options.open_after) == (
        "png",
        3,
        2,
        False,
    )
    assert options.out_dir == tmp_path / "pngs"


def test_dialog_browse_and_ask(qtbot, clip, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QDialog, QFileDialog

    dialog = VideoImportDialog(default_options(clip), probe(clip))
    qtbot.addWidget(dialog)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: str(tmp_path / "x"))
    dialog._browse()
    assert dialog.out_edit.text() == str(tmp_path / "x")
    monkeypatch.setattr(VideoImportDialog, "exec", lambda self: QDialog.DialogCode.Rejected)
    assert VideoImportDialog.ask(default_options(clip), probe(clip)) is None
    monkeypatch.setattr(VideoImportDialog, "exec", lambda self: QDialog.DialogCode.Accepted)
    assert VideoImportDialog.ask(default_options(clip), probe(clip)).video == clip


def test_import_png_to_chosen_folder_then_open(qtbot, make_window, clip, tmp_path):
    out = tmp_path / "frames-png"
    win = make_window(Settings())
    win.ask_video_import = lambda d, info, parent: ImportOptions(clip, out, fmt="png", step=2)
    win.open_path(clip)
    assert win.progress.is_running("video")
    qtbot.waitUntil(lambda: win.video_worker is None, timeout=5000)
    assert [p.name for p in win.image_paths] == [f"frame_00000{i}.png" for i in (0, 2, 4, 6)]
    assert not win.progress.is_running("video")


def test_import_without_opening(qtbot, make_window, clip, tmp_path):
    out = tmp_path / "frames"
    win = make_window(Settings())
    win.ask_video_import = lambda d, info, parent: ImportOptions(clip, out, open_after=False)
    win.import_video(clip)
    qtbot.waitUntil(lambda: win.video_worker is None, timeout=5000)
    assert len(list_images(out)) == 8 and win.image_path is None
    assert "Imported 8 images" in win.statusBar().currentMessage()


def test_cancelled_dialog_does_nothing(make_window, clip):
    win = make_window(Settings())
    win.ask_video_import = lambda d, info, parent: None
    win.open_path(clip)
    assert win.video_worker is None and not list_images(clip.parent)


def test_reopening_an_imported_video_skips_the_dialog(qtbot, make_window, clip):
    win = make_window(Settings())
    win.ask_video_import = lambda d, info, parent: d
    win.open_path(clip)
    qtbot.waitUntil(lambda: win.video_worker is None, timeout=5000)

    def must_not_ask(*args):
        raise AssertionError("frames exist: no dialog")

    again = make_window(Settings())
    again.ask_video_import = must_not_ask
    again.open_path(clip)
    assert again.image_path is not None


def test_folder_with_images_is_opened_not_overwritten(make_window, clip, image_dir):
    win = make_window(Settings())
    win.ask_video_import = lambda d, info, parent: ImportOptions(clip, image_dir)
    win.import_video(clip)
    assert win.video_worker is None and win.image_path.parent == image_dir


def test_worker_cancel_and_failure(qtbot, clip, tmp_path):
    worker = VideoImportWorker(ImportOptions(clip, tmp_path / "out"))
    worker.cancel()
    worker.start()
    worker.wait()
    assert len(list_images(tmp_path / "out")) == 1  # stops after the first frame
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"nope")
    failing = VideoImportWorker(ImportOptions(broken, tmp_path / "b"))
    messages = []
    failing.failed.connect(messages.append)
    failing.start()
    failing.wait()
    qtbot.wait(10)
    assert messages and "Could not open video" in messages[0]


def test_file_menu_import_entry(make_window, clip, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    win = make_window(Settings())
    asked = []
    win.ask_video_import = lambda d, info, parent: asked.append(d) or None
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(clip), ""))
    win.import_video_dialog()
    assert asked and asked[0].video == clip
