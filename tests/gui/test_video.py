"""The Video menu: open a video, interpolate keyframes, propagate with a fake model."""

import cv2
import numpy as np
import pytest
from PySide6.QtCore import Qt

from easy_mask.core import annotation as ann_io
from easy_mask.core.annotation import Annotation
from easy_mask.core.shapes import Shape, ShapeType
from easy_mask.core.video import INTERPOLATED_FLAG, PROPAGATED_FLAG, is_auto

# The `window` fixture opens img1, img2, img10 (100 x 80): three "frames".
FRAMES = ("img1", "img2", "img10")


def rect(x, label="car", **flags):
    return Shape(label, ShapeType.RECTANGLE, [(x, 10), (x + 9, 19)], flags=flags)


def saved(image_dir, stem):
    path = image_dir / f"{stem}.json"
    return ann_io.load(path).shapes if path.exists() else None


def save_frame(image_dir, stem, shapes):
    ann_io.save(Annotation(f"{stem}.png", 100, 80, shapes), image_dir / f"{stem}.json")


@pytest.fixture
def video(window, fake_propagator):
    """The window's Video menu wired to the fake model, propagating as far as possible."""
    window.video.make_propagator = lambda: fake_propagator
    window.ask_number = lambda parent, title, label, value, lo, hi: hi
    return window.video


def propagate(qtbot, video, direction=+1):
    video.propagate(direction)
    qtbot.waitUntil(lambda: not video.running, timeout=5000)


def test_propagate_forward_writes_flagged_masks_to_later_frames(qtbot, window, video, image_dir):
    window.canvas.set_shapes([rect(10)])

    propagate(qtbot, video)

    for step, stem in enumerate(FRAMES[1:], start=1):
        (shape,) = saved(image_dir, stem)
        assert shape.shape_type == ShapeType.MASK and shape.flags == {PROPAGATED_FLAG: True}
        assert shape.points[0] == (10 + 2 * step, 10)  # the fake moves 2 px per frame
    assert window.file_list.item(2).text() != ""  # list refreshed without errors
    assert "Updated 2 frames" in window.statusBar().currentMessage()


def test_propagate_backward_from_last_frame(qtbot, window, video, image_dir):
    window._load(2)
    window.canvas.set_shapes([rect(50)])

    propagate(qtbot, video, -1)

    assert saved(image_dir, "img2")[0].points[0] == (52, 10)
    assert saved(image_dir, "img1")[0].points[0] == (54, 10)
    assert saved(image_dir, "img10") is None  # the start frame was not saved by us


def test_frames_per_run_limits_the_run(qtbot, window, video, image_dir):
    window.ask_number = lambda *args: 1
    video.ask_frames_per_run()
    window.canvas.set_shapes([rect(10)])
    propagate(qtbot, video)
    assert saved(image_dir, "img2") is not None and saved(image_dir, "img10") is None


def test_cancelled_frames_per_run_keeps_the_value(window, video):
    window.ask_number = lambda *args: None
    video.ask_frames_per_run()
    assert video.frames_per_run == 50


def test_hand_drawn_frames_are_reported_not_silently_skipped(qtbot, window, video, image_dir):
    for stem in FRAMES[1:]:
        save_frame(image_dir, stem, [rect(40)])  # hand-drawn car everywhere
    window.canvas.set_shapes([rect(10)])
    propagate(qtbot, video)
    assert "Nothing changed" in window.statusBar().currentMessage()
    assert not video.undo_run_action.isEnabled()


def test_nothing_to_propagate_explains(window, video):
    video.propagate(+1)
    assert not video.running
    assert "first" in window.statusBar().currentMessage()


def test_no_frames_in_that_direction(window, video):
    window.canvas.set_shapes([rect(10)])
    video.propagate(-1)
    assert not video.running and "No more frames" in window.statusBar().currentMessage()


def test_start_frame_is_confirmed_as_manual(qtbot, window, video):
    window.canvas.set_shapes([rect(10, propagated=True)])
    propagate(qtbot, video)
    assert not is_auto(window.canvas.shapes[0])


def test_repropagation_replaces_auto_shapes_but_keeps_manual_ones(
    qtbot, window, video, image_dir, fake_propagator
):
    save_frame(image_dir, "img2", [rect(0, propagated=True), rect(60, label="tree")])
    save_frame(image_dir, "img10", [rect(40)])  # a hand correction of the car
    window.canvas.set_shapes([rect(10)])

    propagate(qtbot, video)

    middle = saved(image_dir, "img2")
    assert [(s.label, s.flags) for s in middle] == [("tree", {}), ("car", {PROPAGATED_FLAG: True})]
    assert saved(image_dir, "img10") == [rect(40)]
    assert sorted(fake_propagator.calls[0][1]) == [0, 2]  # the correction was a prompt


def test_stop_cancels_after_the_current_frame(qtbot, window, video, image_dir, fake_propagator):
    original = fake_propagator.propagate

    def stopping(frames, prompts):
        for count, result in enumerate(original(frames, prompts)):
            if count == 1:
                video.stop()
            yield result

    fake_propagator.propagate = stopping
    window.canvas.set_shapes([rect(10)])
    propagate(qtbot, video)
    assert saved(image_dir, "img2") is not None and saved(image_dir, "img10") is None


def test_ai_propagate_without_extra_offers_copy(qtbot, window, video, image_dir, dialogs):
    from PySide6.QtWidgets import QMessageBox

    def missing():
        raise ImportError("No module named 'torch'")

    video.make_propagator = missing
    window.canvas.set_shapes([rect(10)])
    dialogs.answer = QMessageBox.StandardButton.No  # declined: nothing happens
    video.propagate(+1)
    assert "uv sync --extra video" in dialogs.questions[0]
    assert not video.running and saved(image_dir, "img2") is None

    dialogs.answer = QMessageBox.StandardButton.Yes  # accepted: copies instead
    video.propagate(+1)
    qtbot.waitUntil(lambda: not video.running, timeout=5000)
    for stem in FRAMES[1:]:
        (shape,) = saved(image_dir, stem)
        assert shape.flags == {PROPAGATED_FLAG: True}
        assert shape.points == [(10, 10), (19, 19)]  # same pixels, not moved


def test_copy_to_frames_never_uses_the_model(qtbot, window, video, image_dir, fake_propagator):
    window.canvas.set_shapes([rect(10)])
    video.ai_action.setChecked(False)
    video.propagate(+1)
    qtbot.waitUntil(lambda: not video.running, timeout=5000)
    assert fake_propagator.calls == []
    assert saved(image_dir, "img2")[0].points == [(10, 10), (19, 19)]


def test_tracking_without_extra_copies_without_asking(qtbot, window, image_dir):
    def missing():
        raise ImportError("No module named 'torch'")

    window.video.make_propagator = missing
    window.settings.autosave = True
    window.canvas.set_shapes([rect(10)])
    window.video.track_action.setChecked(True)
    window.next_image()  # the dialogs fixture would fail on any question
    qtbot.waitUntil(lambda: not window.video.running, timeout=5000)
    assert window.canvas.shapes[0].points == [(10, 10), (19, 19)]


def test_copy_model_can_be_chosen_explicitly(qtbot, window, image_dir):
    from easy_mask.gui.video_menu import default_propagator
    from easy_mask.propagation.tracker import CopyPropagator

    assert isinstance(default_propagator("copy"), CopyPropagator)


def test_only_selected_objects_are_propagated(qtbot, window, video, image_dir):
    window.canvas.set_shapes([rect(10, label="car"), rect(40, label="bike")])
    window.canvas.select(1)

    propagate(qtbot, video)

    (shape,) = saved(image_dir, "img2")
    assert shape.label == "bike"
    (kept_car, kept_bike) = window.canvas.shapes  # the start frame keeps both objects
    assert (kept_car.label, kept_bike.label) == ("car", "bike")


def test_model_failure_is_reported(qtbot, window, video, dialogs, fake_propagator):
    def broken(frames, prompts):
        raise RuntimeError("out of memory")
        yield  # pragma: no cover

    dialogs.allow_warnings = True
    fake_propagator.propagate = broken
    window.canvas.set_shapes([rect(10)])
    propagate(qtbot, video)
    assert dialogs.warnings == ["RuntimeError: out of memory"]


def test_propagator_is_built_once(qtbot, window, video):
    built = []
    make = video.make_propagator
    video.make_propagator = lambda: built.append(1) or make()
    window.canvas.set_shapes([rect(10)])
    propagate(qtbot, video)
    propagate(qtbot, video)
    assert built == [1]


def test_result_for_the_frame_on_screen_goes_to_the_canvas(window, video):
    window.canvas.set_shapes([rect(10, propagated=True), rect(60, label="tree")])
    video._store(0, {("car", None)}, [rect(20, propagated=True)])
    assert window.canvas.shapes == [rect(60, label="tree"), rect(20, propagated=True)]
    assert window.dirty


def test_closing_during_a_run_waits_for_the_worker(qtbot, make_window, image_dir, fake_propagator):
    win = make_window()
    win.open_path(image_dir)
    win.video.make_propagator = lambda: fake_propagator
    win.ask_number = lambda *args: 2
    win.canvas.set_shapes([rect(10)])
    win.video.propagate(+1)
    win.dirty = False
    win.close()
    assert win.video.worker is None or win.video.worker.isFinished()


# ---------------------------------------------------------------------- interpolation


def test_interpolate_fills_frames_between_keyframes(window, image_dir):
    save_frame(image_dir, "img1", [rect(0)])
    save_frame(image_dir, "img2", [rect(90, label="old", interpolated=True)])
    window._load(2)
    window.canvas.set_shapes([rect(40)])

    window.video.interpolate()

    middle = saved(image_dir, "img2")
    assert [s.label for s in middle] == ["old", "car"]
    assert middle[1].flags == {INTERPOLATED_FLAG: True}
    assert middle[1].points == [(20, 10), (29, 19)]
    assert "Interpolated 1 frames" in window.statusBar().currentMessage()


def test_interpolate_skips_frames_with_only_auto_shapes(window, image_dir):
    save_frame(image_dir, "img1", [rect(0, propagated=True)])
    window._load(2)
    window.canvas.set_shapes([rect(40)])
    window.video.interpolate()
    assert "No earlier keyframe" in window.statusBar().currentMessage()


# ---------------------------------------------------------------------- opening videos


def test_open_video_extracts_and_shows_frames(qtbot, make_window, tmp_path):
    video_path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    for k in range(6):
        writer.write(np.full((48, 64, 3), 30 * k, np.uint8))
    writer.release()
    from easy_mask.gui.main_window import Settings

    win = make_window(Settings(frame_step=2))
    win.ask_video_import = lambda defaults, info, parent: defaults  # accept the defaults
    win.open_path(video_path)
    qtbot.waitUntil(lambda: win.video_worker is None, timeout=5000)
    assert [p.name for p in win.image_paths] == [
        "frame_000000.jpg",
        "frame_000002.jpg",
        "frame_000004.jpg",
    ]
    assert win.image_size == (64, 48)


def test_open_unreadable_video_warns(make_window, tmp_path, dialogs):
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"nope")
    dialogs.allow_warnings = True
    win = make_window()
    win.open_path(broken)
    assert win.image_path is None and "Could not open video" in dialogs.warnings[0]


def test_worker_body_runs_in_the_calling_thread(fake_propagator, image_dir):
    """run() directly (no thread) so coverage sees it; the GUI tests above use start()."""
    from easy_mask.gui.propagate import PropagationWorker

    frames = [image_dir / f"{stem}.png" for stem in FRAMES]
    worker = PropagationWorker(fake_propagator, frames, [[rect(10)], [], []], (80, 100))
    done = []
    worker.frame_done.connect(lambda position, shapes: done.append(position))
    worker.run()
    assert done == [1, 2]
    worker.cancel()
    worker.run()
    assert done == [1, 2]


def test_shape_list_marks_auto_shapes(window):
    window.canvas.set_shapes([rect(10), rect(30, propagated=True)])
    window._refresh_panels()
    texts = [window.shape_list.item(i).text() for i in range(2)]
    assert "(auto)" not in texts[0] and texts[1].endswith("(auto)")


def test_default_backend_without_video_extra_offers_copy(
    qtbot, window, image_dir, monkeypatch, dialogs
):
    import importlib.util

    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    window.ask_number = lambda *args: 1
    dialogs.answer = QMessageBox.StandardButton.Yes
    window.canvas.set_shapes([rect(10)])
    window.video.propagate(+1)
    assert window.video.propagator is None  # the model was not loaded
    qtbot.waitUntil(lambda: not window.video.running, timeout=5000)
    assert saved(image_dir, "img2")[0].points == [(10, 10), (19, 19)]


def test_tracking_carries_objects_into_next_frame(qtbot, window, video, image_dir):
    window.settings.autosave = True
    window.canvas.set_shapes([rect(10), rect(40, label="bike")])
    window.video.track_action.setChecked(True)
    window.next_image()
    qtbot.waitUntil(lambda: not video.running, timeout=5000)
    assert window.image_path.name == "img2.png"
    labels = sorted(s.label for s in window.canvas.shapes)
    assert labels == ["bike", "car"]
    assert all(s.flags == {PROPAGATED_FLAG: True} for s in window.canvas.shapes)
    car = next(s for s in window.canvas.shapes if s.label == "car")
    assert car.points[0] == (12, 10)  # the fake model moved it 2 px


def test_tracking_only_selected_objects(qtbot, window, video):
    window.settings.autosave = True
    window.canvas.set_shapes([rect(10), rect(40, label="bike")])
    window.canvas.select(1)
    window.video.track_action.setChecked(True)
    window.next_image()
    qtbot.waitUntil(lambda: not video.running, timeout=5000)
    assert [s.label for s in window.canvas.shapes] == ["bike"]


def test_tracking_off_or_going_back_does_nothing(qtbot, window, video, fake_propagator):
    window.settings.autosave = True
    window.canvas.set_shapes([rect(10)])
    window.next_image()  # tracking is off by default
    assert window.canvas.shapes == [] and fake_propagator.calls == []
    window.video.track_action.setChecked(True)
    window.canvas.set_shapes([rect(10)])
    window.prev_image()  # only Next Image tracks
    assert fake_propagator.calls == []


def test_tracking_never_overwrites_hand_drawn_shapes(qtbot, window, video, image_dir):
    save_frame(image_dir, "img2", [rect(60)])  # the user already drew the car on img2
    window.settings.autosave = True
    window.canvas.set_shapes([rect(10)])
    window.video.track_action.setChecked(True)
    window.next_image()
    qtbot.waitUntil(lambda: not video.running, timeout=5000)
    (car,) = window.canvas.shapes
    assert car.points[0] == (60, 10) and not is_auto(car)


def test_tracking_toggle_is_remembered(make_window):
    from easy_mask.gui.main_window import Settings

    first = make_window(Settings())
    first.video.track_action.setChecked(True)
    first.session.save()
    assert make_window(Settings()).video.tracking


# ---------------------------------------------------------------------- cut, clear, undo a run


def wing_and_leg(window):
    """A wide 'wing' and a 'leg' over its right part (different labels)."""
    wing = Shape("wing", ShapeType.RECTANGLE, [(10, 10), (49, 29)])
    leg = Shape("leg", ShapeType.RECTANGLE, [(40, 20), (49, 39)])
    window.canvas.set_shapes([wing, leg])
    window._commit()  # an undo step, as if the user had drawn them


def pixels(shapes, label):
    from easy_mask.core.rasterize import shape_to_mask

    found = [s for s in shapes if s.label == label]
    return shape_to_mask(found[0], 80, 100) if found else np.zeros((80, 100), bool)


def lasso(window, x0, x1, y0=0, y1=79):
    """Erase the box x0..x1, y0..y1 (what an eraser stroke or an Erase shape reports)."""
    region = np.zeros((80, 100), bool)
    region[y0 : y1 + 1, x0 : x1 + 1] = True
    window.cut(region)


def test_erasing_cuts_every_mask_under_it_without_asking_a_label(window):
    window.ask_label = lambda *args: pytest.fail("the cut tool must not ask for a label")
    wing_and_leg(window)
    lasso(window, 35, 60)
    wing = pixels(window.canvas.shapes, "wing")
    assert wing[15, 20] and not wing[15, 40]
    assert [s.label for s in window.canvas.shapes] == ["wing"]  # leg cut away completely
    window.undo()
    assert len(window.canvas.shapes) == 2


def test_cut_with_a_selection_only_cuts_those_objects(window):
    wing_and_leg(window)
    window.canvas.select(0)
    lasso(window, 35, 60)
    assert not pixels(window.canvas.shapes, "wing")[15, 40]
    assert pixels(window.canvas.shapes, "leg")[30, 45]


def test_cut_outside_every_mask_changes_nothing(window):
    wing_and_leg(window)
    lasso(window, 70, 90)
    assert len(window.canvas.shapes) == 2
    assert "No mask there" in window.statusBar().currentMessage()


def test_propagate_carries_the_cut_like_an_object(qtbot, window, video, image_dir):
    for stem in FRAMES[1:]:
        save_frame(image_dir, stem, [Shape("wing", ShapeType.RECTANGLE, [(10, 10), (59, 29)])])
    wing_and_leg(window)
    lasso(window, 35, 99)
    assert video.last_cut is not None
    part = video.last_cut.part
    assert part[20, 40] and not part[5, 40]  # only the removed pixels, not the whole area
    propagate(qtbot, video)  # the fake model moves every object 2 px per frame
    frame2 = pixels(saved(image_dir, "img2"), "wing")
    assert frame2[20, 36] and not frame2[20, 38]  # the cut moved from x=35 to x=37


def test_cut_is_carried_by_track_too(qtbot, window, video, image_dir):
    save_frame(image_dir, "img2", [Shape("wing", ShapeType.RECTANGLE, [(10, 10), (59, 29)])])
    window.settings.autosave = True
    wing_and_leg(window)
    lasso(window, 35, 99)
    video.track_action.setChecked(True)
    window.next_image()
    qtbot.waitUntil(lambda: not video.running, timeout=5000)
    here = pixels(window.canvas.shapes, "wing")
    assert here[20, 30] and not here[20, 40]


def test_two_cuts_on_a_frame_are_carried_together(window, video):
    wing_and_leg(window)
    lasso(window, 40, 49, 10, 15)
    lasso(window, 10, 15, 10, 29)
    part = video.last_cut.part
    assert part[12, 45] and part[20, 12]


def test_clear_frames_and_one_undo_puts_them_back(window, video, image_dir):
    save_frame(image_dir, "img2", [rect(10)])
    window.canvas.set_shapes([rect(30)])
    window._commit()
    window.ask_number = lambda *args: 2
    video.clear_frames()
    assert window.canvas.shapes == [] and saved(image_dir, "img2") == []
    window.undo()
    assert len(window.canvas.shapes) == 1 and len(saved(image_dir, "img2")) == 1


def test_undo_after_propagating_reverts_every_frame(qtbot, window, video, image_dir):
    window.canvas.set_shapes([rect(10)])
    window._commit()
    propagate(qtbot, video)
    assert saved(image_dir, "img2") is not None
    window.undo()
    assert saved(image_dir, "img2") is None and saved(image_dir, "img10") is None
    assert window.file_list.item(1).checkState() == Qt.CheckState.Unchecked
    assert len(window.canvas.shapes) == 1  # this frame keeps its own shape
    assert "Undid Propagate" in window.statusBar().currentMessage()


def test_editing_after_a_run_makes_undo_local_again(qtbot, window, video, image_dir):
    window.canvas.set_shapes([rect(10)])
    window._commit()
    propagate(qtbot, video)
    window.canvas.set_shapes([rect(10), rect(40)])
    window._commit()
    window.undo()  # undoes the edit, not the run
    assert len(window.canvas.shapes) == 1 and saved(image_dir, "img2") is not None
    assert video.undo_run_action.isEnabled()
    video.undo_run_action.trigger()  # the run can still be undone explicitly
    assert saved(image_dir, "img2") is None


def test_masks_a_cut_fixes_do_not_steer_the_run(qtbot, window, video, image_dir, fake_propagator):
    """Regression: later hand-drawn masks still holding the part pulled it back."""
    for stem in FRAMES[1:]:
        save_frame(image_dir, stem, [Shape("wing", ShapeType.RECTANGLE, [(10, 10), (59, 29)])])
    wing_and_leg(window)
    lasso(window, 35, 99)
    propagate(qtbot, video)
    assert sorted(fake_propagator.calls[0][1]) == [0]  # only this frame prompts the model


def test_propagate_a_polygon_then_erase_with_it(qtbot, window, video, image_dir):
    """Draw a shape, propagate it, then use it to cut the other masks on every frame."""
    for stem in FRAMES[1:]:
        save_frame(image_dir, stem, [Shape("wing", ShapeType.RECTANGLE, [(10, 10), (59, 29)])])
    wing = Shape("wing", ShapeType.RECTANGLE, [(10, 10), (59, 29)])
    leg = Shape("leg", ShapeType.RECTANGLE, [(40, 5), (49, 35)])
    window.canvas.set_shapes([wing, leg])
    window._commit()
    window.canvas.select(1)
    propagate(qtbot, video)  # only the leg goes (selected); the fake moves it 2 px per frame
    window.canvas.select(1)
    video.erase_with(1)
    assert [s.label for s in window.canvas.shapes] == ["wing"]
    assert not pixels(window.canvas.shapes, "wing")[20, 45]
    for stem in FRAMES[1:]:
        later = saved(image_dir, stem)
        assert [s.label for s in later] == ["wing"]  # the leg copy is gone
    assert not pixels(saved(image_dir, "img2"), "wing")[20, 47]  # cut where the leg moved
    window.undo()  # one undo: every frame back
    assert len(saved(image_dir, "img2")) == 2 and len(window.canvas.shapes) == 2


def test_right_click_offers_erase_with_this(window, ui):
    window.canvas.set_shapes([rect(10)])
    shown = []
    window.show_menu = lambda menu, pos: shown.append(menu)
    ui.click(12, 12, Qt.MouseButton.RightButton)
    (erase,) = [a for a in shown[0].actions() if a.text() == "Erase With This"]
    assert erase.isEnabled()
