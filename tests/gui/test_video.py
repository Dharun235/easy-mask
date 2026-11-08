"""The Video menu: open a video, interpolate keyframes, propagate with a fake model."""

import cv2
import numpy as np
import pytest

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


def test_video_menu_sits_before_help(window):
    titles = [action.text() for action in window.menuBar().actions()]
    assert titles[-1] == "&Help" and titles.index("&Video") > titles.index("&View")


def test_propagate_forward_writes_flagged_masks_to_later_frames(qtbot, window, video, image_dir):
    window.canvas.set_shapes([rect(10)])

    propagate(qtbot, video)

    for step, stem in enumerate(FRAMES[1:], start=1):
        (shape,) = saved(image_dir, stem)
        assert shape.shape_type == ShapeType.MASK and shape.flags == {PROPAGATED_FLAG: True}
        assert shape.points[0] == (10 + 2 * step, 10)  # the fake moves 2 px per frame
    assert window.file_list.item(2).text() != ""  # list refreshed without errors
    assert "finished" in window.statusBar().currentMessage()


def test_propagate_backward_from_last_frame(qtbot, window, video, image_dir):
    window._load(2)
    window.canvas.set_shapes([rect(50)])

    propagate(qtbot, video, -1)

    assert saved(image_dir, "img2")[0].points[0] == (52, 10)
    assert saved(image_dir, "img1")[0].points[0] == (54, 10)
    assert saved(image_dir, "img10") is None  # the start frame was not saved by us


def test_frames_to_propagate_limits_the_run(qtbot, window, video, image_dir):
    window.ask_number = lambda *args: 1
    window.canvas.set_shapes([rect(10)])
    propagate(qtbot, video)
    assert saved(image_dir, "img2") is not None and saved(image_dir, "img10") is None


def test_cancelled_frame_count_does_nothing(window, video, fake_propagator):
    window.ask_number = lambda *args: None
    window.canvas.set_shapes([rect(10)])
    video.propagate(+1)
    assert not video.running and fake_propagator.calls == []


def test_nothing_to_propagate_explains(window, video):
    video.propagate(+1)
    assert not video.running
    assert "first" in window.statusBar().currentMessage()


def test_no_frames_in_that_direction(window, video):
    window.canvas.set_shapes([rect(10)])
    video.propagate(-1)
    assert not video.running and "No frames" in window.statusBar().currentMessage()


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
    video.propagate(+1, ai=False)
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


# ---------------------------------------------------------------------- subtract through frames


def wing_and_cutter(window):
    """A wide 'wing' rectangle and a polygon over its right part, selected wing-first."""
    wing = Shape("wing", ShapeType.RECTANGLE, [(10, 10), (49, 29)])
    cutter = Shape("wing", ShapeType.POLYGON, [(35, 5), (55, 5), (55, 35), (35, 35)])
    window.canvas.set_shapes([wing, cutter])
    window._commit()  # an undo step, as if the user had drawn them
    window.canvas.set_selection([0, 1])


def wing_pixels(shapes):
    from easy_mask.core import masks as m

    (wing,) = [s for s in shapes if s.label == "wing"]
    return m.to_full(wing, 80, 100)


def test_subtract_copy_cuts_this_and_next_frames(qtbot, window, video, image_dir):
    for stem in FRAMES[1:]:
        save_frame(image_dir, stem, [Shape("wing", ShapeType.RECTANGLE, [(10, 10), (49, 29)])])
    wing_and_cutter(window)
    video.edit_through("subtract", ai=False)
    qtbot.waitUntil(lambda: not video.running, timeout=5000)
    here = wing_pixels(window.canvas.shapes)
    assert len(window.canvas.shapes) == 1  # the cutting polygon is gone
    assert here[20, 20] and not here[20, 40]
    for stem in FRAMES[1:]:
        later = wing_pixels(saved(image_dir, stem))
        assert later[20, 20] and not later[20, 40]


def test_subtract_ai_follows_the_moving_region(qtbot, window, video, image_dir):
    for stem in FRAMES[1:]:
        save_frame(image_dir, stem, [Shape("wing", ShapeType.RECTANGLE, [(10, 10), (59, 29)])])
    wing_and_cutter(window)
    video.edit_through("subtract")  # the fake model moves the region 2 px per frame
    qtbot.waitUntil(lambda: not video.running, timeout=5000)
    frame2 = wing_pixels(saved(image_dir, "img2"))
    assert frame2[20, 36] and not frame2[20, 38]  # cut moved from x=35 to x=37


def test_subtract_needs_target_then_cutter(window, video):
    window.canvas.set_shapes([Shape("wing", ShapeType.RECTANGLE, [(10, 10), (49, 29)])])
    window.canvas.select(0)
    video.edit_through("subtract")
    assert "select the mask to fix" in window.statusBar().currentMessage()


def test_subtract_this_frame_only_with_zero_frames(window, video):
    window.ask_number = lambda *args: 0
    wing_and_cutter(window)
    video.edit_through("subtract")
    assert not video.running
    assert not wing_pixels(window.canvas.shapes)[20, 40]
    window.undo()
    assert len(window.canvas.shapes) == 2  # one undo brings back the wing and the cutter


def test_subtract_cancelled_dialog_changes_nothing(window, video):
    window.ask_number = lambda *args: None
    wing_and_cutter(window)
    video.edit_through("subtract")
    assert len(window.canvas.shapes) == 2


def test_add_through_frames_grows_the_object_and_creates_it_where_missing(
    qtbot, window, video, image_dir
):
    save_frame(image_dir, "img2", [Shape("wing", ShapeType.RECTANGLE, [(10, 10), (29, 29)])])
    # img10 has no wing at all: Add creates it there from the region
    wing = Shape("wing", ShapeType.RECTANGLE, [(10, 10), (29, 29)])
    paw = Shape("wing", ShapeType.RECTANGLE, [(40, 10), (49, 29)])
    window.canvas.set_shapes([wing, paw])
    window._commit()
    window.canvas.set_selection([0, 1])
    video.edit_through("add", ai=False)
    qtbot.waitUntil(lambda: not video.running, timeout=5000)
    here = wing_pixels(window.canvas.shapes)
    assert here[20, 20] and here[20, 45] and len(window.canvas.shapes) == 1
    frame2 = wing_pixels(saved(image_dir, "img2"))
    assert frame2[20, 20] and frame2[20, 45]
    (created,) = saved(image_dir, "img10")
    assert created.flags.get(PROPAGATED_FLAG) and wing_pixels([created])[20, 45]
    assert not wing_pixels([created])[20, 20]  # only the added region exists there


def test_add_hint_names_the_operation(window, video):
    window.canvas.set_shapes([Shape("wing", ShapeType.RECTANGLE, [(10, 10), (49, 29)])])
    window.canvas.select(0)
    video.edit_through("add")
    assert window.statusBar().currentMessage().startswith("Add Through Frames")
