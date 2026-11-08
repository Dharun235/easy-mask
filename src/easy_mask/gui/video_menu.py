"""Video: one Propagate button carries this frame's work through the next frames.

A video is just the folder of frames the main window already shows, so everything here
works frame by frame on the same per-image JSON files.

**Propagate** sends the selected objects (or all of them) *and the last cut made on this
frame* through the next frames together: the cut part is tracked like one more object and
cut from the masks on every frame. The button's dropdown holds the rest: backward, AI or
plain copying, auto-propagate on Next, how many frames, interpolate, clear and undo; the
progress bar's Cancel stops a run.

Shapes made automatically are flagged (see :mod:`easy_mask.core.video`) and replaced on the
next run; hand-drawn ones are kept and steer the model. Every run is one step in the
window's undo history, so one Undo restores every frame it changed.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtWidgets import QFileDialog, QMenu, QMessageBox

from easy_mask.core import annotation as ann_io
from easy_mask.core import masks
from easy_mask.core.interpolate import interpolate_frames
from easy_mask.core.keypoints import is_keypoint, with_keypoints
from easy_mask.core.rasterize import shape_to_mask
from easy_mask.core.shapes import Shape
from easy_mask.core.video import (
    ObjectKey,
    confirm,
    cut_region,
    is_auto,
    merge_auto,
    object_key,
)
from easy_mask.gui.propagate import PropagationWorker
from easy_mask.gui.video_import import VIDEO_FILTER
from easy_mask.propagation.tracker import CopyPropagator, Propagator

if TYPE_CHECKING:  # the main window imports this module
    from easy_mask.gui.main_window import MainWindow

DEFAULT_RUN = 50  # frames per propagation run: a few minutes on a laptop CPU
COPY_MODEL = "copy"  # --video-model value for model-free propagation
MISSING_EXTRA = (
    "AI tracking uses EfficientTAM (or SAM 2.1), which needs the optional video add-on "
    "(PyTorch, about 1 GB):\n\n    SAM2_BUILD_CUDA=0 uv sync --extra video"
)
REGION_LABEL = "__region__"  # the cut part, tracked like an object during a run
NOTHING_HINT = "Draw an object (or erase a part) on this frame first, then press Propagate"
FALLBACK_NOTE = "AI tracking needs the video add-on (uv sync --extra video): copying instead"


@dataclass
class Cut:
    """A part cut out on one frame: the pixels removed and from which objects (None: all)."""

    frame: int
    part: np.ndarray
    keys: set[ObjectKey] | None


@dataclass
class Carry:
    """What a run starts from: the frame, its shapes, the objects to send and a cut."""

    frame: int
    shapes: list[Shape]
    objects: set[ObjectKey]
    cut: Cut | None


def default_propagator(model: str) -> Propagator:
    """The chosen backend: the real model (needs PyTorch) or plain copying."""
    if model == COPY_MODEL:
        return CopyPropagator()
    from easy_mask.propagation.backend import Sam2Propagator

    return Sam2Propagator(model)


class VideoMenu:
    """The Propagate button, its dropdown and the Track toggle for ``window``."""

    def __init__(self, window: "MainWindow"):
        self.window = window
        # Swappable so tests can use a fake model.
        self.make_propagator: Callable[[], Propagator] = lambda: default_propagator(
            window.settings.video_model
        )
        self.propagator: Propagator | None = None  # the AI model: built once, then reused
        self.copier = CopyPropagator()  # plain copying, and the fallback without the extra
        self.worker: PropagationWorker | None = None
        self.frames_per_run = DEFAULT_RUN
        self.last_cut: Cut | None = None  # carried along by the next run from its frame
        self.changed: set[int] = set()  # frames the current / last run changed
        self._run_frames = 0

        menu = QMenu("Propagate", window)  # the Propagate button's dropdown
        self.menu = menu
        a = window._action
        self.propagate_action = a(menu, "Propagate Forward", lambda: self.propagate(+1))
        backward = a(menu, "Propagate Backward", lambda: self.propagate(-1))
        menu.addSeparator()
        self.ai_action = a(menu, "Use AI (EfficientTAM)", self._toggle_ai, checkable=True)
        self.ai_action.setChecked(window.settings.video_model != COPY_MODEL)
        self.track_action = a(
            menu, "Auto-Propagate on Next Image", self._toggle_tracking, checkable=True
        )
        frames = a(menu, "How Many Frames...", self.ask_frames_per_run)
        menu.addSeparator()
        interpolate = a(menu, "Interpolate From Previous Keyframe", self.interpolate)
        clear = a(menu, "Clear Labels From Here...", self.clear_frames)
        for action, tip in (
            (self.propagate_action, "Send this frame's objects (and what you erased) forward"),
            (backward, "Send this frame's objects (and what you erased) to earlier frames"),
            (self.ai_action, "On: EfficientTAM follows each object. Off: masks are copied as is"),
            (self.track_action, "Each Next carries the objects into the next frame by itself"),
            (frames, "How many frames one Propagate goes through"),
            (interpolate, "Fill the frames between the last hand-made frame and this one"),
            (clear, "Delete every label on this frame and the next ones"),
        ):
            action.setStatusTip(tip)

    # ------------------------------------------------------------------ options

    @property
    def use_ai(self) -> bool:
        """True to track with the video model, False to copy masks unchanged."""
        return self.ai_action.isChecked()

    @property
    def tracking(self) -> bool:
        """True if Next Image carries the objects (and a cut) into the next frame."""
        return self.track_action.isChecked()

    def _toggle_ai(self, checked: bool) -> None:
        self.window.statusBar().showMessage(
            "Propagate tracks objects with AI" if checked else "Propagate copies masks unchanged",
            5000,
        )

    def _toggle_tracking(self, checked: bool) -> None:
        self.window.statusBar().showMessage(
            "Auto-propagate on: Next carries the objects and what you erased into each frame"
            if checked
            else "Auto-propagate off",
            5000,
        )

    def ask_frames_per_run(self) -> None:
        """Ask how many frames one Propagate goes through."""
        w = self.window
        count = w.ask_number(
            w, "Propagate", "Frames to go through:", self.frames_per_run, 1, 10_000
        )
        if count is not None:
            self.frames_per_run = int(count)

    def open_video_dialog(self) -> None:
        """Ask for a video file and open it (importing its frames first if needed)."""
        name, _ = QFileDialog.getOpenFileName(self.window, "Open Video", "", VIDEO_FILTER)
        if name:
            self.window.open_path(Path(name))

    @property
    def running(self) -> bool:
        """True while a propagation run is in progress."""
        return self.worker is not None

    # ------------------------------------------------------------------ propagate

    def _carry(self, shapes: list[Shape]) -> Carry | None:
        """This frame's selected (or all) objects and its cut; None if there is neither."""
        w = self.window
        assert w.current_index is not None
        chosen = [shapes[i] for i in w.canvas.selection if i < len(shapes)] or shapes
        objects = {object_key(s) for s in chosen if s.shape_type.is_area}
        cut = self.last_cut if self.last_cut and self.last_cut.frame == w.current_index else None
        if not objects and cut is None:
            return None
        return Carry(w.current_index, shapes, objects, cut)

    def propagate(self, direction: int = +1) -> None:
        """Send the selected (or all) objects and the last cut forward (+1) or back (-1)."""
        w = self.window
        current = w.current_index
        if current is None or self.running:
            return
        shapes = confirm(w.canvas.shapes)  # propagating from a frame vouches for it
        carry = self._carry(shapes)
        if carry is None:
            w.statusBar().showMessage(NOTHING_HINT, 6000)
            return
        remaining = len(w.image_paths) - 1 - current if direction > 0 else current
        if remaining == 0:
            w.statusBar().showMessage("No more frames in that direction", 4000)
            return
        propagator = self._propagator_for(self.use_ai, ask=True)
        if propagator is None:
            return
        if shapes != w.canvas.shapes:
            w._replace_shapes(shapes)
        self._run(carry, direction, min(self.frames_per_run, remaining), propagator)

    def capture_for_tracking(self) -> Carry | None:
        """Before leaving a frame: what Track carries into the next one (None if nothing)."""
        w = self.window
        if not self.tracking or w.current_index is None:
            return None
        return self._carry(list(w.canvas.shapes))

    def track(self, carry: Carry) -> None:
        """After arriving on the next frame: propagate the captured objects and cut into it."""
        if self.running:
            self.window.statusBar().showMessage("Still tracking the previous frame...", 3000)
            return
        propagator = self._propagator_for(self.use_ai, ask=False)
        assert propagator is not None  # without asking, there is always a fallback
        self._run(carry, +1, 1, propagator)

    def _propagator_for(self, ai: bool, ask: bool) -> Propagator | None:
        """The copy propagator, or the AI model (loaded once).

        If the video extra is missing, ``ask`` offers copying instead (None if declined);
        without ``ask`` (automatic tracking) it copies and says so in the status bar.
        """
        if not ai or self.window.settings.video_model == COPY_MODEL:
            return self.copier
        if self.propagator is None:
            try:
                self.propagator = self.make_propagator()
            except ImportError:
                w = self.window
                if not ask:
                    w.statusBar().showMessage(FALLBACK_NOTE, 6000)
                    return self.copier
                answer = QMessageBox.question(
                    w, "Propagate", f"{MISSING_EXTRA}\n\nCopy the masks unchanged instead?"
                )
                if answer != QMessageBox.StandardButton.Yes:
                    return None
                self.ai_action.setChecked(False)
                return self.copier
        return self.propagator

    def _run(self, carry: Carry, direction: int, count: int, propagator: Propagator) -> None:
        """Propagate ``carry`` over ``count`` frames in ``direction`` in the background."""
        w = self.window
        height, width = w.image_size[1], w.image_size[0]
        indices = [carry.frame + direction * step for step in range(count + 1)]
        frame_shapes = [self._frame_shapes(i) for i in indices]
        frame_shapes[0] = [s for s in carry.shapes if object_key(s) in carry.objects]
        cut = carry.cut
        # Objects with key points get fresh ones on every frame the run reaches.
        pointed = {
            object_key(s) for s in carry.shapes if is_keypoint(s) and object_key(s) in carry.objects
        }
        if cut is not None:
            region = masks.to_shape(REGION_LABEL, cut.part)
            if region is not None:
                frame_shapes[0].append(region)
            # Later masks the cut is fixing still contain the part: as prompts they would
            # pull it back into their object, so they do not steer this run.
            for later in frame_shapes[1:]:
                later[:] = [
                    s
                    for s in later
                    if not s.shape_type.is_area or (cut.keys and object_key(s) not in cut.keys)
                ]

        def on_frame(index: int, new: list[Shape]) -> None:
            moved = np.zeros((height, width), dtype=bool)
            for shape in new:
                if shape.label == REGION_LABEL:
                    moved |= shape_to_mask(shape, height, width)
            objects = [s for s in new if s.label != REGION_LABEL]

            def transform(shapes: list[Shape]) -> list[Shape]:
                shapes = merge_auto(shapes, carry.objects, objects)
                if cut is not None and moved.any():
                    shapes = cut_region(shapes, moved, height, width, cut.keys)
                if pointed:
                    shapes = with_keypoints(shapes, height, width, pointed)
                return shapes

            self._update_frame(index, transform)

        self._begin_run()
        self._start(indices, frame_shapes, propagator, on_frame)

    def _start(
        self,
        indices: list[int],
        frame_shapes: list[list[Shape]],
        propagator: Propagator,
        on_frame: Callable[[int, list[Shape]], None],
    ) -> None:
        """Run ``propagator`` over frames ``indices`` in the background with progress.

        ``on_frame(frame index, propagated shapes)`` is called for each frame after the
        first, on the GUI thread.
        """
        w = self.window
        frames = [w.image_paths[i] for i in indices]
        size = (w.image_size[1], w.image_size[0])
        worker = PropagationWorker(propagator, frames, frame_shapes, size)
        worker.frame_done.connect(
            lambda position, new: self._on_frame_done(indices, position, new, on_frame)
        )
        worker.failed.connect(w._error)
        worker.finished.connect(self._on_finished)
        self.worker = worker
        self._run_frames = len(indices) - 1
        task = "Copying masks" if isinstance(propagator, CopyPropagator) else "AI tracking"
        w.statusBar().showMessage(f"{task}...")
        w.progress.begin("propagate", task, total=len(indices) - 1, cancel=self.stop)
        worker.start()

    def _on_frame_done(
        self,
        indices: list[int],
        position: int,
        new: list[Shape],
        on_frame: Callable[[int, list[Shape]], None],
    ) -> None:
        on_frame(indices[position], new)
        self.window.progress.report("propagate", position)

    def stop(self, wait: bool = False) -> None:
        """Cancel the running propagation after the current frame.

        Args:
            wait: Block until the worker thread has exited (used when closing the window).
        """
        if self.worker is not None:
            self.worker.cancel()
            if wait:
                self.worker.wait()

    def _on_finished(self) -> None:
        if self.worker is not None:
            self.worker.deleteLater()
        self.worker = None
        self._end_run()
        self.window.progress.end("propagate")
        changed, total = len(self.changed), self._run_frames
        if changed == 0:
            text = (
                "Nothing changed: those frames already have hand-drawn masks of these objects "
                "(delete them there, or Clear Labels From Here, then Propagate again)"
            )
        elif changed < total:
            text = f"Updated {changed} of {total} frames (frames with hand-drawn masks keep them)"
        else:
            text = f"Updated {changed} frames  ·  Undo reverts all of them"
        self.window.statusBar().showMessage(text, 8000)

    # ------------------------------------------------------------------ frame files

    def _frame_shapes(self, index: int) -> list[Shape]:
        """Current shapes of frame ``index``: from the canvas if shown, else from disk."""
        w = self.window
        if index == w.current_index:
            return list(w.canvas.shapes)
        json_path = ann_io.annotation_path_for(w.image_paths[index], w.settings.output_dir)
        return ann_io.load(json_path).shapes if json_path.exists() else []

    def _store(self, index: int, objects: set[ObjectKey], new: list[Shape]) -> None:
        """Merge automatic shapes into frame ``index`` (on screen, or straight to disk)."""
        self._update_frame(index, lambda shapes: merge_auto(shapes, objects, new))

    def _update_frame(self, index: int, transform: Callable[[list[Shape]], list[Shape]]) -> None:
        """Apply ``transform`` to frame ``index``'s shapes: on screen or on disk.

        Frames that would not change are left alone; the others are recorded in the undo
        history (one step per run, see :meth:`_begin_run`).
        """
        w = self.window
        old = self._frame_shapes(index)
        new = transform(list(old))
        if new == old:
            return
        self.changed.add(index)
        if index == w.current_index:
            w._replace_shapes(new)
            return
        before = w.frame_state(index)
        w.write_frame(index, new)
        w.history.record(index, before, new)

    # ------------------------------------------------------------------ erase with a shape

    def erase_with(self, index: int | None) -> None:
        """Use shape ``index``'s object as an eraser, here and on its propagated frames.

        On this frame and on every neighbouring frame (both directions, without a gap) that
        has an automatic copy of the object, the object is cut out of the other masks under
        it and then removed. One Undo reverts all of it.
        """
        w = self.window
        current = w.current_index
        if current is None or index is None or self.running:
            return
        key = object_key(w.canvas.shapes[index])
        height, width = w.image_size[1], w.image_size[0]

        def has_copy(i: int) -> bool:
            return any(is_auto(s) and object_key(s) == key for s in self._frame_shapes(i))

        frames = [current]
        for step in (+1, -1):
            i = current + step
            while 0 <= i < len(w.image_paths) and has_copy(i):
                frames.append(i)
                i += step

        def erase(shapes: list[Shape]) -> list[Shape]:
            eraser = np.zeros((height, width), dtype=bool)
            for s in shapes:
                if object_key(s) == key and s.shape_type.is_area:
                    eraser |= shape_to_mask(s, height, width)
            rest = [s for s in shapes if object_key(s) != key]
            return cut_region(rest, eraser, height, width)

        self._begin_run()
        for i in sorted(frames):
            self._update_frame(i, erase)
        self._end_run()
        w.statusBar().showMessage(f"Erased on {len(self.changed)} frame(s)  ·  Undo reverts", 6000)

    # ------------------------------------------------------------------ interpolate, clear

    def interpolate(self) -> None:
        """Fill the frames between the previous keyframe and this one.

        The previous keyframe is the nearest earlier frame with hand-drawn area shapes.
        """
        w = self.window
        current = w.current_index
        if current is None or self.running:
            return
        end = [s for s in w.canvas.shapes if not is_auto(s)]
        previous = next(
            (
                i
                for i in range(current - 1, -1, -1)
                if any(s.shape_type.is_area and not is_auto(s) for s in self._frame_shapes(i))
            ),
            None,
        )
        if previous is None or current - previous < 2:
            w.statusBar().showMessage("No earlier keyframe with a gap to fill", 4000)
            return
        start = [s for s in self._frame_shapes(previous) if not is_auto(s)]
        height, width = w.image_size[1], w.image_size[0]
        frames = interpolate_frames(start, end, current - previous, height, width)
        self._begin_run()
        for offset, shapes in enumerate(frames, start=1):
            self._store(previous + offset, {object_key(s) for s in shapes}, shapes)
        self._end_run()
        w.statusBar().showMessage(
            f"Interpolated {len(frames)} frames from {w.image_paths[previous].name}", 5000
        )

    def clear_frames(self) -> None:
        """Delete every shape on this frame and the next ones (asks how many)."""
        w = self.window
        current = w.current_index
        if current is None or self.running:
            return
        remaining = len(w.image_paths) - current
        count = w.ask_number(
            w, "Clear Labels", "Frames to clear, starting with this one:", 1, 1, remaining
        )
        if count is None:
            return
        self._begin_run()
        for index in range(current, current + int(count)):
            self._update_frame(index, lambda shapes: [])
        self._end_run()
        w.statusBar().showMessage(f"Cleared {len(self.changed)} frame(s)  ·  Undo puts them back")

    def remove_label(self, label: str) -> None:
        """Delete every shape with ``label`` on every frame of the folder (one undo step)."""
        w = self.window
        if w.current_index is None or self.running:
            return
        self._begin_run()
        for index in range(len(w.image_paths)):
            self._update_frame(index, lambda shapes: [s for s in shapes if s.label != label])
        self._end_run()
        w.forget_label(label)
        w.statusBar().showMessage(
            f"Deleted “{label}” from {len(self.changed)} frame(s)  ·  Undo brings it back", 8000
        )

    def apply_to_all_frames(
        self, transform: Callable[[list[Shape]], list[Shape]], verb: str
    ) -> None:
        """Run ``transform`` on every frame's shapes as one undo step; report the count."""
        w = self.window
        if w.current_index is None or self.running:
            return
        self._begin_run()
        for index in range(len(w.image_paths)):
            self._update_frame(index, transform)
        self._end_run()
        w.statusBar().showMessage(f"{verb} {len(self.changed)} frame(s)  ·  Undo reverts", 6000)

    def rename_label(self, old: str, new: str) -> None:
        """Rename ``old`` to ``new`` on every frame of the folder (one undo step)."""
        w = self.window
        if w.current_index is None or self.running:
            return

        def rename(shapes: list[Shape]) -> list[Shape]:
            result = []
            for shape in shapes:
                if shape.label == old:
                    shape = shape.copy()
                    shape.label = new
                result.append(shape)
            return result

        self._begin_run()
        for index in range(len(w.image_paths)):
            self._update_frame(index, rename)
        self._end_run()
        was_active = w.active_label == old
        w._remember_label(new)
        w.forget_label(old)
        if was_active:
            w.set_active_label(new)
        w.statusBar().showMessage(
            f"Renamed “{old}” to “{new}” on {len(self.changed)} frame(s)", 6000
        )

    # ------------------------------------------------------------------ one undo step per run

    def _begin_run(self) -> None:
        """Changes from now until :meth:`_end_run` become one undo step."""
        self.changed = set()
        self.window.history.begin_group()

    def _end_run(self) -> None:
        self.window.history.end_group()
        self.window._update_undo_actions()
