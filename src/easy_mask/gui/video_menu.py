"""The Video menu: open a video, interpolate between keyframes, propagate masks.

A video is just the folder of frames the main window already shows, so everything here
works frame by frame on the same per-image JSON files. Shapes made automatically are
flagged (see :mod:`easy_mask.core.video`) and replaced on the next run; hand-drawn ones
are kept and steer the model.
"""

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtWidgets import QFileDialog, QMenu, QMessageBox

from easy_mask.core import annotation as ann_io
from easy_mask.core.annotation import Annotation
from easy_mask.core.interpolate import interpolate_frames
from easy_mask.core.shapes import Shape
from easy_mask.core.video import ObjectKey, confirm, is_auto, merge_auto, object_key
from easy_mask.gui.propagate import PropagationWorker
from easy_mask.gui.video_import import VIDEO_FILTER
from easy_mask.io.images import image_size
from easy_mask.propagation.tracker import CopyPropagator, Propagator

if TYPE_CHECKING:  # the main window imports this module
    from easy_mask.gui.main_window import MainWindow

DEFAULT_RUN = 50  # frames offered per propagation run: a few minutes on a laptop CPU
COPY_MODEL = "copy"  # --video-model value for model-free propagation
MISSING_EXTRA = (
    "AI Propagate uses EfficientTAM (or SAM 2.1), which needs the optional video add-on "
    "(PyTorch, about 1 GB):\n\n    SAM2_BUILD_CUDA=0 uv sync --extra video"
)
FALLBACK_NOTE = "AI tracking needs the video add-on (uv sync --extra video): copying instead"


def default_propagator(model: str) -> Propagator:
    """The chosen backend: the real model (needs PyTorch) or plain copying."""
    if model == COPY_MODEL:
        return CopyPropagator()
    from easy_mask.propagation.backend import Sam2Propagator

    return Sam2Propagator(model)


class VideoMenu:
    """Adds the Video menu to ``window`` and runs its commands."""

    def __init__(self, window: "MainWindow"):
        self.window = window
        # Swappable so tests can use a fake model.
        self.make_propagator: Callable[[], Propagator] = lambda: default_propagator(
            window.settings.video_model
        )
        self.propagator: Propagator | None = None  # the AI model: built once, then reused
        self.copier = CopyPropagator()  # "Copy to frames" and the fallback without the extra
        self.worker: PropagationWorker | None = None

        bar = window.menuBar()
        menu = QMenu("&Video", bar)
        bar.insertMenu(bar.actions()[-1], menu)  # before Help
        window._action(menu, "Open Video...", self.open_video_dialog)
        menu.addSeparator()
        window._action(menu, "Interpolate From Previous Keyframe", self.interpolate, "Ctrl+Shift+I")
        menu.addSeparator()
        self.propagate_action = window._action(
            menu, "AI Propagate Forward...", lambda: self.propagate(+1), "Ctrl+Shift+Right"
        )
        window._action(
            menu, "AI Propagate Backward...", lambda: self.propagate(-1), "Ctrl+Shift+Left"
        )
        self.copy_action = window._action(
            menu, "Copy to Next Frames...", lambda: self.propagate(+1, ai=False), "Ctrl+Alt+Right"
        )
        window._action(
            menu,
            "Copy to Previous Frames...",
            lambda: self.propagate(-1, ai=False),
            "Ctrl+Alt+Left",
        )
        self.track_action = window._action(
            menu, "Track Objects into Next Frame", self._toggle_tracking, "Ctrl+Shift+T", True
        )
        self.stop_action = window._action(menu, "Stop Propagation", self.stop, "Ctrl+.")
        self.stop_action.setEnabled(False)

    # ------------------------------------------------------------------ tracking

    @property
    def tracking(self) -> bool:
        """True if moving to the next frame carries the objects along."""
        return self.track_action.isChecked()

    def _toggle_tracking(self, checked: bool) -> None:
        self.window.statusBar().showMessage(
            "Tracking on: Next Image (D) carries the selected (or all) objects forward"
            if checked
            else "Tracking off",
            5000,
        )

    def capture_for_tracking(self) -> tuple[int, list[Shape], set[ObjectKey]] | None:
        """Before leaving a frame: what to carry into the next one (None if nothing)."""
        w = self.window
        if not self.tracking or w.current_index is None:
            return None
        shapes = list(w.canvas.shapes)
        chosen = [shapes[i] for i in w.canvas.selection] or shapes
        objects = {object_key(s) for s in chosen if s.shape_type.is_area}
        return (w.current_index, shapes, objects) if objects else None

    def track(self, carry: tuple[int, list[Shape], set[ObjectKey]]) -> None:
        """After arriving on the next frame: propagate the captured objects into it."""
        start, shapes, objects = carry
        if self.running:
            self.window.statusBar().showMessage("Still tracking the previous frame...", 3000)
            return
        propagator = self._propagator_for(ai=True, ask=False)
        assert propagator is not None  # without asking, there is always a fallback
        self._run(start, +1, 1, objects, shapes, propagator)

    @property
    def running(self) -> bool:
        """True while a propagation run is in progress."""
        return self.worker is not None

    def open_video_dialog(self) -> None:
        """Ask for a video file and open it (importing its frames first if needed)."""
        name, _ = QFileDialog.getOpenFileName(self.window, "Open Video", "", VIDEO_FILTER)
        if name:
            self.window.open_path(Path(name))

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
        w = self.window
        if index == w.current_index:
            w._replace_shapes(merge_auto(list(w.canvas.shapes), objects, new))
            return
        path = w.image_paths[index]
        json_path = ann_io.annotation_path_for(path, w.settings.output_dir)
        if json_path.exists():
            annotation = ann_io.load(json_path)
        else:
            width, height = image_size(path)
            relative = ann_io.relative_image_path(path, json_path)
            annotation = Annotation(relative, width, height)
        annotation.shapes = merge_auto(annotation.shapes, objects, new)
        ann_io.save(annotation, json_path)
        w.file_list.mark_annotated(index)

    # ------------------------------------------------------------------ interpolation

    def interpolate(self) -> None:
        """Fill the frames between the previous keyframe and this one.

        The previous keyframe is the nearest earlier frame with hand-drawn area shapes.
        """
        w = self.window
        current = w.current_index
        if current is None:
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
        for offset, shapes in enumerate(frames, start=1):
            self._store(previous + offset, {object_key(s) for s in shapes}, shapes)
        w.statusBar().showMessage(
            f"Interpolated {len(frames)} frames from {w.image_paths[previous].name}", 5000
        )

    # ------------------------------------------------------------------ propagation

    def propagate(self, direction: int, ai: bool = True) -> None:
        """Send objects on this frame forward (+1) or backward (-1) through the video.

        With ``ai`` the video model (EfficientTAM by default) tracks and re-segments each
        object from its mask on this frame; otherwise the mask is copied unchanged. With
        shapes selected, only their objects go; otherwise every object on the frame does.
        """
        w = self.window
        current = w.current_index
        if current is None or self.running:
            return
        shapes = confirm(w.canvas.shapes)  # propagating from a frame vouches for it
        chosen = [shapes[i] for i in w.canvas.selection] or shapes
        objects = {object_key(s) for s in chosen if s.shape_type.is_area}
        if not objects:
            w.statusBar().showMessage("Draw or segment an object on this frame first", 4000)
            return
        remaining = len(w.image_paths) - 1 - current if direction > 0 else current
        if remaining == 0:
            w.statusBar().showMessage("No frames in that direction", 4000)
            return
        title = "AI Propagate" if ai else "Copy to Frames"
        count = w.ask_number(
            w, title, "Number of frames:", min(DEFAULT_RUN, remaining), 1, remaining
        )
        if count is None:
            return
        propagator = self._propagator_for(ai, ask=True)
        if propagator is None:
            return
        if shapes != w.canvas.shapes:
            w._replace_shapes(shapes)
        self._run(current, direction, int(count), objects, shapes, propagator)

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
                    w,
                    "AI Propagate",
                    f"{MISSING_EXTRA}\n\nCopy the mask to the next frames instead?",
                )
                return self.copier if answer == QMessageBox.StandardButton.Yes else None
        return self.propagator

    def _run(
        self,
        current: int,
        direction: int,
        count: int,
        objects: set[ObjectKey],
        shapes: list[Shape],
        propagator: Propagator,
    ) -> None:
        """Propagate ``objects`` from frame ``current`` over ``count`` frames in the background."""
        w = self.window
        indices = [current + direction * step for step in range(count + 1)]
        frames = [w.image_paths[i] for i in indices]
        frame_shapes = [self._frame_shapes(i) for i in indices]
        frame_shapes[0] = [s for s in shapes if object_key(s) in objects]  # only chosen objects
        size = (w.image_size[1], w.image_size[0])
        worker = PropagationWorker(propagator, frames, frame_shapes, size)
        worker.frame_done.connect(
            lambda position, new: self._on_frame_done(indices, position, objects, new)
        )
        worker.failed.connect(w._error)
        worker.finished.connect(self._on_finished)
        self.worker = worker
        self.stop_action.setEnabled(True)
        copying = isinstance(propagator, CopyPropagator)
        task = "Copying to frames" if copying else "AI propagating"
        w.statusBar().showMessage(
            f"{task}..." if copying else f"{task}... (the first run loads the model)"
        )
        w.progress.begin("propagate", task, total=count, cancel=self.stop)
        worker.start()

    def _on_frame_done(
        self,
        indices: list[int],
        position: int,
        objects: set[ObjectKey],
        new: list[Shape],
    ) -> None:
        self._store(indices[position], objects, new)
        self.window.progress.report("propagate", position)
        self.window.statusBar().showMessage(
            f"Propagated {position}/{len(indices) - 1} frames (Video > Stop to cancel)"
        )

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
        self.stop_action.setEnabled(False)
        self.window.progress.end("propagate")
        self.window.statusBar().showMessage("Propagation finished", 5000)
