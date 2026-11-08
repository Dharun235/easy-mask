"""Run mask propagation on a background thread so the window stays responsive.

The worker only runs the model; results come back to the GUI thread one frame at a time
through ``frame_done``, and the window decides how to store them.
"""

from pathlib import Path

from PySide6.QtCore import QThread, Signal

from easy_mask.core.shapes import Shape
from easy_mask.propagation.tracker import Propagator, track


class PropagationWorker(QThread):
    """Propagate the first frame's objects through ``frames`` (in that order).

    Signals:
        frame_done(int, list): Position in ``frames`` and its propagated shapes.
        failed(str): The model raised; the message is for the user.
    """

    frame_done = Signal(int, list)
    failed = Signal(str)

    def __init__(
        self,
        propagator: Propagator,
        frames: list[Path],
        frame_shapes: list[list[Shape]],
        size: tuple[int, int],
    ):
        super().__init__()
        self.propagator = propagator
        self.frames = frames
        self.frame_shapes = frame_shapes
        self.height, self.width = size
        self._cancelled = False

    def cancel(self) -> None:
        """Stop after the frame being computed; results so far are kept."""
        self._cancelled = True

    def run(self) -> None:
        """Thread body: iterate the tracker until done or cancelled."""
        results = track(self.propagator, self.frames, self.frame_shapes, self.height, self.width)
        try:
            for position, shapes in results:
                if self._cancelled:
                    break
                self.frame_done.emit(position, shapes)
        except Exception as exc:  # report any model failure to the user, don't crash
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            results.close()  # lets the backend free its temporary files
