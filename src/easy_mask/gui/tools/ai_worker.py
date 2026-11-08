"""Background work for AI Segment: load the model and encode images off the UI thread.

Loading (maybe downloading) the model takes seconds and encoding an image hundreds of
milliseconds, so both run in a :class:`QThreadPool` with one thread. Jobs therefore run
in the order they were started, and results come back to the UI thread as Qt signals.
"""

import threading
from collections.abc import Callable

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Signal

from easy_mask.ai.segmenter import Segmenter
from easy_mask.ai.weights import ProgressFn

SegmenterLoader = Callable[[ProgressFn], Segmenter]


class LazySegmenter:
    """Loads the segmenter on first use, once, even if several jobs ask at the same time."""

    def __init__(self, load: SegmenterLoader):
        self._load = load
        self._lock = threading.Lock()
        self.segmenter: Segmenter | None = None

    def get(self, progress: ProgressFn) -> Segmenter:
        """The loaded segmenter (loading it now if needed)."""
        with self._lock:
            if self.segmenter is None:
                self.segmenter = self._load(progress)
            return self.segmenter


class EncodeSignals(QObject):
    """Results of an :class:`EncodeJob`, delivered on the UI thread.

    Signals:
        progress(str): Human-readable status (download progress, "encoding").
        done(int): The job with this generation number finished; the image is encoded.
        failed(str): Loading or encoding failed; the message is meant for the user.
    """

    progress = Signal(str)
    done = Signal(int)
    failed = Signal(str)


class EncodeJob(QRunnable):
    """Load the segmenter if needed, then compute (or fetch cached) image embedding."""

    def __init__(self, lazy: LazySegmenter, rgb: np.ndarray, key: str, generation: int):
        super().__init__()
        self.lazy = lazy
        self.rgb = rgb
        self.key = key
        self.generation = generation
        self.signals = EncodeSignals()

    def run(self) -> None:
        """Runs on the worker thread."""
        try:
            segmenter = self.lazy.get(self._report_download)
            if not segmenter.is_cached(self.key):
                self.signals.progress.emit("AI Segment: encoding image...")
            segmenter.set_image(self.rgb, self.key)
        except Exception as exc:  # reported to the user, not swallowed
            self.signals.failed.emit(str(exc))
            return
        self.signals.done.emit(self.generation)

    def _report_download(self, name: str, done: int, total: int) -> None:
        self.signals.progress.emit(f"AI Segment: downloading {name} ({100 * done // total}%)")
