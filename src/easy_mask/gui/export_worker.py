"""Run a dataset export on a background thread, reporting progress per image."""

from pathlib import Path

from PySide6.QtCore import QThread, Signal

from easy_mask.io import export
from easy_mask.io.export import PARTS, AnnotatedImage


class ExportCancelledError(Exception):
    """Raised inside the exporter's progress callback to stop it."""


class ExportWorker(QThread):
    """Writes ``items`` in format ``fmt`` to ``out`` without freezing the window.

    Signals:
        progress(int, int): Images done, images total.
        failed(str): The export raised; the message is meant for the user.
        exported(int): Finished; number of images written.
    """

    progress = Signal(int, int)
    failed = Signal(str)
    exported = Signal(int)

    def __init__(
        self,
        items: list[AnnotatedImage],
        fmt: str,
        out: Path,
        labels: list[str],
        parts: tuple[str, ...] = PARTS,
    ):
        super().__init__()
        self.parts = parts  # what the Objects export writes; ignored by other formats
        self.items = items
        self.fmt = fmt
        self.out = out
        self.labels = labels
        self._cancelled = False

    def cancel(self) -> None:
        """Stop after the current image (files written so far are left in place)."""
        self._cancelled = True

    def run(self) -> None:
        """Runs on the worker thread."""
        try:
            if self.fmt == "objects":
                export.export_objects(
                    self.items, self.out, self.labels, progress=self._report, parts=self.parts
                )
            else:
                export.EXPORTERS[self.fmt](self.items, self.out, self.labels, progress=self._report)
        except ExportCancelledError:
            return
        except Exception as exc:  # shown to the user, not swallowed
            self.failed.emit(f"Export failed: {exc}")
            return
        self.exported.emit(len(self.items))

    def _report(self, done: int, total: int) -> None:
        if self._cancelled:
            raise ExportCancelledError
        self.progress.emit(done, total)
