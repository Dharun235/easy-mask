"""Progress feedback for slow work: a status-bar panel with text, a bar and Cancel.

Every long task (model download, preparing an image for AI, propagation, export, video
import) reports here, so the user always sees that something is happening, how far it is,
and how to stop it. Several tasks can run at once; the most recently started is shown.
"""

from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QToolButton, QWidget


@dataclass
class _Task:
    text: str
    done: int = 0
    total: int = 0  # 0 means "unknown": a busy bar instead of a percentage
    cancel: Callable[[], None] | None = None


class ProgressPanel(QWidget):
    """Shows the newest running task; hidden when nothing is running."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.label = QLabel()
        self.bar = QProgressBar()
        self.bar.setFixedWidth(160)
        self.bar.setTextVisible(False)
        self.cancel_button = QToolButton()
        self.cancel_button.setText("Cancel")
        self.cancel_button.clicked.connect(self._cancel_current)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.addWidget(self.label)
        layout.addWidget(self.bar)
        layout.addWidget(self.cancel_button)
        self._tasks: dict[str, _Task] = {}
        self.setVisible(False)

    # ------------------------------------------------------------------ public API

    def begin(
        self, key: str, text: str, total: int = 0, cancel: Callable[[], None] | None = None
    ) -> None:
        """Start (or restart) task ``key``. ``total`` 0 shows a busy bar."""
        self._tasks.pop(key, None)  # re-insert so it becomes the newest
        self._tasks[key] = _Task(text, 0, total, cancel)
        self._show()

    def report(
        self, key: str, done: int, total: int | None = None, text: str | None = None
    ) -> None:
        """Report progress on task ``key`` (ignored if it is not running)."""
        task = self._tasks.get(key)
        if task is None:
            return
        task.done = done
        if total is not None:
            task.total = total
        if text is not None:
            task.text = text
        self._show()

    def end(self, key: str) -> None:
        """Task ``key`` finished, failed or was cancelled."""
        self._tasks.pop(key, None)
        self._show()

    def is_running(self, key: str) -> bool:
        """True while task ``key`` is shown as running."""
        return key in self._tasks

    # ------------------------------------------------------------------ display

    def _current(self) -> tuple[str, _Task] | None:
        return next(reversed(self._tasks.items()), None)

    def _show(self) -> None:
        current = self._current()
        self.setVisible(current is not None)
        if current is None:
            return
        _key, task = current
        if task.total > 0:
            self.bar.setRange(0, task.total)
            self.bar.setValue(min(task.done, task.total))
            percent = 100 * min(task.done, task.total) // task.total
            self.label.setText(f"{task.text}  {percent}%")
        else:
            self.bar.setRange(0, 0)  # busy indicator
            self.label.setText(f"{task.text}...")
        others = len(self._tasks) - 1
        if others:
            self.label.setText(f"{self.label.text()}  (+{others} more)")
        self.cancel_button.setVisible(task.cancel is not None)

    def _cancel_current(self) -> None:
        current = self._current()
        if current is not None and current[1].cancel is not None:
            key, task = current
            assert task.cancel is not None
            task.cancel()
            self.end(key)
