"""Fixtures for GUI tests: windows, a mouse/keyboard driver, and dialog stubs.

GUI tests use real Qt input events (pytest-qt's ``qtbot``) on a headless window, so they
exercise the same code paths as a user clicking. Modal dialogs would block forever
headless, so ``dialogs`` replaces them with recorded, scriptable stand-ins.
"""

from dataclasses import dataclass, field

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QMessageBox

from easy_mask.gui.label_dialog import LabelInfo
from easy_mask.gui.main_window import MainWindow, Settings


@dataclass
class Dialogs:
    """Records message boxes. Set ``answer`` to script the unsaved-changes question."""

    answer: QMessageBox.StandardButton | None = None
    questions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    allow_warnings: bool = False


@pytest.fixture(autouse=True)
def dialogs(monkeypatch):
    """Replace QMessageBox popups; fail on any warning a test did not opt into."""
    record = Dialogs()

    def question(parent, title, text, *args, **kwargs):
        record.questions.append(text)
        if record.answer is None:
            raise AssertionError(f"unexpected question dialog: {text!r}")
        return record.answer

    def warning(parent, title, text, *args, **kwargs):
        record.warnings.append(text)

    monkeypatch.setattr(QMessageBox, "question", question)
    monkeypatch.setattr(QMessageBox, "warning", warning)
    monkeypatch.setattr(QMessageBox, "about", lambda *a, **k: None)
    yield record
    if not record.allow_warnings:
        assert record.warnings == [], f"unexpected warning dialogs: {record.warnings}"


class Driver:
    """Click, drag and type on a window's canvas in *image* pixel coordinates."""

    def __init__(self, qtbot, window: MainWindow):
        self.qtbot = qtbot
        self.window = window

    def screen(self, x: float, y: float) -> QPoint:
        """Widget position of image pixel ``(x, y)``."""
        c = self.window.canvas
        return QPoint(round(c.offset.x() + x * c.scale), round(c.offset.y() + y * c.scale))

    def click(self, x, y, button=Qt.MouseButton.LeftButton):
        self.qtbot.mouseClick(self.window.canvas, button, pos=self.screen(x, y))

    def double_click(self, x, y):
        self.qtbot.mouseDClick(self.window.canvas, Qt.MouseButton.LeftButton, pos=self.screen(x, y))

    def drag(self, start, end, button=Qt.MouseButton.LeftButton):
        c = self.window.canvas
        self.qtbot.mousePress(c, button, pos=self.screen(*start))
        self.qtbot.mouseMove(c, self.screen((start[0] + end[0]) / 2, (start[1] + end[1]) / 2))
        self.qtbot.mouseMove(c, self.screen(*end))
        self.qtbot.mouseRelease(c, button, pos=self.screen(*end))

    def key(self, key, modifier=Qt.KeyboardModifier.NoModifier):
        self.qtbot.keyClick(self.window.canvas, key, modifier)


@pytest.fixture
def make_window(qtbot):
    """Factory for shown windows; closes them without the unsaved-changes prompt."""
    windows = []

    def make(settings: Settings | None = None, label: LabelInfo | None = None) -> MainWindow:
        win = MainWindow(settings or Settings())
        win.ask_label = lambda known, initial, parent: label or LabelInfo("cat")
        win.show()
        qtbot.waitExposed(win)
        windows.append(win)
        return win

    yield make
    for win in windows:
        win.dirty = False
        win.close()


@pytest.fixture
def window(make_window, image_dir):
    """A window with labels cat/dog and flag 'blurry', opened on ``image_dir``."""
    win = make_window(Settings(labels=["cat", "dog"], flags=["blurry"]), LabelInfo("cat"))
    win.open_path(image_dir)
    return win


@pytest.fixture
def ui(qtbot, window) -> Driver:
    """Input driver bound to ``window``."""
    return Driver(qtbot, window)


@pytest.fixture
def driver(qtbot):
    """Build a Driver for any window: ``driver(win)``."""
    return lambda win: Driver(qtbot, win)
