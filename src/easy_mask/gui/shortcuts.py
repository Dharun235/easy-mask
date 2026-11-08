"""Help > Keyboard Shortcuts: every menu command and its keys, built from the menus."""

from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QDialog,
    QHeaderView,
    QMainWindow,
    QMenu,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

EXTRA = [  # canvas keys that are not menu commands
    ("Canvas", "Ctrl + wheel", "zoom at the cursor"),
    ("Canvas", "Wheel / middle-drag", "pan"),
    ("Canvas", "Shift / Ctrl + click", "add to the selection"),
    ("Canvas", "Enter / Backspace / Esc", "finish / undo last point / cancel"),
    ("Canvas", "[  ]", "smaller / bigger brush, tolerance or region"),
]


def shortcut_rows(window: QMainWindow) -> list[tuple[str, str, str]]:
    """``(menu, keys, command)`` for every menu command that has a shortcut."""
    rows = []
    for menu_action in window.menuBar().actions():
        menu = menu_action.menu()
        if isinstance(menu, QMenu):
            rows += _rows(menu, menu_action.text().replace("&", ""))
    return rows + EXTRA


def _rows(menu: QMenu, title: str) -> list[tuple[str, str, str]]:
    rows = []
    for action in menu.actions():
        if action.menu() is not None:
            rows += _rows(action.menu(), title)  # type: ignore[arg-type]
        elif isinstance(action, QAction) and action.shortcuts():
            keys = " / ".join(k.toString(k.SequenceFormat.NativeText) for k in action.shortcuts())
            rows.append((title, keys, action.text().replace("&", "").rstrip(".")))
    return rows


class ShortcutsDialog(QDialog):
    """A searchable-by-eye table of all shortcuts (non-modal, stays open while working)."""

    def __init__(self, window: QMainWindow):
        super().__init__(window)
        self.setWindowTitle("Keyboard Shortcuts")
        rows = shortcut_rows(window)
        table = QTableWidget(len(rows), 3)
        table.setHorizontalHeaderLabels(["Menu", "Keys", "Command"])
        for r, row in enumerate(rows):
            for c, text in enumerate(row):
                table.setItem(r, c, QTableWidgetItem(text))
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setStretchLastSection(True)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table = table
        layout = QVBoxLayout(self)
        layout.addWidget(table)
        self.resize(560, 640)
