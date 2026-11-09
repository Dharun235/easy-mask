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
    """``(group, keys, command)`` for every command with a shortcut.

    The group is the menu a command sits in, or "Tools & labels" for keyboard-only ones
    (tool letters, label numbers, next / previous image).
    """
    titles: dict[QAction, str] = {}
    for menu_action in window.menuBar().actions():
        menu = menu_action.menu()
        if isinstance(menu, QMenu):
            _collect(menu, menu_action.text().replace("&", ""), titles)
    rows = []
    for action in window.actions():
        keys = " / ".join(
            k.toString(k.SequenceFormat.NativeText) for k in action.shortcuts() if not k.isEmpty()
        )
        if keys.strip(" /"):
            group = titles.get(action, "Tools & labels")
            rows.append((group, keys, action.text().replace("&", "").rstrip(".")))
    return rows + EXTRA


def _collect(menu: QMenu, title: str, titles: dict[QAction, str]) -> None:
    for action in menu.actions():
        sub = action.menu()
        if isinstance(sub, QMenu):
            _collect(sub, title, titles)
        else:
            titles[action] = title


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
