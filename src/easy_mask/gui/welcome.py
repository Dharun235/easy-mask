"""Start page shown until an image is open: big open buttons and recent folders."""

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

HINT = (
    "Tip: press 1-9 to pick a label, I for AI Segment, F1 for all shortcuts.\n"
    "Everything stays on this computer."
)


class WelcomePage(QWidget):
    """Open Folder / Image / Video buttons and a clickable list of recent folders."""

    def __init__(
        self,
        open_folder: Callable[[], None],
        open_image: Callable[[], None],
        open_video: Callable[[], None],
        open_path: Callable[[Path], None],
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._open_path = open_path
        title = QLabel("<h1>easy-mask</h1><p>Local-first image annotation for masks.</p>")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        buttons = QHBoxLayout()
        for text, slot in (
            ("Open Folder...", open_folder),
            ("Open Image...", open_image),
            ("Open Video...", open_video),
        ):
            button = QPushButton(text)
            button.setMinimumHeight(40)
            button.clicked.connect(slot)
            buttons.addWidget(button)

        self.recent_list = QListWidget()
        self.recent_list.setMinimumWidth(520)
        self.recent_list.setMaximumWidth(720)
        self.recent_list.setMaximumHeight(200)
        self.recent_list.setTextElideMode(Qt.TextElideMode.ElideMiddle)  # keep both path ends
        self.recent_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.recent_list.itemActivated.connect(self._open_item)
        self.recent_list.itemClicked.connect(self._open_item)
        hint = QLabel(HINT)
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setStyleSheet("color: gray")

        layout = QVBoxLayout(self)
        layout.addStretch()
        layout.addWidget(title)
        layout.addLayout(buttons)
        layout.addWidget(QLabel("Recent folders:"), alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.recent_list, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(hint)
        layout.addStretch()

    def show_recent(self, paths: list[Path]) -> None:
        """List ``paths`` (newest first); a placeholder line if there are none."""
        self.recent_list.clear()
        for path in paths:
            item = QListWidgetItem(str(path))
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            item.setToolTip(str(path))
            self.recent_list.addItem(item)
        if not paths:
            item = QListWidgetItem("(nothing yet)")
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.recent_list.addItem(item)

    def _open_item(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.ItemDataRole.UserRole)
        if path:
            self._open_path(Path(path))
