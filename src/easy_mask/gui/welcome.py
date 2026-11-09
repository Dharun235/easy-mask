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

HINT = "Open  →  click objects with AI or draw them  →  Propagate  →  Export"


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
        title = QLabel("easy-mask")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 30px; font-weight: 600; letter-spacing: -0.5px;")
        subtitle = QLabel("Masks for computer vision. Everything stays on this computer.")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: palette(placeholder-text); font-size: 14px;")

        buttons = QHBoxLayout()
        buttons.setSpacing(10)
        for text, slot, primary in (
            ("Open folder", open_folder, True),
            ("Open image", open_image, False),
            ("Open video", open_video, False),
        ):
            button = QPushButton(text)
            button.setMinimumHeight(40)
            button.setDefault(primary)  # the theme draws the default button filled
            button.clicked.connect(slot)
            buttons.addWidget(button)

        self.recent_list = QListWidget()
        self.recent_list.setMinimumWidth(520)
        self.recent_list.setMaximumHeight(180)
        self.recent_list.setTextElideMode(Qt.TextElideMode.ElideMiddle)  # keep both path ends
        self.recent_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.recent_list.itemActivated.connect(self._open_item)
        self.recent_list.itemClicked.connect(self._open_item)
        hint = QLabel(HINT)
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setWordWrap(True)
        hint.setStyleSheet("color: palette(placeholder-text); font-size: 12px;")
        recent_title = QLabel("RECENT")
        recent_title.setStyleSheet("color: palette(placeholder-text); font-size: 11px;")

        column = QVBoxLayout()  # a calm, narrow column in the middle of the window
        column.setSpacing(14)
        column.addWidget(title)
        column.addWidget(subtitle)
        column.addSpacing(10)
        column.addLayout(buttons)
        column.addSpacing(18)
        column.addWidget(recent_title)
        column.addWidget(self.recent_list)
        column.addSpacing(6)
        column.addWidget(hint)
        holder = QWidget()
        holder.setLayout(column)
        holder.setMaximumWidth(620)

        layout = QVBoxLayout(self)
        layout.addStretch(2)
        layout.addWidget(holder, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addStretch(3)

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
