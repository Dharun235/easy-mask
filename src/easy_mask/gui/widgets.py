"""Side-panel list widgets: files, labels, shapes and image flags.

Each widget only knows how to *show* data it is given and to report user clicks.
The main window decides what the data is.
"""

from pathlib import Path

from PySide6.QtCore import QItemSelectionModel, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import QAbstractItemView, QListWidget, QListWidgetItem, QWidget

from easy_mask.core.labels import label_color
from easy_mask.core.shapes import Shape
from easy_mask.core.video import is_auto


def color_icon(label: str, labels: list[str], size: int = 12) -> QIcon:
    """A small square icon filled with ``label``'s colour (see ``label_color``)."""
    pixmap = QPixmap(size, size)
    pixmap.fill(QColor(*label_color(label, labels)))
    return QIcon(pixmap)


class FileList(QListWidget):
    """Images in the open folder; annotated ones are ticked."""

    def show_files(self, paths: list[Path], annotated: set[Path]) -> None:
        """Replace the list with ``paths``, ticking those in ``annotated``."""
        self.blockSignals(True)
        self.clear()
        for path in paths:
            item = QListWidgetItem(path.name)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if path in annotated else Qt.CheckState.Unchecked
            )
            item.setToolTip(str(path))
            self.addItem(item)
        self.blockSignals(False)

    def mark_annotated(self, row: int, annotated: bool = True) -> None:
        """Tick one row after its annotation is saved (untick when it is removed)."""
        if 0 <= row < self.count():
            state = Qt.CheckState.Checked if annotated else Qt.CheckState.Unchecked
            self.item(row).setCheckState(state)

    def set_current_row_silently(self, row: int) -> None:
        """Highlight ``row`` without emitting a selection signal."""
        self.blockSignals(True)
        self.setCurrentRow(row)
        self.blockSignals(False)


class LabelList(QListWidget):
    """Every known label with its colour, number key and count on this image.

    The active label (used for new shapes without asking) is shown in bold.

    Signals:
        label_clicked(str): The user clicked a label.
    """

    label_clicked = Signal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setToolTip("Click a label to use it for new shapes (keys 1-9); click again to ask")
        self.itemClicked.connect(
            lambda item: self.label_clicked.emit(item.data(Qt.ItemDataRole.UserRole))
        )

    def show_labels(
        self, labels: list[str], shapes: list[Shape], active: str | None = None
    ) -> None:
        """Replace the list with ``labels``, per-label counts and the active label."""
        self.clear()
        for number, label in enumerate(labels, start=1):
            count = sum(1 for s in shapes if s.label == label)
            key = f"{number}  " if number <= 9 else "    "
            text = f"{key}{label}  ({count})" if count else f"{key}{label}"
            item = QListWidgetItem(color_icon(label, labels), text)
            item.setData(Qt.ItemDataRole.UserRole, label)
            if label == active:
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            self.addItem(item)


class ShapeList(QListWidget):
    """Shapes on the current image. The checkbox toggles visibility on the canvas.

    Signals:
        visibility_changed(int, bool): Row index and its new visibility.
    """

    visibility_changed = Signal(int, bool)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.itemChanged.connect(self._on_item_changed)

    def show_shapes(self, shapes: list[Shape], hidden: set[int], labels: list[str]) -> None:
        """Replace the list with ``shapes``; ``hidden`` holds ``id()`` of hidden shapes."""
        self.blockSignals(True)
        self.clear()
        for shape in shapes:
            text = f"{shape.label}  [{shape.shape_type}]"
            if shape.group_id is not None:
                text += f"  g{shape.group_id}"
            if is_auto(shape):  # interpolated or propagated, not yet checked by hand
                text += "  (auto)"
            item = QListWidgetItem(color_icon(shape.label, labels), text)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            visible = id(shape) not in hidden
            item.setCheckState(Qt.CheckState.Checked if visible else Qt.CheckState.Unchecked)
            self.addItem(item)
        self.blockSignals(False)

    def select_rows(self, rows: list[int]) -> None:
        """Select exactly ``rows`` without emitting signals."""
        self.blockSignals(True)
        self.clearSelection()
        for row in rows:
            if 0 <= row < self.count():
                self.item(row).setSelected(True)
        if rows:
            self.setCurrentRow(rows[-1], QItemSelectionModel.SelectionFlag.NoUpdate)
        self.blockSignals(False)

    def selected_rows(self) -> list[int]:
        """Selected rows, top to bottom."""
        return sorted(self.row(item) for item in self.selectedItems())

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        self.visibility_changed.emit(self.row(item), item.checkState() == Qt.CheckState.Checked)


class FlagList(QListWidget):
    """Image-level flags as checkboxes.

    Signals:
        flags_changed(): The user ticked or unticked a flag.
    """

    flags_changed = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.itemChanged.connect(lambda _item: self.flags_changed.emit())

    def show_flags(self, flags: dict[str, bool]) -> None:
        """Replace the list with ``flags``."""
        self.blockSignals(True)
        self.clear()
        for name, value in flags.items():
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if value else Qt.CheckState.Unchecked)
            self.addItem(item)
        self.blockSignals(False)

    def flags(self) -> dict[str, bool]:
        """Current flag values, in display order."""
        return {
            self.item(i).text(): self.item(i).checkState() == Qt.CheckState.Checked
            for i in range(self.count())
        }
