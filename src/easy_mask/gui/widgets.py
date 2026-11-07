"""Side-panel list widgets: files, labels, shapes and image flags.

Each widget only knows how to *show* data it is given and to report user clicks.
The main window decides what the data is.
"""

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import QListWidget, QListWidgetItem

from easy_mask.core.labels import label_color
from easy_mask.core.shapes import Shape


def color_icon(label: str, size: int = 12) -> QIcon:
    """A small square icon filled with ``label``'s colour."""
    pixmap = QPixmap(size, size)
    pixmap.fill(QColor(*label_color(label)))
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

    def mark_annotated(self, row: int) -> None:
        """Tick one row after its annotation is saved."""
        if 0 <= row < self.count():
            self.item(row).setCheckState(Qt.CheckState.Checked)

    def set_current_row_silently(self, row: int) -> None:
        """Highlight ``row`` without emitting a selection signal."""
        self.blockSignals(True)
        self.setCurrentRow(row)
        self.blockSignals(False)


class LabelList(QListWidget):
    """Every known label with its colour and how many shapes on this image use it."""

    def show_labels(self, labels: list[str], shapes: list[Shape]) -> None:
        """Replace the list with ``labels`` and per-label shape counts."""
        self.clear()
        for label in labels:
            count = sum(1 for s in shapes if s.label == label)
            item = QListWidgetItem(color_icon(label), f"{label}  ({count})" if count else label)
            item.setData(Qt.ItemDataRole.UserRole, label)
            self.addItem(item)


class ShapeList(QListWidget):
    """Shapes on the current image. The checkbox toggles visibility on the canvas.

    Signals:
        visibility_changed(int, bool): Row index and its new visibility.
    """

    visibility_changed = Signal(int, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.itemChanged.connect(self._on_item_changed)

    def show_shapes(self, shapes: list[Shape], hidden: set[int]) -> None:
        """Replace the list with ``shapes``; ``hidden`` holds ``id()`` of hidden shapes."""
        self.blockSignals(True)
        self.clear()
        for shape in shapes:
            text = f"{shape.label}  [{shape.shape_type}]"
            if shape.group_id is not None:
                text += f"  g{shape.group_id}"
            item = QListWidgetItem(color_icon(shape.label), text)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            visible = id(shape) not in hidden
            item.setCheckState(Qt.CheckState.Checked if visible else Qt.CheckState.Unchecked)
            self.addItem(item)
        self.blockSignals(False)

    def select_silently(self, row: int) -> None:
        """Select ``row`` (or clear with -1) without emitting signals."""
        self.blockSignals(True)
        self.setCurrentRow(row)
        if row < 0:
            self.clearSelection()
        self.blockSignals(False)

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        self.visibility_changed.emit(self.row(item), item.checkState() == Qt.CheckState.Checked)


class FlagList(QListWidget):
    """Image-level flags as checkboxes.

    Signals:
        flags_changed(): The user ticked or unticked a flag.
    """

    flags_changed = Signal()

    def __init__(self, parent=None):
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
