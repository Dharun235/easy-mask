"""Side-panel list widgets: files, labels, shapes and image flags.

Each widget only knows how to *show* data it is given and to report user clicks.
The main window decides what the data is.
"""

from pathlib import Path

from PySide6.QtCore import QItemSelectionModel, QPoint, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from easy_mask.core.labels import label_color
from easy_mask.core.shapes import Shape
from easy_mask.core.video import ERASE_LABEL, is_auto


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
    """Erase, then every known label with its colour and count on this image.

    The active label (used for new shapes without asking) is shown in bold.

    Signals:
        label_clicked(str): The user clicked a label.
        label_menu_requested(str, QPoint): The user right-clicked a label (global position).
    """

    label_clicked = Signal(str)
    label_menu_requested = Signal(str, QPoint)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setToolTip(
            "Click a label to use it for new shapes (again to be asked); right-click for more"
        )
        self.itemClicked.connect(
            lambda item: self.label_clicked.emit(item.data(Qt.ItemDataRole.UserRole))
        )
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._request_menu)

    def _request_menu(self, pos: QPoint) -> None:
        item = self.itemAt(pos)
        label = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
        if label and label != ERASE_LABEL:
            self.label_menu_requested.emit(label, self.viewport().mapToGlobal(pos))

    def show_labels(
        self, labels: list[str], shapes: list[Shape], active: str | None = None
    ) -> None:
        """Replace the list with ``labels``, per-label counts and the active label.

        The first row is always Erase: shapes drawn with it cut the masks underneath.
        """
        self.clear()
        erase = QListWidgetItem("✂  Erase (cut out of masks)")
        erase.setData(Qt.ItemDataRole.UserRole, ERASE_LABEL)
        erase.setToolTip("Shapes drawn with Erase are cut out of the masks under them")
        self._bold_if(erase, active == ERASE_LABEL)
        self.addItem(erase)
        for label in labels:
            count = sum(1 for s in shapes if s.label == label)
            text = f"{label}  ({count})" if count else label
            item = QListWidgetItem(color_icon(label, labels), text)
            item.setData(Qt.ItemDataRole.UserRole, label)
            self._bold_if(item, label == active)
            self.addItem(item)

    @staticmethod
    def _bold_if(item: QListWidgetItem, bold: bool) -> None:
        font = item.font()
        font.setBold(bold)
        item.setFont(font)


class LabelPanel(QWidget):
    """The Labels panel: an "Add a label" box above the :class:`LabelList`.

    Signals:
        label_added(str): The user typed a new label name and pressed Enter.
    """

    label_added = Signal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.add_edit = QLineEdit()
        self.add_edit.setPlaceholderText("Add a label…  (type, then Enter)")
        self.add_edit.returnPressed.connect(self._add)
        self.label_list = LabelList()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self.add_edit)
        layout.addWidget(self.label_list)

    def _add(self) -> None:
        name = self.add_edit.text().strip()
        self.add_edit.clear()
        if name:
            self.label_added.emit(name)


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
