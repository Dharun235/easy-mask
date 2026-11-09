"""Side-panel widgets: files, labels (with their shapes) and image flags.

Each widget only knows how to *show* data it is given and to report user clicks.
The main window decides what the data is.
"""

from pathlib import Path

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from easy_mask.core.keypoints import is_keypoint
from easy_mask.core.labels import label_color
from easy_mask.core.shapes import Shape
from easy_mask.core.video import ObjectKey, is_auto


def as_list(labels: str | list[str]) -> list[str]:
    """One label or several, as a list."""
    return [labels] if isinstance(labels, str) else list(labels)


def quoted(labels: list[str]) -> str:
    """Labels for a menu or message: “dog”, “dog” + “wing1”, or “4 labels”."""
    if len(labels) > 3:
        return f"{len(labels)} labels"
    return " + ".join(f"“{label}”" for label in labels)


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


LABEL_ROLE = Qt.ItemDataRole.UserRole  # label rows: the label; shape rows: None
SHAPE_ROLE = Qt.ItemDataRole.UserRole + 1  # shape rows: the shape's index on the canvas
KEY_ROLE = Qt.ItemDataRole.UserRole + 2  # shape rows: the shape's object (label, group id)


def shape_text(shape: Shape) -> str:
    """How a shape reads under its label: its kind, group id, and whether it is automatic."""
    text = shape.description if is_keypoint(shape) else str(shape.shape_type)
    if shape.group_id is not None:
        text += f"  g{shape.group_id}"
    if is_auto(shape):  # interpolated or propagated, not yet checked by hand
        text += "  (auto)"
    return text


class LabelTree(QTreeWidget):
    """Labels, each with the shapes that carry it on this frame underneath.

    The tree is a checklist of what Propagate sends: ticking a label ticks all its shapes,
    shapes can be ticked one by one, and a label is half-ticked when only some are. Ticks
    belong to objects ``(label, group id)``, so they carry over to the next frames. The
    active label (used for new shapes without asking) is bold; clicking a shape row
    selects that shape.

    Signals:
        label_clicked(str): The user clicked a label row.
        label_menu_requested(str, QPoint): Right-click on a label row (global position).
        shape_menu_requested(QPoint): Right-click on a shape row (it is selected first).
        shapes_selected(list): Shape indices now selected in the tree.
        shape_double_clicked(int): A shape row was double-clicked.
    """

    label_clicked = Signal(str)
    label_menu_requested = Signal(str, QPoint)
    shape_menu_requested = Signal(QPoint)
    shapes_selected = Signal(list)
    shape_double_clicked = Signal(int)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setHeaderHidden(True)
        self.setIndentation(16)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setToolTip(
            "Tick labels to choose what Propagate sends; click a label to use it for new "
            "shapes; click a shape to select it; right-click either for more"
        )
        self.folded: set[str] = set()  # labels the user folded away
        self.ticked: set[ObjectKey] = set()  # objects ticked for Propagate
        self.ticked_labels: set[str] = set()  # ticked labels with no shape on this frame
        self.itemClicked.connect(self._on_clicked)
        self.itemChanged.connect(self._on_changed)
        self.itemSelectionChanged.connect(lambda: self.shapes_selected.emit(self.selected_shapes()))
        self.itemDoubleClicked.connect(self._on_double_clicked)
        self.itemCollapsed.connect(lambda item: self.folded.add(item.data(0, LABEL_ROLE)))
        self.itemExpanded.connect(lambda item: self.folded.discard(item.data(0, LABEL_ROLE)))
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._request_menu)

    def show_labels(
        self,
        labels: list[str],
        shapes: list[Shape],
        hidden: set[int] | None = None,
        active: str | None = None,
    ) -> None:
        """Rebuild: every label in ``labels`` with its shapes; ``hidden`` holds shape ids."""
        hidden = hidden or set()
        self.blockSignals(True)
        self.clear()
        for label in labels:
            members = [(i, s) for i, s in enumerate(shapes) if s.label == label]
            text = f"{label}  ({len(members)})" if members else label
            row = QTreeWidgetItem([text])
            row.setIcon(0, color_icon(label, labels))
            row.setData(0, LABEL_ROLE, label)
            row.setFlags(row.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            font = row.font(0)
            font.setBold(label == active)
            row.setFont(0, font)
            for index, shape in members:
                text = shape_text(shape) + ("  (hidden)" if id(shape) in hidden else "")
                child = QTreeWidgetItem([text])
                child.setData(0, SHAPE_ROLE, index)
                child.setData(0, KEY_ROLE, (shape.label, shape.group_id))
                child.setFlags(child.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                ticked = (shape.label, shape.group_id) in self.ticked
                child.setCheckState(0, Qt.CheckState.Checked if ticked else Qt.CheckState.Unchecked)
                row.addChild(child)
            self._update_row(row)
            self.addTopLevelItem(row)
            row.setExpanded(label not in self.folded)
        self.blockSignals(False)

    def label_rows(self) -> list[QTreeWidgetItem]:
        """The top-level label rows, in order."""
        rows = (self.topLevelItem(i) for i in range(self.topLevelItemCount()))
        return [row for row in rows if row is not None]

    def shape_rows(self) -> list[QTreeWidgetItem]:
        """Every shape row, label by label."""
        children = (row.child(i) for row in self.label_rows() for i in range(row.childCount()))
        return [child for child in children if child is not None]

    def select_shapes(self, indices: list[int]) -> None:
        """Select exactly the rows of shapes ``indices`` (label rows keep their selection)."""
        self.blockSignals(True)
        wanted = set(indices)
        for item in self.shape_rows():
            chosen = item.data(0, SHAPE_ROLE) in wanted
            item.setSelected(chosen)
            if chosen:
                self.scrollToItem(item)
        self.blockSignals(False)

    def checked_keys(self) -> set[ObjectKey]:
        """Objects ticked for Propagate (shown on this frame)."""
        return {
            item.data(0, KEY_ROLE)
            for item in self.shape_rows()
            if item.checkState(0) == Qt.CheckState.Checked
        }

    def checked_labels(self) -> list[str]:
        """Labels that are fully or partly ticked, in list order."""
        return [
            row.data(0, LABEL_ROLE)
            for row in self.label_rows()
            if row.checkState(0) != Qt.CheckState.Unchecked
        ]

    def _update_row(self, row: QTreeWidgetItem) -> None:
        """A label's tick from its shapes: all, some (half-ticked) or none."""
        children = [row.child(i) for i in range(row.childCount())]
        states = [c.checkState(0) == Qt.CheckState.Checked for c in children if c is not None]
        if not states:
            ticked = row.data(0, LABEL_ROLE) in self.ticked_labels
            state = Qt.CheckState.Checked if ticked else Qt.CheckState.Unchecked
        elif all(states):
            state = Qt.CheckState.Checked
        elif any(states):
            state = Qt.CheckState.PartiallyChecked
        else:
            state = Qt.CheckState.Unchecked
        row.setCheckState(0, state)

    def selected_shapes(self) -> list[int]:
        """Indices of the selected shapes, in canvas order."""
        found = (item.data(0, SHAPE_ROLE) for item in self.selectedItems())
        return sorted(i for i in found if i is not None)

    def _on_clicked(self, item: QTreeWidgetItem) -> None:
        label = item.data(0, LABEL_ROLE)
        if label is not None:
            self.label_clicked.emit(label)

    def _on_changed(self, item: QTreeWidgetItem) -> None:
        self.blockSignals(True)
        try:
            label = item.data(0, LABEL_ROLE)
            if label is not None:  # a label's tick: all its shapes follow
                on = item.checkState(0) != Qt.CheckState.Unchecked
                (self.ticked_labels.add if on else self.ticked_labels.discard)(label)
                for i in range(item.childCount()):
                    child = item.child(i)
                    if child is not None:
                        child.setCheckState(
                            0, item.checkState(0) if on else Qt.CheckState.Unchecked
                        )
                        self._remember_tick(child)
                if on and item.childCount():
                    item.setCheckState(0, Qt.CheckState.Checked)
                return
            self._remember_tick(item)
            parent = item.parent()
            if parent is not None:
                self._update_row(parent)
        finally:
            self.blockSignals(False)

    def _remember_tick(self, item: QTreeWidgetItem) -> None:
        key = item.data(0, KEY_ROLE)
        if item.checkState(0) == Qt.CheckState.Checked:
            self.ticked.add(key)
        else:
            self.ticked.discard(key)

    def _on_double_clicked(self, item: QTreeWidgetItem) -> None:
        index = item.data(0, SHAPE_ROLE)
        if index is not None:
            self.shape_double_clicked.emit(index)

    def _request_menu(self, pos: QPoint) -> None:
        item = self.itemAt(pos)
        if item is None:
            return
        where = self.viewport().mapToGlobal(pos)
        label = item.data(0, LABEL_ROLE)
        if label is not None:
            self.label_menu_requested.emit(label, where)
            return
        if not item.isSelected():
            self.clearSelection()
            item.setSelected(True)  # emits shapes_selected: the canvas follows
        self.shape_menu_requested.emit(where)


class LabelPanel(QWidget):
    """The Labels panel: an "Add a label" box above the :class:`LabelTree`.

    Signals:
        label_added(str): The user typed a new label name and pressed Enter.
    """

    label_added = Signal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.add_edit = QLineEdit()
        self.add_edit.setPlaceholderText("Add a label…  (type, then Enter)")
        self.add_edit.returnPressed.connect(self._add)
        self.tree = LabelTree()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self.add_edit)
        layout.addWidget(self.tree)

    def _add(self) -> None:
        name = self.add_edit.text().strip()
        self.add_edit.clear()
        if name:
            self.label_added.emit(name)


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
