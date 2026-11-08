"""The main application window: menus, side panels, and the annotate-save loop.

Data flow for one edit::

    user input -> Canvas -> signal -> MainWindow._commit() -> History.push
                                                          -> refresh side panels
                                                          -> mark dirty / auto-save
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QActionGroup, QCloseEvent, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QWidget,
)

from easy_mask import __version__
from easy_mask.core import annotation as ann_io
from easy_mask.core.annotation import Annotation
from easy_mask.core.history import History
from easy_mask.core.shapes import Shape, ShapeType
from easy_mask.gui.canvas import Canvas
from easy_mask.gui.label_dialog import LabelDialog, LabelInfo
from easy_mask.gui.widgets import FileList, FlagList, LabelList, ShapeList
from easy_mask.io import export
from easy_mask.io.images import is_image, list_images

DUPLICATE_OFFSET = 10.0  # pixels a duplicated shape is shifted so it is visible

TOOLS: list[tuple[str, ShapeType | None, str]] = [
    ("Edit", None, "E"),
    ("Polygon", ShapeType.POLYGON, "P"),
    ("Rectangle", ShapeType.RECTANGLE, "R"),
    ("Circle", ShapeType.CIRCLE, "C"),
    ("Line", ShapeType.LINE, "L"),
    ("Line strip", ShapeType.LINESTRIP, "T"),
    ("Point", ShapeType.POINT, "O"),
]


@dataclass
class Settings:
    """Startup options, usually from the command line."""

    output_dir: Path | None = None
    labels: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    autosave: bool = False


class MainWindow(QMainWindow):
    """easy-mask's top-level window."""

    def __init__(self, settings: Settings | None = None):
        super().__init__()
        self.settings = settings or Settings()
        self.known_labels: list[str] = list(self.settings.labels)
        self.image_paths: list[Path] = []
        self.current_index: int | None = None
        self.image_size: tuple[int, int] = (0, 0)
        self.history = History()
        self.dirty = False
        self.clipboard: Shape | None = None
        self.last_label = ""  # pre-filled in the label dialog for fast repeated labelling
        # Swappable so tests can answer the label dialog without a human.
        self.ask_label = LabelDialog.ask

        self.canvas = Canvas(self)
        self.canvas.labels = self.known_labels  # same list object: new labels recolour live
        self.setCentralWidget(self.canvas)
        self._build_panels()
        self._build_actions()
        self._connect_canvas()

        self.status_label = QLabel()
        self.statusBar().addPermanentWidget(self.status_label)
        self.resize(1280, 800)
        self._update_title()

    # ------------------------------------------------------------------ construction

    def _dock(self, title: str, widget: QWidget, area: Qt.DockWidgetArea) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setObjectName(title)
        dock.setWidget(widget)
        self.addDockWidget(area, dock)
        return dock

    def _build_panels(self) -> None:
        right = Qt.DockWidgetArea.RightDockWidgetArea
        self.flag_list = FlagList()
        self.label_list = LabelList()
        self.shape_list = ShapeList()
        self.file_list = FileList()
        self.flags_dock = self._dock("Flags", self.flag_list, right)
        self._dock("Labels", self.label_list, right)
        self._dock("Shapes", self.shape_list, right)
        self._dock("Files", self.file_list, right)
        self.flags_dock.setVisible(bool(self.settings.flags))

        self.flag_list.flags_changed.connect(self._mark_dirty)
        self.shape_list.currentRowChanged.connect(
            lambda row: self.canvas.select(row if row >= 0 else None)
        )
        self.shape_list.visibility_changed.connect(self.canvas.set_shape_visible)
        self.shape_list.itemDoubleClicked.connect(lambda _item: self.edit_label())
        self.file_list.currentRowChanged.connect(self._on_file_row_changed)

    def _action(
        self,
        menu: QMenu,
        text: str,
        slot: Callable[..., object],
        shortcut: str | QKeySequence.StandardKey | list[QKeySequence] | None = None,
        checkable: bool = False,
    ) -> QAction:
        action = QAction(text, self)
        if shortcut is not None:
            action.setShortcuts(
                shortcut if isinstance(shortcut, list) else [QKeySequence(shortcut)]
            )
        action.setCheckable(checkable)
        action.triggered.connect(slot)
        menu.addAction(action)
        self.addAction(action)  # keep shortcuts active even when the menu bar is hidden
        return action

    def _build_actions(self) -> None:
        bar = self.menuBar()
        toolbar = self.addToolBar("Tools")
        toolbar.setObjectName("Tools")
        std = QKeySequence.StandardKey

        m = bar.addMenu("&File")
        self._action(m, "Open Image...", self.open_image_dialog, std.Open)
        self._action(m, "Open Folder...", self.open_folder_dialog, "Ctrl+Shift+O")
        self._action(m, "Change Annotation Folder...", self.change_output_dir)
        m.addSeparator()
        self._action(m, "Save", self.save, std.Save)
        self.autosave_action = self._action(m, "Auto Save", self._toggle_autosave, checkable=True)
        self.autosave_action.setChecked(self.settings.autosave)
        m.addSeparator()
        self._action(m, "Next Image", self.next_image, "D")
        self._action(m, "Previous Image", self.prev_image, "A")
        m.addSeparator()
        export_menu = m.addMenu("Export Folder As")
        for name in export.EXPORTERS:
            self._action(export_menu, name, lambda _=False, n=name: self.export_dataset(n))
        m.addSeparator()
        self._action(m, "Quit", self.close, std.Quit)

        m = bar.addMenu("&Edit")
        self.undo_action = self._action(m, "Undo", self.undo, std.Undo)
        self.redo_action = self._action(
            m, "Redo", self.redo, [QKeySequence("Ctrl+Shift+Z"), QKeySequence("Ctrl+Y")]
        )
        m.addSeparator()
        self._action(m, "Edit Label", self.edit_label, "Ctrl+E")
        self._action(m, "Delete Shape", self.delete_selected, std.Delete)
        self._action(m, "Duplicate Shape", self.duplicate_selected, "Ctrl+D")
        self._action(m, "Copy Shape", self.copy_selected, std.Copy)
        self._action(m, "Paste Shape", self.paste, std.Paste)

        m = bar.addMenu("&Draw")
        self.tool_group = QActionGroup(self)
        self.tool_actions: dict[ShapeType | None, QAction] = {}
        for text, mode, key in TOOLS:
            action = self._action(
                m, text, lambda _=False, md=mode: self.set_tool(md), key, checkable=True
            )
            self.tool_group.addAction(action)
            toolbar.addAction(action)
            self.tool_actions[mode] = action
        self.tool_actions[None].setChecked(True)

        m = bar.addMenu("&View")
        self._action(
            m,
            "Zoom In",
            lambda: self.canvas.zoom_by(1.25),
            [QKeySequence("Ctrl+="), QKeySequence("Ctrl++")],
        )
        self._action(m, "Zoom Out", lambda: self.canvas.zoom_by(0.8), "Ctrl+-")
        self._action(m, "Fit Window", self.canvas.fit_to_window, "F")
        self._action(m, "Actual Size", self.canvas.zoom_to_actual_size, "Ctrl+0")
        fill = self._action(m, "Fill Shapes", self._toggle_fill, checkable=True)
        fill.setChecked(True)

        m = bar.addMenu("&Help")
        self._action(m, "About", self._about)
        self._update_undo_actions()

    def _connect_canvas(self) -> None:
        self.canvas.shape_drawn.connect(self._on_shape_drawn)
        self.canvas.shapes_edited.connect(self._commit)
        self.canvas.selection_changed.connect(self.shape_list.select_silently)
        self.canvas.double_clicked_shape.connect(lambda _i: self.edit_label())
        self.canvas.delete_requested.connect(self.delete_selected)
        self.canvas.status_changed.connect(lambda text: self.status_label.setText(text))

    # ------------------------------------------------------------------ opening images

    @property
    def image_path(self) -> Path | None:
        """The image currently shown, if any."""
        if self.current_index is None:
            return None
        return self.image_paths[self.current_index]

    def open_path(self, path: Path) -> None:
        """Open an image file, or every image in a folder."""
        if path.is_dir():
            paths = list_images(path)
            if not paths:
                self._error(f"No images found in {path}")
                return
            start = 0
        elif is_image(path):
            paths = list_images(path.parent)
            start = paths.index(path)
        else:
            self._error(f"Not an image or folder: {path}")
            return
        if not self.maybe_save():
            return
        self.image_paths = paths
        self.current_index = None
        self._refresh_file_list()
        self._load(start)

    def _load(self, index: int) -> None:
        """Show image ``index`` and its annotation (if one exists)."""
        path = self.image_paths[index]
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self._error(f"Could not read image: {path}")
            return

        json_path = ann_io.annotation_path_for(path, self.settings.output_dir)
        shapes: list[Shape] = []
        flags = dict.fromkeys(self.settings.flags, False)
        if json_path.exists():
            try:
                annotation = ann_io.load(json_path)
            except (ValueError, KeyError) as exc:
                self._error(f"Could not read annotation {json_path}:\n{exc}")
                return
            shapes = annotation.shapes
            flags.update(annotation.flags)

        self.current_index = index
        self.image_size = (pixmap.width(), pixmap.height())
        self.canvas.set_image(pixmap)
        self.canvas.set_shapes(shapes)
        self.flag_list.show_flags(flags)
        if flags:
            self.flags_dock.setVisible(True)
        for shape in shapes:
            self._remember_label(shape.label)
        self.history.reset(shapes)
        self.dirty = False
        self.file_list.set_current_row_silently(index)
        self._refresh_panels()
        self._update_title()
        self.canvas.setFocus()

    def _on_file_row_changed(self, row: int) -> None:
        if row < 0 or row == self.current_index:
            return
        if self.maybe_save():
            self._load(row)
        else:
            self.file_list.set_current_row_silently(self.current_index or 0)

    def next_image(self) -> None:
        """Go to the next image in the folder."""
        self._step(+1)

    def prev_image(self) -> None:
        """Go to the previous image in the folder."""
        self._step(-1)

    def _step(self, delta: int) -> None:
        if self.current_index is None:
            return
        target = self.current_index + delta
        if 0 <= target < len(self.image_paths) and self.maybe_save():
            self._load(target)

    def open_image_dialog(self) -> None:
        """Ask for an image file and open it."""
        name, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp)"
        )
        if name:
            self.open_path(Path(name))

    def open_folder_dialog(self) -> None:
        """Ask for a folder and open all images in it."""
        name = QFileDialog.getExistingDirectory(self, "Open Folder")
        if name:
            self.open_path(Path(name))

    def change_output_dir(self) -> None:
        """Choose where annotation JSON files are read from and written to."""
        name = QFileDialog.getExistingDirectory(self, "Annotation Folder")
        if name and self.maybe_save():
            self.settings.output_dir = Path(name)
            self._refresh_file_list()
            if self.current_index is not None:
                self._load(self.current_index)

    # ------------------------------------------------------------------ saving

    def current_annotation(self) -> Annotation | None:
        """Build an :class:`Annotation` from what is on screen."""
        if self.image_path is None:
            return None
        json_path = ann_io.annotation_path_for(self.image_path, self.settings.output_dir)
        width, height = self.image_size
        return Annotation(
            image_path=ann_io.relative_image_path(self.image_path, json_path),
            image_width=width,
            image_height=height,
            shapes=[s.copy() for s in self.canvas.shapes],
            flags=self.flag_list.flags(),
        )

    def save(self) -> bool:
        """Write the current annotation to disk. Returns True on success."""
        annotation = self.current_annotation()
        if annotation is None or self.image_path is None:
            return False
        json_path = ann_io.annotation_path_for(self.image_path, self.settings.output_dir)
        try:
            ann_io.save(annotation, json_path)
        except OSError as exc:
            self._error(f"Could not save {json_path}:\n{exc}")
            return False
        self.dirty = False
        self.file_list.mark_annotated(self.current_index or 0)
        self.statusBar().showMessage(f"Saved {json_path}", 3000)
        self._update_title()
        return True

    def maybe_save(self) -> bool:
        """Handle unsaved changes before leaving an image. Returns False to stay put."""
        if not self.dirty:
            return True
        if self.settings.autosave:
            return self.save()
        answer = QMessageBox.question(
            self,
            "Unsaved changes",
            f"Save changes to {self.image_path.name if self.image_path else 'this image'}?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
        )
        if answer == QMessageBox.StandardButton.Save:
            return self.save()
        return answer == QMessageBox.StandardButton.Discard

    def _mark_dirty(self) -> None:
        self.dirty = True
        self._update_title()

    def _toggle_autosave(self, checked: bool) -> None:
        self.settings.autosave = checked

    # ------------------------------------------------------------------ editing

    def _commit(self) -> None:
        """Record the canvas state after any edit and refresh everything."""
        self.history.push(self.canvas.shapes)
        self._mark_dirty()
        self._refresh_panels()
        self.canvas.update()

    def set_tool(self, mode: ShapeType | None) -> None:
        """Switch the canvas to a drawing tool, or to edit mode with ``None``."""
        self.canvas.set_mode(mode)
        self.tool_actions[mode].setChecked(True)

    def _on_shape_drawn(self, shape: Shape) -> None:
        info = self.ask_label(self.known_labels, LabelInfo(label=self.last_label), self)
        if info is None:
            return
        shape.label, shape.group_id, shape.description = info.label, info.group_id, info.description
        self._remember_label(info.label)
        self.last_label = info.label
        self.canvas.shapes.append(shape)
        self._commit()
        self.canvas.select(len(self.canvas.shapes) - 1)

    def edit_label(self) -> None:
        """Change label / group id / description of the selected shape."""
        index = self.canvas.selected
        if index is None:
            return
        shape = self.canvas.shapes[index]
        info = self.ask_label(
            self.known_labels, LabelInfo(shape.label, shape.group_id, shape.description), self
        )
        if info is None:
            return
        shape.label, shape.group_id, shape.description = info.label, info.group_id, info.description
        self._remember_label(info.label)
        self._commit()
        self.shape_list.select_silently(index)

    def delete_selected(self) -> None:
        """Delete the selected shape."""
        index = self.canvas.selected
        if index is None:
            return
        del self.canvas.shapes[index]
        self.canvas.select(None)
        self._commit()

    def duplicate_selected(self) -> None:
        """Add a slightly shifted copy of the selected shape."""
        index = self.canvas.selected
        if index is None:
            return
        copy = self.canvas.shapes[index].copy()
        copy.translate(DUPLICATE_OFFSET, DUPLICATE_OFFSET)
        self._add_and_select(copy)

    def copy_selected(self) -> None:
        """Copy the selected shape to the in-app clipboard."""
        if self.canvas.selected is not None:
            self.clipboard = self.canvas.shapes[self.canvas.selected].copy()

    def paste(self) -> None:
        """Paste the clipboard shape at its original position (works across images)."""
        if self.clipboard is not None and self.image_path is not None:
            self._add_and_select(self.clipboard.copy())

    def _add_and_select(self, shape: Shape) -> None:
        self.canvas.shapes.append(shape)
        self._commit()
        self.canvas.select(len(self.canvas.shapes) - 1)

    def undo(self) -> None:
        """Revert the last edit."""
        self._restore(self.history.undo())

    def redo(self) -> None:
        """Re-apply the last undone edit."""
        self._restore(self.history.redo())

    def _restore(self, shapes: list[Shape] | None) -> None:
        if shapes is None:
            return
        self.canvas.set_shapes(shapes)
        self._mark_dirty()
        self._refresh_panels()

    # ------------------------------------------------------------------ export

    def export_dataset(self, fmt: str) -> None:
        """Export every annotated image in the open folder to dataset format ``fmt``."""
        if self.image_path is None:
            self._error("Open a folder first.")
            return
        if not self.maybe_save():
            return
        out = QFileDialog.getExistingDirectory(self, f"Export {fmt} to folder")
        if not out:
            return
        items = export.collect(self.image_path.parent, self.settings.output_dir)
        # Predefined labels first, then any seen in the folder, so nothing is dropped and
        # class ids follow the same order as the on-screen colours.
        for item in items:
            for shape in item.annotation.shapes:
                self._remember_label(shape.label)
        export.EXPORTERS[fmt](items, Path(out), list(self.known_labels))
        self.statusBar().showMessage(f"Exported {len(items)} images as {fmt} to {out}", 5000)

    # ------------------------------------------------------------------ view helpers

    def _toggle_fill(self, checked: bool) -> None:
        self.canvas.fill = checked
        self.canvas.update()

    def _remember_label(self, label: str) -> None:
        if label and label not in self.known_labels:
            self.known_labels.append(label)

    def _refresh_panels(self) -> None:
        self.label_list.show_labels(self.known_labels, self.canvas.shapes)
        self.shape_list.show_shapes(self.canvas.shapes, self.canvas.hidden, self.known_labels)
        self.shape_list.select_silently(
            -1 if self.canvas.selected is None else self.canvas.selected
        )
        self._update_undo_actions()

    def _refresh_file_list(self) -> None:
        annotated = {
            p
            for p in self.image_paths
            if ann_io.annotation_path_for(p, self.settings.output_dir).exists()
        }
        self.file_list.show_files(self.image_paths, annotated)

    def _update_undo_actions(self) -> None:
        self.undo_action.setEnabled(self.history.can_undo)
        self.redo_action.setEnabled(self.history.can_redo)

    def _update_title(self) -> None:
        title = "easy-mask"
        if self.image_path is not None:
            title += f" - {self.image_path.name}{' *' if self.dirty else ''}"
        self.setWindowTitle(title)

    def _error(self, message: str) -> None:
        QMessageBox.warning(self, "easy-mask", message)

    def _about(self) -> None:
        QMessageBox.about(
            self,
            "About easy-mask",
            f"easy-mask {__version__}\nLocal-first image annotation for computer-vision masks.",
        )

    def closeEvent(self, event: QCloseEvent) -> None:
        """Offer to save before quitting."""
        if self.maybe_save():
            event.accept()
        else:
            event.ignore()
