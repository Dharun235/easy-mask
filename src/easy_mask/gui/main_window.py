"""The main application window: menus, side panels, and the annotate-save loop.

Data flow for one edit::

    user input -> Canvas -> signal -> MainWindow._commit() -> History.push
                                                          -> refresh side panels
                                                          -> mark dirty / auto-save
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QAction, QActionGroup, QCloseEvent, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QSizePolicy,
    QStackedWidget,
    QWidget,
)

from easy_mask import __version__
from easy_mask.ai import efficient_sam
from easy_mask.ai.segmenter import Segmenter
from easy_mask.ai.weights import ProgressFn
from easy_mask.core import annotation as ann_io
from easy_mask.core import masks, operations
from easy_mask.core.annotation import Annotation
from easy_mask.core.history import History
from easy_mask.core.shapes import Shape, ShapeType
from easy_mask.gui.canvas import Canvas
from easy_mask.gui.display_dialog import DisplayDialog
from easy_mask.gui.export_worker import ExportWorker
from easy_mask.gui.icons import icon
from easy_mask.gui.label_dialog import LabelDialog, LabelInfo
from easy_mask.gui.progress import ProgressPanel
from easy_mask.gui.session import Session
from easy_mask.gui.shortcuts import ShortcutsDialog
from easy_mask.gui.threshold_dialog import ThresholdDialog
from easy_mask.gui.tools.ai_segment import AiSegmentTool
from easy_mask.gui.tools.base import CanvasTool
from easy_mask.gui.tools.brush import BrushTool
from easy_mask.gui.tools.region import MagicWandTool
from easy_mask.gui.video_import import (
    VIDEO_FILTER,
    ImportOptions,
    VideoImportDialog,
    VideoImportWorker,
    default_options,
)
from easy_mask.gui.video_menu import VideoMenu
from easy_mask.gui.welcome import WelcomePage
from easy_mask.gui.widgets import FileList, FlagList, LabelList, ShapeList
from easy_mask.gui.workflow import Workflow
from easy_mask.io import export
from easy_mask.io.images import is_image, list_images
from easy_mask.io.video import frames_dir_for, is_video, planned_count, probe

PREFETCH_AHEAD = 3  # images encoded ahead for AI Segment: the current one and the next two
DUPLICATE_OFFSET = 10.0  # pixels a duplicated shape is shifted so it is visible

# Toolbar entries: (name, short label, tool, shortcut, icon, group). A ShapeType draws that
# shape, None is edit mode, and a string names a pixel tool from MainWindow.pixel_tools. A
# new group starts a new section of the toolbar; MENU_ONLY tools (not masks) are kept out of
# the toolbar to keep it calm but stay in the Draw menu with their keys.
MENU_ONLY = "menu"
TOOLS: list[tuple[str, str, ShapeType | str | None, str, str, str]] = [
    ("Select / Edit", "Select", None, "E", "edit", "select"),
    ("Polygon", "Polygon", ShapeType.POLYGON, "P", "polygon", "shapes"),
    ("Rectangle", "Box", ShapeType.RECTANGLE, "R", "rectangle", "shapes"),
    ("Circle", "Circle", ShapeType.CIRCLE, "C", "circle", "shapes"),
    ("Line", "Line", ShapeType.LINE, "L", "line", MENU_ONLY),
    ("Line strip", "Path", ShapeType.LINESTRIP, "T", "linestrip", MENU_ONLY),
    ("Point", "Point", ShapeType.POINT, "O", "point", MENU_ONLY),
    ("Brush", "Brush", "brush", "B", "brush", "paint"),
    ("Eraser", "Erase", "eraser", "X", "eraser", "paint"),
    ("AI Segment", "AI", "ai", "I", "ai", "smart"),
    ("Magic wand", "Wand", "wand", "W", "wand", "smart"),
]


def ask_number(
    parent: QWidget, title: str, label: str, value: float, minimum: float, maximum: float
) -> float | None:
    """Ask for a number in a small dialog. Returns None if cancelled."""
    number, ok = QInputDialog.getDouble(parent, title, label, value, minimum, maximum, 2)
    return number if ok else None


@dataclass
class Settings:
    """Startup options, usually from the command line."""

    output_dir: Path | None = None
    labels: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    autosave: bool = True  # on by default: edits are undoable and saves are atomic
    strict_labels: bool = False  # only labels from ``labels`` are accepted
    ai_model: str = efficient_sam.DEFAULT_MODEL
    frame_step: int = 1  # when opening a video: keep every n-th frame
    max_frames: int | None = None  # when opening a video: stop after this many frames
    video_model: str = "efficienttam_ti_512"  # propagation model, see propagation/backend.py


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
        self.last_export: tuple[str, Path] | None = None  # (format, folder), per project
        self.export_worker: ExportWorker | None = None
        self.video_worker: VideoImportWorker | None = None
        self.ask_video_import = VideoImportDialog.ask  # swappable in tests
        # Tidy AI / smart-tool masks (specks, pinholes) before storing them; brush is exact.
        self.auto_cleanup = True
        # With an active label, new shapes take it directly instead of opening the dialog.
        self.active_label: str | None = None
        # Swappable so tests can answer the label dialog without a human.
        self.ask_label: Callable[[list[str], LabelInfo, QWidget | None], LabelInfo | None] = (
            partial(LabelDialog.ask, allowed=self._allowed_labels())
        )
        self.ask_number = ask_number
        self.load_segmenter: Callable[[ProgressFn], Segmenter] = lambda progress: (
            efficient_sam.load(self.settings.ai_model, progress)
        )

        self.canvas = Canvas(self)
        self.ai_tool = self._make_ai_tool()
        self.pixel_tools: dict[str, CanvasTool] = {
            "brush": BrushTool(self._apply_stroke),
            "eraser": BrushTool(self._apply_stroke, erase=True),
            "wand": MagicWandTool(self._apply_smart),
            "ai": self.ai_tool,
        }
        self.threshold_dialog: ThresholdDialog | None = None
        self.display_dialog: DisplayDialog | None = None
        self.canvas.labels = self.known_labels  # same list object: new labels recolour live
        self.session = Session(self)  # remembered layout, tool sizes, recent folders, projects
        self.welcome = WelcomePage(
            self.open_folder_dialog,
            self.open_image_dialog,
            lambda: self.video.open_video_dialog(),
            self.open_path,
        )
        self.pages = QStackedWidget()  # welcome page until an image is open, then the canvas
        self.pages.addWidget(self.welcome)
        self.pages.addWidget(self.canvas)
        self.setCentralWidget(self.pages)
        self._build_panels()
        self._build_actions()
        self.video = VideoMenu(self)
        self._add_video_buttons()
        self._connect_canvas()
        self.workflow = Workflow(self)  # review mode, copy previous, statistics

        self.progress = ProgressPanel()
        self.statusBar().addPermanentWidget(self.progress)
        self.active_label_status = QLabel()
        self.statusBar().addPermanentWidget(self.active_label_status)
        self.status_label = QLabel()
        self.statusBar().addPermanentWidget(self.status_label)
        self._show_active_label()
        self.resize(1280, 800)
        self.session.restore()
        self.welcome.show_recent(self.session.recent())
        self.shortcuts_dialog: ShortcutsDialog | None = None
        self._update_title()

    # ------------------------------------------------------------------ construction

    def _make_ai_tool(self) -> AiSegmentTool:
        tool = AiSegmentTool(
            self._apply_smart,
            lambda progress: self.load_segmenter(progress),  # late-bound: tests swap it
            lambda: str(self.image_path) if self.image_path else None,
        )
        tool.status.connect(lambda text: self.statusBar().showMessage(text))
        tool.error.connect(self._on_ai_error)
        tool.encoded.connect(self._prefetch_ahead)
        tool.busy.connect(self._on_ai_busy)
        tool.idle.connect(lambda: self.progress.end("ai"))
        return tool

    def _on_ai_busy(self, text: str, done: int, total: int) -> None:
        if self.progress.is_running("ai"):
            self.progress.report("ai", done, total, text)
        else:
            self.progress.begin("ai", text, total)

    def _prefetch_ahead(self) -> None:
        """Head start for AI Segment on this and the next images (once the model is up)."""
        if self.current_index is not None:
            ahead = self.image_paths[self.current_index : self.current_index + PREFETCH_AHEAD]
            self.ai_tool.prefetch([str(p) for p in ahead])

    def _on_ai_error(self, message: str) -> None:
        self.set_tool(None)
        self._error(message)

    def _dock(self, title: str, widget: QWidget, area: Qt.DockWidgetArea) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setObjectName(title)
        dock.setWidget(widget)
        # No float / close buttons: less chrome. Panels can still be moved, and toggled
        # from View > Panels.
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable)
        self.addDockWidget(area, dock)
        self.docks.append(dock)
        return dock

    def _build_panels(self) -> None:
        right = Qt.DockWidgetArea.RightDockWidgetArea
        self.docks: list[QDockWidget] = []
        self.flag_list = FlagList()
        self.label_list = LabelList()
        self.label_list.label_clicked.connect(self._on_label_clicked)
        self.shape_list = ShapeList()
        self.file_list = FileList()
        self.flags_dock = self._dock("Flags", self.flag_list, right)
        self._dock("Labels", self.label_list, right)
        self._dock("Shapes", self.shape_list, right)
        self._dock("Files", self.file_list, right)
        self._show_panels(False)  # the start page stands alone; panels appear with an image

        self.flag_list.flags_changed.connect(self._mark_dirty)
        self.shape_list.itemSelectionChanged.connect(
            lambda: self.canvas.set_selection(self.shape_list.selected_rows())
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
        toolbar = self.addToolBar("Tools")  # one quiet row: file | tools | export
        self.toolbar = toolbar
        toolbar.setObjectName("Tools")
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        toolbar.setIconSize(QSize(20, 20))
        std = QKeySequence.StandardKey

        m = bar.addMenu("&File")
        self._action(m, "Open Image...", self.open_image_dialog, std.Open)
        open_folder = self._action(m, "Open Folder...", self.open_folder_dialog, "Ctrl+Shift+O")
        m.addMenu(self.session.recent_menu)
        import_video = self._action(m, "Import Video Frames...", self.import_video_dialog)
        self._action(m, "Change Annotation Folder...", self.change_output_dir)
        m.addSeparator()
        save = self._action(m, "Save", self.save, std.Save)
        self.autosave_action = self._action(m, "Auto Save", self._toggle_autosave, checkable=True)
        self.autosave_action.setChecked(self.settings.autosave)
        m.addSeparator()
        next_image = self._action(m, "Next Image", self.next_image, "D")
        prev_image = self._action(m, "Previous Image", self.prev_image, "A")
        m.addSeparator()
        export_menu = m.addMenu("Export Folder As")
        for name in export.EXPORTERS:
            self._action(export_menu, name, lambda _=False, n=name: self.export_dataset(n))
        self._export_again_action = self._action(
            m, "Export Again", self.export_again, "Ctrl+Shift+E"
        )
        self.export_on_close_action = self._action(
            m,
            "Export Again When Closing",
            lambda _=False: self.session.save_project(),
            checkable=True,
        )
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
        self.tool_actions: dict[ShapeType | str | None, QAction] = {}
        group = None
        for text, short, mode, key, icon_name, tool_group in TOOLS:
            action = self._action(
                m, text, lambda _=False, md=mode: self.set_tool(md), key, checkable=True
            )
            action.setIcon(icon(icon_name))
            action.setIconText(short)
            action.setToolTip(f"{text}  ·  {key}")
            self.tool_group.addAction(action)
            self.tool_actions[mode] = action
            if tool_group == MENU_ONLY:
                continue
            if group is not None and tool_group != group:
                toolbar.addSeparator()
            group = tool_group
            toolbar.addAction(action)
        self.tool_actions[None].setChecked(True)

        m = bar.addMenu("&Mask")
        self._action(m, "Union", lambda: self.combine("union"))
        self._action(m, "Subtract", lambda: self.combine("subtract"))
        self._action(m, "Intersect", lambda: self.combine("intersect"))
        m.addSeparator()
        self._action(m, "Fill Holes", lambda: self.map_selected(masks.fill_holes))
        self._action(m, "Remove Small Specks...", self.remove_specks)
        self._action(m, "Grow 1 px", lambda: self.map_selected(lambda f: masks.grow(f, 1)))
        self._action(m, "Shrink 1 px", lambda: self.map_selected(lambda f: masks.shrink(f, 1)))
        self._action(m, "Smooth Edges", lambda: self.map_selected(masks.smooth))
        m.addSeparator()
        self.cleanup_action = self._action(
            m, "Auto Clean-Up of AI / Smart-Tool Masks", self._toggle_cleanup, checkable=True
        )
        self.cleanup_action.setChecked(self.auto_cleanup)
        m.addSeparator()
        self._action(m, "Threshold / Colour Range...", self.open_threshold, "Ctrl+T")
        m.addSeparator()
        self._action(m, "Convert to Mask", self.convert_to_masks)
        self._action(m, "Convert to Polygons...", self.convert_to_polygons)

        m = bar.addMenu("&Labels")
        for number in range(1, 10):
            self._action(
                m,
                f"Use Label {number}",
                lambda _=False, n=number: self.set_active_label_number(n),
                str(number),
            )
        self._action(m, "Ask for Label Every Time", lambda: self.set_active_label(None), "0")

        m = bar.addMenu("&View")
        panels = m.addMenu("Panels")
        for dock in self.docks:
            # Qt disables a dock's own toggle action without a close button, so use ours.
            toggle = QAction(dock.windowTitle(), self, checkable=True)
            toggle.toggled.connect(dock.setVisible)
            dock.visibilityChanged.connect(toggle.setChecked)
            panels.addAction(toggle)
        m.addSeparator()
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
        self._action(m, "Adjust Display...", self.open_display)

        # File and history actions go in front of the tools; export sits alone on the right.
        first_tool = toolbar.actions()[0]
        for section in (
            (
                (open_folder, "open", "Open folder  ·  Ctrl+Shift+O"),
                (import_video, "video", "Import video frames"),
                (save, "save", "Save  ·  Ctrl+S"),
            ),
            (
                (self.undo_action, "undo", "Undo  ·  Ctrl+Z"),
                (self.redo_action, "redo", "Redo  ·  Ctrl+Shift+Z"),
            ),
            (
                (prev_image, "prev", "Previous image  ·  A"),
                (next_image, "next", "Next image  ·  D"),
            ),
        ):
            for action, icon_name, tip in section:
                action.setIcon(icon(icon_name))
                action.setToolTip(tip)
                toolbar.insertAction(first_tool, action)
            toolbar.insertSeparator(first_tool)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)
        self._export_again_action.setIcon(icon("export"))
        self._export_again_action.setToolTip(
            "Export again (last format and folder)  ·  Ctrl+Shift+E"
        )
        toolbar.addAction(self._export_again_action)

        m = bar.addMenu("&Help")
        self._action(
            m, "Keyboard Shortcuts", self.show_shortcuts, [QKeySequence("F1"), QKeySequence("?")]
        )
        self._action(m, "About", self._about)
        self._update_undo_actions()

    def _connect_canvas(self) -> None:
        self.canvas.shape_drawn.connect(self._on_shape_drawn)
        self.canvas.shapes_edited.connect(self._commit)
        self.canvas.selection_changed.connect(self.shape_list.select_rows)
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
        """Open an image file, every image in a folder, or the frames of a video.

        A video whose frames were extracted before opens straight away; otherwise the
        import dialog asks how to turn it into images.
        """
        if is_video(path):
            frames = frames_dir_for(path)
            if frames.is_dir() and list_images(frames):
                path = frames
            else:
                self.import_video(path)
                return
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
        folder = path if path.is_dir() else path.parent
        self.session.open_folder(folder)
        self.session.add_recent(folder)
        self.image_paths = paths
        self.current_index = None
        self._refresh_file_list()
        self._load(start)

    # ------------------------------------------------------------------ video import

    def import_video_dialog(self) -> None:
        """File > Import Video Frames: pick a video, then choose how to import it."""
        name, _ = QFileDialog.getOpenFileName(self, "Import Video Frames", "", VIDEO_FILTER)
        if name:
            self.import_video(Path(name))

    def import_video(self, video: Path) -> None:
        """Ask how to turn ``video`` into images, then extract them in the background."""
        if self.video_worker is not None:
            self.statusBar().showMessage("A video import is already running", 4000)
            return
        try:
            info = probe(video)
        except ValueError as exc:
            self._error(str(exc))
            return
        defaults = default_options(video, self.settings.frame_step, self.settings.max_frames)
        options = self.ask_video_import(defaults, info, self)
        if options is None:
            return
        if options.out_dir.is_dir() and list_images(options.out_dir):
            self.statusBar().showMessage(f"{options.out_dir} already has images: opened", 5000)
            self.open_path(options.out_dir)
            return
        worker = VideoImportWorker(options)
        planned = planned_count(info.frames, options.step, options.max_frames)
        self.progress.begin("video", f"Importing {video.name}", planned, cancel=worker.cancel)
        worker.progress.connect(lambda done, total: self.progress.report("video", done, total))
        worker.failed.connect(self._error)
        worker.done.connect(lambda folder: self._on_video_imported(folder, options))
        worker.finished.connect(self._on_video_worker_finished)
        self.video_worker = worker
        worker.start()

    def _on_video_imported(self, folder: Path, options: ImportOptions) -> None:
        count = len(list_images(folder))
        self.statusBar().showMessage(f"Imported {count} images to {folder}", 5000)
        if options.open_after:
            self.open_path(folder)

    def _on_video_worker_finished(self) -> None:
        if self.video_worker is not None:
            self.video_worker.deleteLater()
        self.video_worker = None
        self.progress.end("video")

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
        if self.pages.currentWidget() is not self.canvas:
            self.pages.setCurrentWidget(self.canvas)
            self._show_panels(True)
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
        self._prefetch_ahead()

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
            carry = self.video.capture_for_tracking() if delta == 1 else None
            self._load(target)
            if carry is not None and self.current_index == target:
                self.video.track(carry)

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
            self.session.save_project()
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

    def set_tool(self, tool: ShapeType | str | None) -> None:
        """Switch to a shape tool, a pixel tool by name (``"brush"``), or edit mode (None)."""
        if isinstance(tool, str) and not isinstance(tool, ShapeType):
            self.canvas.set_tool(self.pixel_tools[tool])
        else:
            self.canvas.set_mode(tool)
        self.tool_actions[tool].setChecked(True)

    def _allowed_labels(self) -> list[str] | None:
        """The only labels accepted with ``--strict-labels``, else None (anything goes)."""
        return list(self.settings.labels) if self.settings.strict_labels else None

    def _ask_label(self, initial: LabelInfo) -> LabelInfo | None:
        """Ask for a label; with strict labels, refuse any outside the label list."""
        info = self.ask_label(self.known_labels, initial, self)
        allowed = self._allowed_labels()
        if info is not None and allowed is not None and info.label not in allowed:
            self.statusBar().showMessage(f"'{info.label}' is not in the label list", 5000)
            return None
        return info

    # ------------------------------------------------------------------ active label

    def set_active_label(self, label: str | None) -> None:
        """Label new shapes with ``label`` without asking, or ask every time with None."""
        if label is not None:
            self._remember_label(label)
        self.active_label = label
        self.session.save_project()
        self._show_active_label()
        self.label_list.show_labels(self.known_labels, self.canvas.shapes, self.active_label)

    def set_active_label_number(self, number: int) -> None:
        """Make the ``number``-th known label (1-based) active; ignored if there is none."""
        if 1 <= number <= len(self.known_labels):
            self.set_active_label(self.known_labels[number - 1])

    def _on_label_clicked(self, label: str) -> None:
        """Clicking a label activates it; clicking the active label switches back to asking."""
        self.set_active_label(None if label == self.active_label else label)

    def _show_active_label(self) -> None:
        if self.active_label is None:
            text = "Label: ask each time (1-9 to pick)"
        else:
            text = f"Label: {self.active_label}  (0 = ask)"
        self.active_label_status.setText(text)

    def _label_for_new_shape(self) -> LabelInfo | None:
        """The active label, or ask the user (None if they cancel)."""
        if self.active_label is not None:
            return LabelInfo(label=self.active_label)
        return self._ask_label(LabelInfo(label=self.last_label))

    def _on_shape_drawn(self, shape: Shape) -> None:
        info = self._label_for_new_shape()
        if info is None:
            return
        shape.label, shape.group_id, shape.description = info.label, info.group_id, info.description
        self._remember_label(info.label)
        self.last_label = info.label
        self.canvas.shapes.append(shape)
        self._commit()
        self.canvas.select(len(self.canvas.shapes) - 1)

    # ------------------------------------------------------------------ masks

    @property
    def _size(self) -> tuple[int, int]:
        """Current image ``(height, width)``."""
        return self.image_size[1], self.image_size[0]

    def _replace_shapes(self, shapes: list[Shape], select: int | None = None) -> None:
        self.canvas.set_shapes(shapes)
        self._commit()
        self.canvas.select(select)

    def _toggle_cleanup(self, checked: bool) -> None:
        self.auto_cleanup = checked

    def _apply_smart(self, full: np.ndarray, index: int | None) -> None:
        """Store an AI / smart-tool result, tidied first if auto clean-up is on."""
        if self.auto_cleanup:
            full = masks.clean_up(full, masks.cleanup_area(*full.shape))
        self._apply_stroke(full, index)

    def _apply_stroke(self, full: np.ndarray, index: int | None) -> None:
        """Store a finished brush/eraser stroke (called by the brush tool)."""
        shapes = list(self.canvas.shapes)
        if index is None:  # a new mask: active label, or ask for one
            info = self._label_for_new_shape()
            new = None
            if info is not None:
                new = masks.to_shape(
                    info.label, full, group_id=info.group_id, description=info.description
                )
            if new is None:
                self.canvas.update()
                return
            self._remember_label(new.label)
            self.last_label = new.label
            self._replace_shapes([*shapes, new], select=len(shapes))
            return
        changed = operations.with_pixels(shapes[index], full)
        if changed is None:
            del shapes[index]
            self._replace_shapes(shapes)
        else:
            shapes[index] = changed
            self._replace_shapes(shapes, select=index)

    def combine(self, op: operations.CombineOp) -> None:
        """Union / subtract / intersect the selected shapes; the first picked is the target."""
        if len(self.canvas.selection) < 2:
            self.statusBar().showMessage("Select two or more shapes (Shift+click) to combine", 4000)
            return
        shapes, index = operations.combine(
            self.canvas.shapes, self.canvas.selection, op, *self._size
        )
        self._replace_shapes(shapes, select=index)

    def map_selected(self, fn: operations.MaskFn) -> None:
        """Apply a mask clean-up operation to every selected shape."""
        if self.canvas.selection:
            selection = list(self.canvas.selection)
            shapes = operations.map_pixels(self.canvas.shapes, selection, fn, *self._size)
            kept_all = len(shapes) == len(self.canvas.shapes)
            self._replace_shapes(shapes, select=selection[-1] if kept_all else None)

    def remove_specks(self) -> None:
        """Remove blobs smaller than a size the user picks."""
        if not self.canvas.selection:
            return
        area = self.ask_number(
            self, "Remove Small Specks", "Minimum blob size (pixels):", 50, 1, 1_000_000
        )
        if area is not None:
            self.map_selected(lambda full: masks.remove_small(full, int(area)))

    def convert_to_masks(self) -> None:
        """Turn selected polygons / rectangles / circles into mask shapes."""
        if self.canvas.selection:
            shapes = operations.to_masks(self.canvas.shapes, self.canvas.selection, *self._size)
            self._replace_shapes(shapes)

    def convert_to_polygons(self) -> None:
        """Turn selected masks into polygons, simplified to a tolerance the user picks."""
        if not self.canvas.selection:
            return
        tolerance = self.ask_number(
            self, "Convert to Polygons", "Max deviation (pixels):", 1.0, 0.1, 50.0
        )
        if tolerance is not None:
            shapes = operations.to_polygons(
                self.canvas.shapes, self.canvas.selection, *self._size, tolerance=tolerance
            )
            self._replace_shapes(shapes)

    def open_threshold(self) -> None:
        """Show the threshold / colour-range panel; its preview replaces the active tool."""
        if self.image_path is None:
            return
        if self.threshold_dialog is None:
            self.threshold_dialog = ThresholdDialog(self.canvas, self._apply_smart, self)
            self.threshold_dialog.finished.connect(self._threshold_closed)
        self.canvas.set_tool(self.threshold_dialog.tool)
        self.threshold_dialog.refresh()
        self.threshold_dialog.show()

    def _threshold_closed(self) -> None:
        if self.threshold_dialog is not None and self.canvas.tool is self.threshold_dialog.tool:
            self.set_tool(None)

    def edit_label(self) -> None:
        """Change label / group id / description of the selected shape."""
        index = self.canvas.selected
        if index is None:
            return
        shape = self.canvas.shapes[index]
        info = self._ask_label(LabelInfo(shape.label, shape.group_id, shape.description))
        if info is None:
            return
        shape.label, shape.group_id, shape.description = info.label, info.group_id, info.description
        self._remember_label(info.label)
        self._commit()
        self.shape_list.select_rows([index])

    def delete_selected(self) -> None:
        """Delete every selected shape."""
        if not self.canvas.selection:
            return
        selected = set(self.canvas.selection)
        self._replace_shapes([s for i, s in enumerate(self.canvas.shapes) if i not in selected])

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
        self.last_export = (fmt, Path(out))
        self.session.save_project()
        self._export(fmt, Path(out))

    def export_again(self) -> None:
        """Repeat the last export (format and folder) without asking."""
        if self.last_export is None:
            self.statusBar().showMessage("Export once with File > Export Folder As first", 5000)
            return
        if self.image_path is not None and self.maybe_save():
            self._export(*self.last_export)

    def _export(self, fmt: str, out: Path, wait: bool = False) -> None:
        """Export the open folder in the background; ``wait`` blocks until done (on close)."""
        assert self.image_path is not None
        if self.export_worker is not None:
            self.statusBar().showMessage("An export is already running", 4000)
            return
        items = export.collect(self.image_path.parent, self.settings.output_dir)
        # Predefined labels first, then any seen in the folder, so nothing is dropped and
        # class ids follow the same order as the on-screen colours.
        for item in items:
            for shape in item.annotation.shapes:
                self._remember_label(shape.label)
        worker = ExportWorker(items, fmt, out, list(self.known_labels))
        self.progress.begin("export", f"Exporting {fmt}", total=len(items), cancel=worker.cancel)
        worker.progress.connect(lambda done, total: self.progress.report("export", done, total))
        worker.failed.connect(self._error)
        worker.exported.connect(
            lambda n: self.statusBar().showMessage(f"Exported {n} images as {fmt} to {out}", 5000)
        )
        worker.finished.connect(self._on_export_finished)
        self.export_worker = worker
        worker.start()
        if wait:
            worker.wait()
            self._on_export_finished()

    def _on_export_finished(self) -> None:
        if self.export_worker is not None:
            self.export_worker.deleteLater()
        self.export_worker = None
        self.progress.end("export")

    # ------------------------------------------------------------------ view helpers

    def open_display(self) -> None:
        """Show the display-adjustment panel (brightness, CLAHE, mask opacity, ...)."""
        if self.display_dialog is None:
            self.display_dialog = DisplayDialog(self.canvas, self)
        self.display_dialog.show()

    def _toggle_fill(self, checked: bool) -> None:
        self.canvas.fill = checked
        self.canvas.update()

    def _add_video_buttons(self) -> None:
        """Propagate and Track buttons right after the image arrows (the menu is easy to miss)."""
        actions = self.toolbar.actions()
        next_index = next(i for i, a in enumerate(actions) if a.text() == "Next Image")
        before = actions[next_index + 1]
        for action, icon_name, tip in (
            (
                self.video.propagate_action,
                "propagate",
                "AI Propagate: track the selected (or all) objects through the next frames "
                "with EfficientTAM  ·  Ctrl+Shift+Right",
            ),
            (
                self.video.copy_action,
                "copy",
                "Copy the selected (or all) masks unchanged to the next frames  ·  Ctrl+Alt+Right",
            ),
            (
                self.video.track_action,
                "track",
                "Track objects into each next frame when pressing D: AI if installed, else copy "
                "(on / off)  ·  Ctrl+Shift+T",
            ),
        ):
            action.setIcon(icon(icon_name))
            action.setToolTip(tip)
            self.toolbar.insertAction(before, action)

    def _show_panels(self, visible: bool) -> None:
        """Show the side panels (the Flags panel only when there are flags)."""
        for dock in self.docks:
            dock.setVisible(visible and (dock is not self.flags_dock or bool(self.settings.flags)))

    def _remember_label(self, label: str) -> None:
        if label and label not in self.known_labels:
            self.known_labels.append(label)
            self.session.save_project()

    def _refresh_panels(self) -> None:
        self.label_list.show_labels(self.known_labels, self.canvas.shapes, self.active_label)
        self.shape_list.show_shapes(self.canvas.shapes, self.canvas.hidden, self.known_labels)
        self.shape_list.select_rows(self.canvas.selection)
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

    def show_shortcuts(self) -> None:
        """Show every keyboard shortcut in a window that can stay open."""
        if self.shortcuts_dialog is None:
            self.shortcuts_dialog = ShortcutsDialog(self)
        self.shortcuts_dialog.show()
        self.shortcuts_dialog.raise_()

    def _about(self) -> None:
        QMessageBox.about(
            self,
            "About easy-mask",
            f"easy-mask {__version__}\nLocal-first image annotation for computer-vision masks.",
        )

    def closeEvent(self, event: QCloseEvent) -> None:
        """Offer to save before quitting."""
        if self.maybe_save():
            self.video.stop(wait=True)
            if self.export_on_close_action.isChecked() and self.last_export is not None:
                self._export(*self.last_export, wait=True)
            self.session.save()
            event.accept()
        else:
            event.ignore()
