"""Remember things between runs so the user picks up where they left off.

* App-wide (Qt settings, per user): window size and panel layout, brush / tolerance sizes,
  recent folders (File > Open Recent).
* Per folder (``.easy-mask.json``, see :mod:`easy_mask.io.project`): labels, flags,
  active label and annotation folder.

Command-line options always win over remembered values.
"""

import contextlib
import os
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QMenu

from easy_mask.io import project as project_io

if TYPE_CHECKING:  # the main window imports this module
    from easy_mask.gui.main_window import MainWindow

MAX_RECENT = 8
TOOL_SIZE_ATTRS = ("radius", "tolerance", "size")  # numeric knobs pixel tools expose


def app_settings() -> QSettings:
    """The per-user settings store: an INI file in the user's config folder.

    ``EASY_MASK_CONFIG_DIR`` points it elsewhere (the test suite uses a temp folder).
    """
    override = os.environ.get("EASY_MASK_CONFIG_DIR")
    if override:
        return QSettings(str(Path(override) / "easy-mask.ini"), QSettings.Format.IniFormat)
    return QSettings(
        QSettings.Format.IniFormat, QSettings.Scope.UserScope, "easy-mask", "easy-mask"
    )


class Session:
    """Loads and saves remembered state for ``window``."""

    def __init__(self, window: "MainWindow"):
        self.window = window
        self.store = app_settings()
        self.folder: Path | None = None  # image folder whose project file is in use
        self.recent_menu = QMenu("Open Recent", window)
        self.recent_menu.aboutToShow.connect(self._fill_recent_menu)

    # ------------------------------------------------------------------ app-wide

    def restore(self) -> None:
        """Restore window layout and tool sizes (call once the window is built)."""
        w = self.window
        geometry = self.store.value("window/geometry")
        if geometry is not None:
            w.restoreGeometry(geometry)
        state = self.store.value("window/state")
        if state is not None:
            w.restoreState(state)
        cleanup = self.store.value("auto_cleanup")
        if cleanup is not None:
            w.auto_cleanup = str(cleanup).lower() == "true"
            w.cleanup_action.setChecked(w.auto_cleanup)
        for name, tool in w.pixel_tools.items():
            for attr in TOOL_SIZE_ATTRS:
                value = self.store.value(f"tools/{name}/{attr}")
                if value is not None and hasattr(tool, attr):
                    setattr(tool, attr, int(value))

    def save(self) -> None:
        """Remember window layout, tool sizes and the current folder's project."""
        w = self.window
        self.store.setValue("window/geometry", w.saveGeometry())
        self.store.setValue("window/state", w.saveState())
        self.store.setValue("auto_cleanup", w.auto_cleanup)
        for name, tool in w.pixel_tools.items():
            for attr in TOOL_SIZE_ATTRS:
                if hasattr(tool, attr):
                    self.store.setValue(f"tools/{name}/{attr}", int(getattr(tool, attr)))
        self.store.sync()
        self.save_project()

    def recent(self) -> list[Path]:
        """Recently opened paths that still exist, newest first."""
        stored = self.store.value("recent", [])
        if isinstance(stored, str):  # INI stores a one-item list as a plain string
            paths = [stored]
        elif isinstance(stored, list):
            paths = stored
        else:
            paths = []
        return [Path(p) for p in paths if Path(p).exists()]

    def add_recent(self, path: Path) -> None:
        """Put ``path`` at the top of the recent list."""
        resolved = path.resolve()
        paths = [str(resolved)] + [str(p) for p in self.recent() if p.resolve() != resolved]
        self.store.setValue("recent", paths[:MAX_RECENT])

    def _fill_recent_menu(self) -> None:
        self.recent_menu.clear()
        paths = self.recent()
        if not paths:
            self.recent_menu.addAction("(none)").setEnabled(False)
        for path in paths:
            action = self.recent_menu.addAction(str(path))
            action.triggered.connect(lambda _=False, p=path: self.window.open_path(p))

    # ------------------------------------------------------------------ per folder

    def open_folder(self, folder: Path) -> None:
        """Apply ``folder``'s project file (if any) before its first image loads."""
        self.save_project()  # the folder we are leaving
        self.folder = folder
        project = project_io.load(folder)
        if project is None:
            return
        w = self.window
        for label in project.labels:
            w._remember_label(label)
        for flag in project.flags:
            if flag not in w.settings.flags:
                w.settings.flags.append(flag)
        if w.settings.output_dir is None:
            w.settings.output_dir = project_io.resolve_dir(project.output_dir, folder)
        if w.active_label is None and project.active_label in w.known_labels:
            w.set_active_label(project.active_label)

    def save_project(self) -> None:
        """Write the current folder's project file (no-op until a folder is open)."""
        if self.folder is None or not self.folder.is_dir():
            return
        w = self.window
        project = project_io.Project(
            labels=list(w.known_labels),
            flags=list(w.settings.flags),
            active_label=w.active_label,
            output_dir=project_io.relative_dir(w.settings.output_dir, self.folder),
        )
        # A read-only folder just isn't remembered: a convenience, never an error.
        with contextlib.suppress(OSError):
            project_io.save(project, self.folder)
