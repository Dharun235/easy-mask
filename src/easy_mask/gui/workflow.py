"""Workflow helpers for getting through a dataset quickly: review, copy previous, statistics.

They live in a ``Workflow`` menu attached to the :class:`MainWindow` and use only its
editing primitives, so every change they make is undoable like any other edit.

Review mode needs no separate state: the selected shape *is* the review cursor.
``N`` selects and zooms to the next object; stepping past the last object marks the image
``reviewed`` (an image flag), saves it and moves to the next image. ``Shift+N`` goes back.
"""

from typing import TYPE_CHECKING

from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QPlainTextEdit, QVBoxLayout, QWidget

from easy_mask.core import annotation as ann_io
from easy_mask.core.annotation import REVIEWED_FLAG
from easy_mask.core.video import is_auto
from easy_mask.io import stats

if TYPE_CHECKING:
    from easy_mask.gui.main_window import MainWindow


def show_report(parent: QWidget, title: str, text: str) -> None:
    """Show plain text (a table) in a read-only, monospaced, modal dialog."""
    dialog = QDialog(parent)
    dialog.setWindowTitle(title)
    view = QPlainTextEdit(text)
    view.setReadOnly(True)
    view.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
    view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
    buttons.rejected.connect(dialog.reject)
    layout = QVBoxLayout(dialog)
    layout.addWidget(view)
    layout.addWidget(buttons)
    metrics = view.fontMetrics()
    dialog.resize(metrics.horizontalAdvance("x" * 70), metrics.height() * 24)
    dialog.exec()


class Workflow:
    """Review mode, copy-from-previous-image and folder statistics for one window."""

    def __init__(self, window: "MainWindow"):
        self.window = window
        self.show_report = show_report  # swappable so tests never open a modal dialog
        bar = window.menuBar()
        menu = bar.addMenu("&Workflow")
        bar.insertMenu(bar.actions()[-2], menu)  # before Help
        window._action(menu, "Review Next Object", self.next_object, "N")
        window._action(menu, "Review Previous Object", self.prev_object, "Shift+N")
        menu.addSeparator()
        window._action(menu, "Copy Shapes from Previous Image", self.copy_previous, "Ctrl+Shift+V")
        menu.addSeparator()
        window._action(menu, "Statistics...", self.show_statistics, "Ctrl+I")

    # ------------------------------------------------------------------ review

    def next_object(self) -> None:
        """Focus the next object; past the last one, mark reviewed and go to the next image."""
        win = self.window
        if win.current_index is None:
            return
        order = self.review_order()
        position = order.index(win.canvas.selected) if win.canvas.selected in order else -1
        if position + 1 < len(order):
            self.focus(order[position + 1])
            return
        self.mark_reviewed()
        if not win.save():
            return
        if win.current_index + 1 >= len(win.image_paths):
            win.statusBar().showMessage("Last image reviewed", 5000)
            return
        win.next_image()
        self._focus_first_or_announce(first=True)

    def prev_object(self) -> None:
        """Focus the previous object; before the first one, go to the previous image's last."""
        win = self.window
        if win.current_index is None:
            return
        order = self.review_order()
        position = order.index(win.canvas.selected) if win.canvas.selected in order else len(order)
        if position - 1 >= 0:
            self.focus(order[position - 1])
            return
        before = win.current_index
        win.prev_image()
        if win.current_index != before:
            self._focus_first_or_announce(first=False)

    def review_order(self) -> list[int]:
        """Shape indices in review order: automatic shapes first, then hand-drawn ones.

        Propagated / interpolated shapes were made by a model or an interpolation, so they
        are the ones most worth checking.
        """
        shapes = self.window.canvas.shapes
        auto = [i for i, s in enumerate(shapes) if is_auto(s)]
        return auto + [i for i in range(len(shapes)) if i not in auto]

    def _focus_first_or_announce(self, first: bool) -> None:
        order = self.review_order()
        if order:
            self.focus(order[0] if first else order[-1])
        else:
            self.window.statusBar().showMessage("No objects on this image: N to move on", 5000)

    def focus(self, index: int) -> None:
        """Select shape ``index``, zoom to it and say where we are in the review."""
        win = self.window
        shape = win.canvas.shapes[index]
        win.canvas.select(index)
        win.canvas.zoom_to_box(shape.bounding_box())
        image = (win.current_index or 0) + 1
        step = self.review_order().index(index) + 1
        made_by = "  (auto: check it)" if is_auto(shape) else ""
        win.statusBar().showMessage(
            f"Object {step}/{len(win.canvas.shapes)}: {shape.label}{made_by}   "
            f"(image {image}/{len(win.image_paths)})"
        )

    def mark_reviewed(self) -> None:
        """Set the current image's ``reviewed`` flag (unsaved until the next save)."""
        win = self.window
        flags = win.flag_list.flags()
        if flags.get(REVIEWED_FLAG):
            return
        flags[REVIEWED_FLAG] = True
        win.flag_list.show_flags(flags)
        win.flags_dock.setVisible(True)
        win._mark_dirty()

    # ------------------------------------------------------------------ copy previous

    def copy_previous(self) -> None:
        """Append the previous image's saved shapes to this image (one undo step).

        "Previous" is the image before this one in the folder, as with ``A``; handy for
        video frames and near-duplicate images. The copies are selected so they can be
        moved together.
        """
        win = self.window
        if not win.current_index:
            return
        previous = win.image_paths[win.current_index - 1]
        json_path = ann_io.annotation_path_for(previous, win.settings.output_dir)
        if not json_path.exists():
            win.statusBar().showMessage(f"{previous.name} has no annotations", 4000)
            return
        try:
            copied = ann_io.load(json_path).shapes
        except (ValueError, KeyError) as exc:
            win._error(f"Could not read annotation {json_path}:\n{exc}")
            return
        if not copied:
            return
        for shape in copied:
            win._remember_label(shape.label)
        first = len(win.canvas.shapes)
        win._replace_shapes([*win.canvas.shapes, *copied])
        win.canvas.set_selection(list(range(first, first + len(copied))))

    # ------------------------------------------------------------------ statistics

    def show_statistics(self) -> None:
        """Show folder statistics and, with a label list, the label validation result."""
        win = self.window
        if win.image_path is None:
            win._error("Open a folder first.")
            return
        allowed = list(win.settings.labels) or None
        try:
            result = stats.compute(win.image_path.parent, win.settings.output_dir, allowed)
        except (ValueError, KeyError) as exc:
            win._error(f"Could not read every annotation:\n{exc}")
            return
        text = stats.format_table(result, validated=allowed is not None)
        if win.dirty:
            text = "Unsaved changes on the current image are not included.\n\n" + text
        self.show_report(win, "Statistics", text)
