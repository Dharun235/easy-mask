"""The Export window: format, what to save, which frames, and where, in one place.

Every Export opens it with the last choices filled in, so repeating an export is one
click on Export, and changing any part of it needs no menus.
"""

from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from easy_mask.io.export import EXPORT_NAMES, PARTS

PART_NAMES = {
    "masks": "Binary masks  (one black-and-white PNG per object)",
    "cutouts": "Cutouts  (the object's own pixels, transparent around it)",
    "coordinates": "Coordinates  (box, area and outline points)",
    "centroid": "Centroid  (the centre point of each mask)",
    "corners": "Corners  (the 4 corners of each mask's box, straight and rotated)",
    "points": "Points  (each mask's key points as placed or moved by hand)",
    "overlays": "Frames with overlays  (each frame with its masks and points drawn on)",
}


@dataclass
class ExportChoice:
    """Everything an export needs. Frames are 0-based and inclusive."""

    fmt: str
    out: Path
    parts: list[str] = field(default_factory=lambda: list(PARTS))
    first: int = 0
    last: int = 0
    on_close: bool = False


class ExportDialog(QDialog):
    """Pick format, parts (for Objects), frames and folder; Export stays off until valid."""

    def __init__(
        self, choice: ExportChoice, frame_count: int, current: int, parent: QWidget | None = None
    ):
        super().__init__(parent)
        self.setWindowTitle("Export")
        self.frame_count, self.current = frame_count, current

        self.format_box = QComboBox()
        for key, name in EXPORT_NAMES.items():
            self.format_box.addItem(name, key)
        self.format_box.setCurrentIndex(max(list(EXPORT_NAMES).index(choice.fmt), 0))

        self.parts_box = QWidget()
        parts_layout = QVBoxLayout(self.parts_box)
        parts_layout.setContentsMargins(0, 0, 0, 0)
        self.part_checks: dict[str, QCheckBox] = {}
        for key in PARTS:
            check = QCheckBox(PART_NAMES[key])
            check.setChecked(key in choice.parts)
            check.toggled.connect(self._refresh)
            self.part_checks[key] = check
            parts_layout.addWidget(check)

        self.all_frames = QRadioButton(f"All frames ({frame_count})")
        self.this_frame = QRadioButton(f"Only this frame ({current + 1})")
        self.range_frames = QRadioButton("Frames")
        self.first_spin, self.last_spin = QSpinBox(), QSpinBox()
        for spin, value in ((self.first_spin, choice.first), (self.last_spin, choice.last)):
            spin.setRange(1, frame_count)
            spin.setValue(min(value, frame_count - 1) + 1)
            spin.valueChanged.connect(self._use_range)
        group = QButtonGroup(self)
        for button in (self.all_frames, self.this_frame, self.range_frames):
            group.addButton(button)
        whole = choice.first == 0 and choice.last >= frame_count - 1
        (self.all_frames if whole else self.range_frames).setChecked(True)
        frames_row = QHBoxLayout()
        frames_row.addWidget(self.range_frames)
        frames_row.addWidget(self.first_spin)
        frames_row.addWidget(QLabel("to"))
        frames_row.addWidget(self.last_spin)
        frames_row.addStretch(1)
        frames = QVBoxLayout()
        for widget in (self.all_frames, self.this_frame):
            frames.addWidget(widget)
        frames.addLayout(frames_row)

        self.folder_edit = QLineEdit(str(choice.out))
        self.folder_edit.textChanged.connect(self._refresh)
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)
        folder_row = QHBoxLayout()
        folder_row.addWidget(self.folder_edit)
        folder_row.addWidget(browse)

        self.on_close = QCheckBox("Export again automatically when closing")
        self.on_close.setChecked(choice.on_close)

        form = QFormLayout()
        form.addRow("Format", self.format_box)
        form.addRow("Save", self.parts_box)
        form.addRow("Frames", frames)
        form.addRow("Folder", folder_row)
        self.form = form
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Export")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.on_close)
        layout.addWidget(self.buttons)
        self.format_box.currentIndexChanged.connect(self._refresh)
        for button in (self.all_frames, self.this_frame, self.range_frames):
            button.toggled.connect(self._refresh)
        self._refresh()

    def choice(self) -> ExportChoice:
        """The choices currently in the window."""
        if self.this_frame.isChecked():
            first = last = self.current
        elif self.range_frames.isChecked():
            first, last = sorted((self.first_spin.value() - 1, self.last_spin.value() - 1))
        else:
            first, last = 0, self.frame_count - 1
        return ExportChoice(
            fmt=self.format_box.currentData(),
            out=Path(self.folder_edit.text().strip()).expanduser(),
            parts=[key for key, check in self.part_checks.items() if check.isChecked()],
            first=first,
            last=last,
            on_close=self.on_close.isChecked(),
        )

    def _use_range(self, _value: int) -> None:
        self.range_frames.setChecked(True)  # typing a frame number means "these frames"

    def _refresh(self) -> None:
        objects = self.format_box.currentData() == "objects"
        self.form.setRowVisible(1, objects)  # the parts only apply to Objects
        choice = self.choice()
        ok = bool(self.folder_edit.text().strip()) and (bool(choice.parts) or not objects)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(ok)
        self.adjustSize()

    def _browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Export to", self.folder_edit.text())
        if folder:
            self.folder_edit.setText(folder)


def ask_export(
    choice: ExportChoice, frame_count: int, current: int, parent: QWidget | None = None
) -> ExportChoice | None:
    """Show the Export window modally; the choices, or None if cancelled."""
    dialog = ExportDialog(choice, frame_count, current, parent)
    try:
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return dialog.choice()
        return None
    finally:
        dialog.deleteLater()  # freed by Qt, never by Python's garbage collector at exit
