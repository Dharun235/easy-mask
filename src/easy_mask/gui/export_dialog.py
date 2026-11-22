"""The Export window: what to save, which objects and frames, and where, in one place.

Every Export opens it with the last choices filled in, so repeating an export is one
click. Training-dataset formats (COCO, YOLO, ...) sit in one dropdown at the bottom.
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

OBJECTS = "objects"
PART_NAMES = {
    "masks": "Binary masks  (black and white)",
    "cutouts": "Original pixels  (only the object, the rest transparent)",
    "overlays": "Overlays  (the frame with masks and points drawn on)",
    "corners": "Box corners and centroid  (CSV table)",
    "points": "Points  (as placed or moved by hand, CSV table)",
}
NO_DATASET = "None"


@dataclass
class ExportChoice:
    """Everything an export needs. Frames are 0-based and inclusive.

    ``fmt`` is ``"objects"`` (the parts below) or a training-dataset format.
    """

    fmt: str
    out: Path
    parts: list[str] = field(default_factory=lambda: list(PARTS))
    first: int = 0
    last: int = 0
    on_close: bool = False
    combine: bool = False  # all objects of a frame in one image
    video: bool = False  # MP4 instead of PNG frames
    only_ticked: bool = False  # only the labels ticked in the Labels panel


class ExportDialog(QDialog):
    """Pick parts, objects, frames and folder; Export stays off until valid.

    Args:
        ticked: Labels ticked in the Labels panel (empty: the option is off).
    """

    def __init__(
        self,
        choice: ExportChoice,
        frame_count: int,
        current: int,
        parent: QWidget | None = None,
        ticked: list[str] | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Export")
        self.frame_count, self.current = frame_count, current

        self.parts_box = QWidget()
        parts_layout = QVBoxLayout(self.parts_box)
        parts_layout.setContentsMargins(0, 0, 0, 0)
        parts_layout.setSpacing(6)
        self.part_checks: dict[str, QCheckBox] = {}
        for key in PARTS:
            check = QCheckBox(PART_NAMES[key])
            check.setChecked(key in choice.parts)
            check.toggled.connect(self._refresh)
            self.part_checks[key] = check
            parts_layout.addWidget(check)

        ticked = ticked or []
        names = " + ".join(f"“{t}”" for t in ticked[:3]) + (" ..." if len(ticked) > 3 else "")
        self.only_ticked = QCheckBox(
            f"Only ticked labels: {names}" if ticked else "Only ticked labels (none ticked)"
        )
        self.only_ticked.setEnabled(bool(ticked))
        self.only_ticked.setChecked(bool(ticked) and choice.only_ticked)
        self.combine = QCheckBox("All objects together in one image per frame")
        self.combine.setChecked(choice.combine)
        objects = QVBoxLayout()
        objects.setSpacing(6)
        objects.addWidget(self.only_ticked)
        objects.addWidget(self.combine)

        self.png = QRadioButton("PNG frames")
        self.mp4 = QRadioButton("MP4 video")
        (self.mp4 if choice.video else self.png).setChecked(True)
        kind = QButtonGroup(self)
        kind.addButton(self.png)
        kind.addButton(self.mp4)
        kind_row = QHBoxLayout()
        kind_row.setSpacing(20)
        kind_row.addWidget(self.png)
        kind_row.addWidget(self.mp4)
        kind_row.addStretch(1)

        self.all_frames = QRadioButton(f"All frames ({frame_count})")
        self.this_frame = QRadioButton(f"Only this frame ({current + 1})")
        self.range_frames = QRadioButton("From")
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
        frames_row.setSpacing(8)
        frames_row.addWidget(self.range_frames)
        frames_row.addWidget(self.first_spin)
        frames_row.addWidget(QLabel("to"))
        frames_row.addWidget(self.last_spin)
        frames_row.addStretch(1)
        frames = QVBoxLayout()
        frames.setSpacing(6)
        for widget in (self.all_frames, self.this_frame):
            frames.addWidget(widget)
        frames.addLayout(frames_row)

        self.folder_edit = QLineEdit(str(choice.out))
        self.folder_edit.textChanged.connect(self._refresh)
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)
        folder_row = QHBoxLayout()
        folder_row.setSpacing(8)
        folder_row.addWidget(self.folder_edit)
        folder_row.addWidget(browse)

        self.dataset_box = QComboBox()
        self.dataset_box.addItem(NO_DATASET, OBJECTS)
        for key, name in EXPORT_NAMES.items():
            if key != OBJECTS:
                self.dataset_box.addItem(name, key)
        self.dataset_box.setCurrentIndex(max(self.dataset_box.findData(choice.fmt), 0))
        self.dataset_box.setToolTip("For training a model: export a standard dataset instead")

        self.on_close = QCheckBox("Export again automatically when closing")
        self.on_close.setChecked(choice.on_close)

        form = QFormLayout()
        form.setVerticalSpacing(14)
        form.setHorizontalSpacing(18)
        form.addRow("Save", self.parts_box)
        form.addRow("Objects", objects)
        form.addRow("Images as", kind_row)
        form.addRow("Frames", frames)
        form.addRow("Folder", folder_row)
        form.addRow("Training dataset", self.dataset_box)
        self.form = form
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Export")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 18)
        layout.setSpacing(14)
        layout.addLayout(form)
        layout.addWidget(self.on_close)
        layout.addWidget(self.buttons)
        self.dataset_box.currentIndexChanged.connect(self._refresh)
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
            fmt=self.dataset_box.currentData(),
            out=Path(self.folder_edit.text().strip()).expanduser(),
            parts=[key for key, check in self.part_checks.items() if check.isChecked()],
            first=first,
            last=last,
            on_close=self.on_close.isChecked(),
            combine=self.combine.isChecked(),
            video=self.mp4.isChecked(),
            only_ticked=self.only_ticked.isChecked(),
        )

    def _use_range(self, _value: int) -> None:
        self.range_frames.setChecked(True)  # typing a frame number means "these frames"

    def _refresh(self) -> None:
        objects = self.dataset_box.currentData() == OBJECTS
        for row in range(3):  # Save, Objects, Images as: only for our own export
            self.form.setRowVisible(row, objects)
        choice = self.choice()
        ok = bool(self.folder_edit.text().strip()) and (bool(choice.parts) or not objects)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(ok)
        self.adjustSize()

    def _browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Export to", self.folder_edit.text())
        if folder:
            self.folder_edit.setText(folder)


def ask_export(
    choice: ExportChoice,
    frame_count: int,
    current: int,
    parent: QWidget | None = None,
    ticked: list[str] | None = None,
) -> ExportChoice | None:
    """Show the Export window modally; the choices, or None if cancelled."""
    dialog = ExportDialog(choice, frame_count, current, parent, ticked)
    try:
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return dialog.choice()
        return None
    finally:
        dialog.deleteLater()  # freed by Qt, never by Python's garbage collector at exit
