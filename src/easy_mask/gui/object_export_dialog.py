"""Choose what the Objects export writes: binary masks, cutouts and / or coordinates."""

from PySide6.QtWidgets import QCheckBox, QDialog, QDialogButtonBox, QLabel, QVBoxLayout, QWidget

CHOICES = [
    ("masks", "Binary masks", "one PNG per object, white object on black (full image size)"),
    ("cutouts", "Cutouts", "the object's original pixels, cropped, transparent around it"),
    ("coordinates", "Coordinates", "objects.json and objects.csv: box, area, outline points"),
]


class ObjectExportDialog(QDialog):
    """Three checkboxes; Export is disabled until at least one is ticked."""

    def __init__(self, current: list[str], parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Export Objects")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Save each annotated object as:"))
        self.boxes: dict[str, QCheckBox] = {}
        for key, title, detail in CHOICES:
            box = QCheckBox(f"{title}  —  {detail}")
            box.setChecked(key in current)
            box.toggled.connect(self._refresh)
            self.boxes[key] = box
            layout.addWidget(box)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Choose folder...")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self._refresh()

    def parts(self) -> list[str]:
        """The ticked parts, in a fixed order."""
        return [key for key, box in self.boxes.items() if box.isChecked()]

    def _refresh(self) -> None:
        ok = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        ok.setEnabled(bool(self.parts()))


def ask_object_parts(current: list[str], parent: QWidget | None = None) -> list[str] | None:
    """Show the dialog modally; the ticked parts, or None if cancelled."""
    dialog = ObjectExportDialog(current, parent)
    if dialog.exec() == QDialog.DialogCode.Accepted:
        return dialog.parts()
    return None
