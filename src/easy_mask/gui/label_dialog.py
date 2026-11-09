"""Dialog asking for a shape's label, group id and description."""

from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtGui import QIntValidator
from PySide6.QtWidgets import (
    QCompleter,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from easy_mask.core.video import ERASE_LABEL
from easy_mask.gui.widgets import color_icon


@dataclass
class LabelInfo:
    """What the user entered in the label dialog."""

    label: str
    group_id: int | None = None
    description: str = ""


class LabelDialog(QDialog):
    """Pick or type a label. Known labels autocomplete and are listed for one-click use.

    With ``allowed`` (``--strict-labels``) only those labels are offered and accepted.
    """

    def __init__(
        self,
        known_labels: list[str],
        initial: LabelInfo,
        parent: QWidget | None = None,
        allowed: list[str] | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Label")
        self.allowed = allowed
        if allowed is not None:
            known_labels = allowed

        self.label_edit = QLineEdit(initial.label)
        self.label_edit.setPlaceholderText("label, e.g. cat")
        completer = QCompleter(known_labels, self)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self.label_edit.setCompleter(completer)

        self.group_edit = QLineEdit("" if initial.group_id is None else str(initial.group_id))
        self.group_edit.setValidator(QIntValidator(0, 1_000_000, self))
        self.group_edit.setPlaceholderText("optional: same id = same object")
        self.description_edit = QLineEdit(initial.description)
        self.description_edit.setPlaceholderText("optional note")

        self.label_list = QListWidget()
        for label in known_labels:
            self.label_list.addItem(QListWidgetItem(color_icon(label, known_labels), label))
        self.label_list.itemClicked.connect(lambda item: self.label_edit.setText(item.text()))
        self.label_list.itemDoubleClicked.connect(lambda item: self._accept_label(item.text()))

        self.error_label = QLabel()
        self.error_label.setStyleSheet("color: #d33")
        self.error_label.hide()

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        erase = buttons.addButton("Erase With It", QDialogButtonBox.ButtonRole.ActionRole)
        erase.setToolTip("Don't keep the shape: cut it out of the masks underneath")
        erase.clicked.connect(self._accept_erase)

        form = QFormLayout()
        form.addRow("Label", self.label_edit)
        form.addRow("Group id", self.group_edit)
        form.addRow("Description", self.description_edit)
        self.form = form
        # Group id and description are rarely needed: tucked away until asked for.
        self.more = QToolButton()
        self.more.setText("More options")
        self.more.setCheckable(True)
        self.more.setAutoRaise(True)
        self.more.toggled.connect(self._show_more)
        self._show_more(False)
        self.more.setChecked(initial.group_id is not None or bool(initial.description))
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.more)
        layout.addWidget(self.label_list)
        layout.addWidget(self.error_label)
        layout.addWidget(buttons)

        self.label_edit.selectAll()
        self.label_edit.setFocus()

    def _show_more(self, shown: bool) -> None:
        self.form.setRowVisible(1, shown)
        self.form.setRowVisible(2, shown)
        self.more.setText("Fewer options" if shown else "More options")

    def result_info(self) -> LabelInfo:
        """The values currently entered."""
        group = self.group_edit.text().strip()
        return LabelInfo(
            label=self.label_edit.text().strip(),
            group_id=int(group) if group else None,
            description=self.description_edit.text().strip(),
        )

    def _accept_erase(self) -> None:
        self.label_edit.setText(ERASE_LABEL)
        self.accept()

    def _accept_label(self, label: str) -> None:
        self.label_edit.setText(label)
        self._accept_if_valid()

    def _accept_if_valid(self) -> None:
        label = self.label_edit.text().strip()
        if not label:
            return
        if self.allowed is not None and label not in self.allowed:
            self.error_label.setText(f"'{label}' is not in the label list")
            self.error_label.show()
            return
        self.accept()

    @staticmethod
    def ask(
        known_labels: list[str],
        initial: LabelInfo,
        parent: QWidget | None = None,
        allowed: list[str] | None = None,
    ) -> LabelInfo | None:
        """Show the dialog modally. Returns the entered info, or None if cancelled."""
        dialog = LabelDialog(known_labels, initial, parent, allowed)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return dialog.result_info()
        return None
