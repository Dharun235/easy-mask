"""View > Adjust Display: brightness, contrast, gamma, CLAHE, channel and mask opacity.

Non-modal; every change is shown immediately. Only the canvas drawing changes; saved
annotations, exports and pixel tools always see the original image.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QSlider,
    QWidget,
)

from easy_mask.core import pixel_ops
from easy_mask.core.display import DisplaySettings
from easy_mask.gui.canvas import Canvas


def _slider(minimum: int, maximum: int, value: int) -> QSlider:
    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setRange(minimum, maximum)
    slider.setValue(value)
    return slider


class DisplayDialog(QDialog):
    """Sliders bound to :class:`DisplaySettings` of one canvas."""

    def __init__(self, canvas: Canvas, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Adjust Display")
        self.canvas = canvas
        self.brightness = _slider(-150, 150, 0)
        self.contrast = _slider(10, 400, 100)  # percent
        self.gamma = _slider(10, 400, 100)  # x100
        self.clahe = QCheckBox("Local contrast (CLAHE)")
        self.view_channel = QComboBox()
        self.view_channel.addItems(["rgb", *pixel_ops.CHANNELS])
        self.opacity = _slider(0, 100, 100)  # percent

        form = QFormLayout(self)
        form.addRow("Brightness", self.brightness)
        form.addRow("Contrast", self.contrast)
        form.addRow("Gamma", self.gamma)
        form.addRow("", self.clahe)
        form.addRow("Channel", self.view_channel)
        form.addRow("Mask opacity", self.opacity)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Reset | QDialogButtonBox.StandardButton.Close
        )
        reset = buttons.button(QDialogButtonBox.StandardButton.Reset)
        reset.clicked.connect(self.reset)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

        for slider in (self.brightness, self.contrast, self.gamma, self.opacity):
            slider.valueChanged.connect(self.apply)
        self.clahe.toggled.connect(self.apply)
        self.view_channel.currentIndexChanged.connect(self.apply)

    def settings(self) -> DisplaySettings:
        """The settings the controls currently show."""
        return DisplaySettings(
            brightness=self.brightness.value(),
            contrast=self.contrast.value() / 100,
            gamma=self.gamma.value() / 100,
            clahe=self.clahe.isChecked(),
            view_channel=self.view_channel.currentText(),  # type: ignore[arg-type]
            mask_opacity=self.opacity.value() / 100,
        )

    def apply(self) -> None:
        """Push the controls' settings to the canvas."""
        self.canvas.set_display(self.settings())

    def reset(self) -> None:
        """Back to the unadjusted image."""
        for slider, value in (
            (self.brightness, 0),
            (self.contrast, 100),
            (self.gamma, 100),
            (self.opacity, 100),
        ):
            slider.setValue(value)
        self.clahe.setChecked(False)
        self.view_channel.setCurrentIndex(0)
        self.apply()
