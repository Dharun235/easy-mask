"""Threshold / colour-range panel with a live preview on the canvas.

The panel is a non-modal dialog. While it is open, the canvas shows the would-be mask and
clicking the image picks the colour for *Colour range*. *Add mask* turns the preview into
a new mask shape (its label is asked as usual); the panel stays open for the next one.
"""

from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QInputEvent, QPainter
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QSlider,
    QSpinBox,
    QWidget,
)

from easy_mask.core import pixel_ops
from easy_mask.core.geometry import Point
from easy_mask.core.rasterize import shape_to_mask
from easy_mask.gui.tools.base import CanvasTool, mask_image
from easy_mask.gui.tools.brush import StrokeCallback

if TYPE_CHECKING:
    from easy_mask.gui.canvas import Canvas

METHODS = ["Manual", "Otsu", "Adaptive", "Colour range"]
PREVIEW_RGB = (255, 0, 200)


class PreviewTool(CanvasTool):
    """Canvas side of the panel: draws the preview, clicks pick the colour."""

    def __init__(self, panel: "ThresholdDialog"):
        self.panel = panel

    def press(self, canvas: "Canvas", p: Point, event: QInputEvent) -> None:
        """Pick the clicked pixel's colour for the colour-range mode."""
        self.panel.set_color(pixel_ops.color_at(canvas.image_rgb(), p))

    def paint(self, canvas: "Canvas", painter: QPainter) -> None:
        """Overlay the current result."""
        painter.drawImage(QPointF(0, 0), mask_image(self.panel.preview(), PREVIEW_RGB, 140))


def _spin(minimum: int, maximum: int, value: int) -> QSpinBox:
    spin = QSpinBox()
    spin.setRange(minimum, maximum)
    spin.setValue(value)
    return spin


class ThresholdDialog(QDialog):
    """Controls for :func:`pixel_ops.threshold` and :func:`pixel_ops.color_range`."""

    def __init__(self, canvas: "Canvas", on_result: StrokeCallback, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Threshold / Colour Range")
        self.canvas = canvas
        self.on_result = on_result
        self.tool = PreviewTool(self)
        self.color = (128, 128, 128)
        self._mask: np.ndarray | None = None
        self._mask_for = 0  # id() of the image array the cached mask was computed on

        self.method = QComboBox()
        self.method.addItems(METHODS)
        self.method.setCurrentText("Otsu")
        self.channel = QComboBox()
        self.channel.addItems(list(pixel_ops.CHANNELS))
        self.level = QSlider(Qt.Orientation.Horizontal)
        self.level.setRange(0, 255)
        self.level.setValue(128)
        self.invert = QCheckBox("Select the darker side")
        self.block = _spin(3, 501, 31)
        self.offset = _spin(-100, 100, 5)
        self.color_label = QLabel()
        self.hue_tol = _spin(0, 90, 10)
        self.sat_tol = _spin(0, 255, 60)
        self.val_tol = _spin(0, 255, 60)
        self.inside_selection = QCheckBox("Only inside the selected shape")

        form = QFormLayout(self)
        form.addRow("Method", self.method)
        form.addRow("Channel", self.channel)
        form.addRow("Level (manual)", self.level)
        form.addRow("", self.invert)
        form.addRow("Block size (adaptive)", self.block)
        form.addRow("Offset (adaptive)", self.offset)
        form.addRow("Colour (click image)", self.color_label)
        form.addRow("Hue tolerance", self.hue_tol)
        form.addRow("Saturation tolerance", self.sat_tol)
        form.addRow("Value tolerance", self.val_tol)
        form.addRow("", self.inside_selection)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.addButton("Add mask", QDialogButtonBox.ButtonRole.ApplyRole).clicked.connect(
            self.apply
        )
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

        for combo in (self.method, self.channel):
            combo.currentIndexChanged.connect(self.refresh)
        spins = (self.block, self.offset, self.hue_tol, self.sat_tol, self.val_tol)
        for widget in (self.level, *spins):
            widget.valueChanged.connect(self.refresh)
        for box in (self.invert, self.inside_selection):
            box.toggled.connect(self.refresh)
        self.set_color(self.color)

    def set_color(self, color: tuple[int, int, int]) -> None:
        """Use ``color`` for the colour-range mode."""
        self.color = color
        self.color_label.setText(f"RGB {color[0]}, {color[1]}, {color[2]}")
        self.color_label.setStyleSheet(f"background: rgb{color}; padding: 2px")
        self.refresh()

    def refresh(self) -> None:
        """Recompute the preview after a control changed."""
        self._mask = None
        self.canvas.update()

    def preview(self) -> np.ndarray:
        """The result for the current settings (cached until something changes)."""
        rgb = self.canvas.image_rgb()
        if self._mask is None or self._mask_for != id(rgb):
            self._mask = self._compute(rgb)
            self._mask_for = id(rgb)
        return self._mask

    def _compute(self, rgb: np.ndarray) -> np.ndarray:
        region = None
        index = self.canvas.selected
        if self.inside_selection.isChecked() and index is not None:
            region = shape_to_mask(self.canvas.shapes[index], *rgb.shape[:2])
        method = self.method.currentText()
        if method == "Colour range":
            result = pixel_ops.color_range(
                rgb, self.color, self.hue_tol.value(), self.sat_tol.value(), self.val_tol.value()
            )
            if self.invert.isChecked():
                result = ~result
            return result if region is None else result & region
        return pixel_ops.threshold(
            rgb,
            method=method.lower(),  # type: ignore[arg-type]
            level=self.level.value(),
            plane=self.channel.currentText(),  # type: ignore[arg-type]
            invert=self.invert.isChecked(),
            region=region,
            block_size=self.block.value(),
            offset=self.offset.value(),
        )

    def apply(self) -> None:
        """Turn the preview into a new mask shape."""
        result = self.preview()
        if result.any():
            self.on_result(result.copy(), None)
            self.refresh()  # the selection changed, so a region-limited preview may too
