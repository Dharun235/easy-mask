"""Display-only image adjustments: what the user sees, never what is saved or exported.

Dark or low-contrast images are hard to outline. These settings change only the pixels
drawn on the canvas; annotations, pixel tools and exports always use the original image.
"""

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from easy_mask.core.pixel_ops import Channel, channel

ViewChannel = Literal["rgb"] | Channel


@dataclass(frozen=True)
class DisplaySettings:
    """How to draw the image and masks.

    Attributes:
        brightness: Added to every pixel, -255 to 255.
        contrast: Multiplies the distance from mid-gray; 1 keeps the image as is.
        gamma: Values above 1 brighten the shadows, below 1 darken them.
        clahe: Local histogram equalisation (CLAHE) to bring out detail in flat areas.
        view_channel: ``rgb`` for colour, or one channel shown as grayscale.
        mask_opacity: Multiplier for mask and shape fill opacity, 0 to 1.
    """

    brightness: int = 0
    contrast: float = 1.0
    gamma: float = 1.0
    clahe: bool = False
    view_channel: ViewChannel = "rgb"
    mask_opacity: float = 1.0

    @property
    def changes_image(self) -> bool:
        """True if :func:`adjust` would change any pixel."""
        return (
            self.brightness != 0
            or self.contrast != 1.0
            or self.gamma != 1.0
            or self.clahe
            or self.view_channel != "rgb"
        )


def adjust(rgb: np.ndarray, settings: DisplaySettings) -> np.ndarray:
    """Apply channel view, CLAHE, contrast/brightness and gamma, in that order."""
    if settings.view_channel == "rgb":
        out = rgb
    else:
        out = cv2.cvtColor(channel(rgb, settings.view_channel), cv2.COLOR_GRAY2RGB)
    if settings.clahe:
        out = _clahe(out)
    lut = np.arange(256, dtype=np.float64)
    lut = (lut - 128.0) * settings.contrast + 128.0 + settings.brightness
    lut = np.clip(lut, 0, 255) / 255.0
    lut = 255.0 * lut ** (1.0 / max(settings.gamma, 1e-3))
    return cv2.LUT(np.ascontiguousarray(out), np.round(lut).astype(np.uint8))


def _clahe(rgb: np.ndarray) -> np.ndarray:
    """Equalise lightness only, so colours do not shift."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
