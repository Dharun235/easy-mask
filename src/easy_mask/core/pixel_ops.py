"""Classic, model-free segmentation of an RGB image into a full-size boolean mask.

Every function takes an ``(height, width, 3)`` uint8 RGB image and returns a
``(height, width)`` bool mask, the working format of :mod:`easy_mask.core.masks`.
Points are image pixels ``(x, y)``; boxes are ``(x0, y0, x1, y1)`` with exclusive ends.
"""

from typing import Literal

import cv2
import numpy as np

from easy_mask.core.geometry import Point

Channel = Literal["gray", "red", "green", "blue", "hue", "saturation", "value"]
CHANNELS: tuple[Channel, ...] = ("gray", "red", "green", "blue", "hue", "saturation", "value")
ThresholdMethod = Literal["manual", "otsu", "adaptive"]
Box = tuple[int, int, int, int]


def channel(rgb: np.ndarray, name: Channel) -> np.ndarray:
    """One ``uint8`` plane of the image: gray, R, G, B or H, S, V (hue is 0-179)."""
    if name == "gray":
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if name in ("red", "green", "blue"):
        return np.ascontiguousarray(rgb[..., ("red", "green", "blue").index(name)])
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return np.ascontiguousarray(hsv[..., ("hue", "saturation", "value").index(name)])


def _pixel(rgb: np.ndarray, p: Point) -> tuple[int, int]:
    """``(x, y)`` as integer indices clamped to the image."""
    height, width = rgb.shape[:2]
    return min(max(int(p[0]), 0), width - 1), min(max(int(p[1]), 0), height - 1)


# ---------------------------------------------------------------------- threshold


def threshold(
    rgb: np.ndarray,
    method: ThresholdMethod = "otsu",
    level: int = 128,
    plane: Channel = "gray",
    invert: bool = False,
    region: np.ndarray | None = None,
    block_size: int = 31,
    offset: float = 5.0,
) -> np.ndarray:
    """Pixels brighter than a threshold on one channel.

    Args:
        rgb: RGB image.
        method: ``manual`` uses ``level``; ``otsu`` picks the level that best splits the
            histogram (of ``region`` only, if given); ``adaptive`` compares each pixel with
            the mean of its ``block_size`` neighbourhood minus ``offset``.
        level: Threshold for ``manual``: pixels ``> level`` are foreground.
        plane: Which channel to threshold.
        invert: Select the darker side instead.
        region: Optional bool mask; the result is limited to it.
        block_size: Neighbourhood size for ``adaptive`` (made odd, at least 3).
        offset: Constant subtracted from the neighbourhood mean for ``adaptive``.
    """
    values = channel(rgb, plane)
    if method == "adaptive":
        size = max(3, block_size | 1)
        mask = cv2.adaptiveThreshold(
            values, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, size, offset
        ).astype(bool)
    else:
        if method == "otsu":
            sample = values[region] if region is not None else values.ravel()
            if sample.size == 0:
                return np.zeros(values.shape, dtype=bool)
            level = int(otsu_level(sample))
        mask = values > level
    if invert:
        mask = ~mask
    if region is not None:
        mask &= region
    return np.asarray(mask, dtype=bool)


def otsu_level(values: np.ndarray) -> float:
    """Otsu's threshold for any array of ``uint8`` values."""
    level, _ = cv2.threshold(
        values.reshape(-1, 1).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return float(level)


# ---------------------------------------------------------------------- colour


def magic_wand(rgb: np.ndarray, seed: Point, tolerance: int = 32) -> np.ndarray:
    """Flood fill from ``seed``: the connected region within ``tolerance`` of its colour.

    Each RGB channel may differ from the seed pixel by at most ``tolerance`` (0-255);
    neighbours are 4-connected.
    """
    height, width = rgb.shape[:2]
    fill = np.zeros((height + 2, width + 2), np.uint8)
    diff = (tolerance,) * 3
    flags = 4 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (1 << 8)
    cv2.floodFill(np.ascontiguousarray(rgb), fill, _pixel(rgb, seed), (0, 0, 0), diff, diff, flags)
    return fill[1:-1, 1:-1].astype(bool)


def color_at(rgb: np.ndarray, p: Point) -> tuple[int, int, int]:
    """The RGB colour of the pixel at ``p``."""
    x, y = _pixel(rgb, p)
    r, g, b = rgb[y, x]
    return int(r), int(g), int(b)


def color_range(
    rgb: np.ndarray,
    color: tuple[int, int, int],
    hue_tol: int = 10,
    sat_tol: int = 60,
    val_tol: int = 60,
) -> np.ndarray:
    """Every pixel whose HSV is within a band around ``color`` (anywhere in the image).

    Hue wraps around (red at 0 and 179 are neighbours). Hue is OpenCV's 0-179 scale;
    saturation and value are 0-255.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.int16)
    target = cv2.cvtColor(np.array([[color]], np.uint8), cv2.COLOR_RGB2HSV)[0, 0].astype(np.int16)
    hue_diff = np.abs(hsv[..., 0] - target[0])
    hue_diff = np.minimum(hue_diff, 180 - hue_diff)
    return np.asarray(
        (hue_diff <= hue_tol)
        & (np.abs(hsv[..., 1] - target[1]) <= sat_tol)
        & (np.abs(hsv[..., 2] - target[2]) <= val_tol),
        dtype=bool,
    )


# ---------------------------------------------------------------------- graph-based


def grabcut(
    rgb: np.ndarray,
    box: Box,
    foreground: np.ndarray | None = None,
    background: np.ndarray | None = None,
    iterations: int = 5,
) -> np.ndarray:
    """GrabCut: separate the object inside ``box`` from its surroundings.

    Args:
        rgb: RGB image.
        box: ``(x0, y0, x1, y1)``; everything outside is background.
        foreground: Optional bool scribbles that must be object.
        background: Optional bool scribbles that must be background.
        iterations: GrabCut iterations (more is slower and usually a little better).

    Returns:
        The object mask; empty if the box has no area.
    """
    height, width = rgb.shape[:2]
    x0, y0 = max(int(box[0]), 0), max(int(box[1]), 0)
    x1, y1 = min(int(box[2]), width), min(int(box[3]), height)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return np.zeros((height, width), dtype=bool)
    labels = np.full((height, width), cv2.GC_BGD, np.uint8)
    labels[y0:y1, x0:x1] = cv2.GC_PR_FGD
    if not (labels == cv2.GC_BGD).any():  # box covers the image: borders are likely bg
        labels[[0, -1], :] = cv2.GC_PR_BGD
        labels[:, [0, -1]] = cv2.GC_PR_BGD
    if background is not None:
        labels[background] = cv2.GC_BGD
    if foreground is not None:
        labels[foreground] = cv2.GC_FGD
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(
        np.ascontiguousarray(rgb[..., ::-1]),
        labels,
        (0, 0, 0, 0),  # unused with GC_INIT_WITH_MASK
        bgd_model,
        fgd_model,
        iterations,
        cv2.GC_INIT_WITH_MASK,
    )
    return np.asarray((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD), dtype=bool)
