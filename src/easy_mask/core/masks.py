"""Bitmap masks: encoding, building mask shapes, editing and conversion.

A *mask shape* stores a boolean array cropped to its bounding box, exactly like labelme:
``points = [(x1, y1), (x2, y2)]`` are the inclusive top-left and bottom-right pixels, and
``mask`` has shape ``(y2 - y1 + 1, x2 - x1 + 1)``. In JSON the crop is a base64 PNG
(see :func:`easy_mask.core.shapes.encode_mask`).

Full-size masks (``height x width`` bool arrays) are the working format for every
operation here; :func:`to_shape` crops back down for storage.
"""

from typing import Any

import cv2
import numpy as np

from easy_mask.core.geometry import Point
from easy_mask.core.shapes import Shape, ShapeType

# ---------------------------------------------------------------------- shape <-> full mask


def to_shape(label: str, full: np.ndarray, **fields: Any) -> Shape | None:
    """Build a mask shape from a full-size boolean mask, cropped to its content.

    Args:
        label: Class name for the shape.
        full: ``(height, width)`` boolean mask.
        **fields: Other :class:`Shape` fields (``group_id``, ``description``, ``flags``).

    Returns:
        The shape, or None if the mask is empty.
    """
    ys, xs = np.nonzero(full)
    if len(xs) == 0:
        return None
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    crop = full[y1 : y2 + 1, x1 : x2 + 1].copy()
    return Shape(label, ShapeType.MASK, [(x1, y1), (x2, y2)], mask=crop, **fields)


def to_full(shape: Shape, height: int, width: int) -> np.ndarray:
    """Place a mask shape's crop into a full-size ``(height, width)`` boolean array.

    Parts that fall outside the image (e.g. after dragging the shape off the edge) are
    clipped.
    """
    full = np.zeros((height, width), dtype=bool)
    if shape.mask is None:
        return full
    x1, y1 = round(shape.points[0][0]), round(shape.points[0][1])
    crop_h, crop_w = shape.mask.shape
    # Intersect the crop rectangle with the image rectangle.
    fx0, fy0 = max(x1, 0), max(y1, 0)
    fx1, fy1 = min(x1 + crop_w, width), min(y1 + crop_h, height)
    if fx0 >= fx1 or fy0 >= fy1:
        return full
    full[fy0:fy1, fx0:fx1] = shape.mask[fy0 - y1 : fy1 - y1, fx0 - x1 : fx1 - x1]
    return full


# ---------------------------------------------------------------------- editing


def paint_stroke(full: np.ndarray, start: Point, end: Point, radius: int, value: bool) -> None:
    """Paint (``value=True``) or erase a round brush stroke from ``start`` to ``end`` in place."""
    canvas = full.view(np.uint8)
    p0 = (int(start[0]), int(start[1]))
    p1 = (int(end[0]), int(end[1]))
    color = 1 if value else 0
    cv2.line(canvas, p0, p1, color, thickness=max(1, 2 * radius), lineType=cv2.LINE_8)
    cv2.circle(canvas, p0, radius, color, thickness=-1)
    cv2.circle(canvas, p1, radius, color, thickness=-1)


def fill_holes(full: np.ndarray) -> np.ndarray:
    """Fill every background region not connected to the image border."""
    padded = np.pad(full, 1).astype(np.uint8)
    flood = padded.copy()
    mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 1)  # paint the outside background
    holes = flood == 0
    return np.asarray(full | holes[1:-1, 1:-1], dtype=bool)


def remove_small(full: np.ndarray, min_area: int) -> np.ndarray:
    """Drop connected foreground blobs smaller than ``min_area`` pixels."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        full.astype(np.uint8), connectivity=8
    )
    keep = np.zeros(count, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return np.asarray(keep[labels], dtype=bool)


def keep_regions_containing(full: np.ndarray, points: list[Point]) -> np.ndarray:
    """Keep only the connected blobs that contain at least one of ``points``.

    Removes stray regions a segmentation model adds far from what the user clicked. If no
    point lands on the mask, it is returned unchanged rather than emptied.
    """
    count, labels = cv2.connectedComponents(full.astype(np.uint8), connectivity=8)
    if count <= 2:  # background plus at most one blob: nothing to drop
        return full
    height, width = full.shape
    hit = {
        int(labels[min(max(int(y), 0), height - 1), min(max(int(x), 0), width - 1)])
        for x, y in points
    } - {0}
    if not hit:
        return full
    return np.isin(labels, list(hit))


def fill_small_holes(full: np.ndarray, max_area: int) -> np.ndarray:
    """Fill enclosed background holes smaller than ``max_area`` pixels; keep big ones.

    Unlike :func:`fill_holes`, a deliberate hole (a donut, or one cut with Alt) survives.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (~full).astype(np.uint8), connectivity=4
    )
    border = set(np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])))
    small = [
        i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] < max_area and i not in border
    ]
    return np.asarray(full | np.isin(labels, small), dtype=bool)


def clean_up(full: np.ndarray, min_area: int) -> np.ndarray:
    """Remove specks and fill pinholes smaller than ``min_area`` (model output tidy-up)."""
    return fill_small_holes(remove_small(full, min_area), min_area)


def cleanup_area(height: int, width: int) -> int:
    """Default speck / pinhole size for an image: 0.05% of its area, at least 20 pixels."""
    return max(20, round(0.0005 * height * width))


# ---------------------------------------------------------------------- conversion


def to_polygons(
    full: np.ndarray, tolerance: float = 1.0, min_area: float = 4.0
) -> list[list[Point]]:
    """Trace the outer boundary of each blob as a simplified polygon.

    Args:
        full: Boolean mask.
        tolerance: Max distance in pixels between the polygon and the true boundary
            (Douglas-Peucker). Larger means fewer vertices.
        min_area: Blobs whose polygon area is below this are skipped.

    Holes are not represented: each polygon is an outer boundary only.
    """
    contours, _ = cv2.findContours(full.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    polygons = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, tolerance, closed=True)
        if len(approx) >= 3 and cv2.contourArea(approx) >= min_area:
            polygons.append([(float(x), float(y)) for x, y in approx[:, 0, :]])
    return polygons


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union of two boolean masks (1.0 if both are empty)."""
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum() / union)


def shape_polygons(shape: Shape, tolerance: float = 1.0) -> list[list[Point]]:
    """Outline polygons of any area shape, in image coordinates.

    Polygons, rectangles and circles give one polygon; masks give one per blob. Used by
    exporters that need polygons (COCO, YOLO).
    """
    if shape.shape_type != ShapeType.MASK:
        return [shape.outline()]
    if shape.mask is None:
        return []
    x1, y1 = round(shape.points[0][0]), round(shape.points[0][1])
    return [
        [(x + x1, y + y1) for x, y in polygon]
        for polygon in to_polygons(shape.mask, tolerance=tolerance)
    ]
