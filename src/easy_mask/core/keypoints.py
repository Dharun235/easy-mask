"""Key points of a mask: its centroid and the four corners of its tightest rotated box.

Key points are stored as ordinary point shapes, so they are drawn, dragged, deleted and
saved like any other shape. Each carries the label and group id of its object, its role
in ``description`` (``centroid``, ``corner 1`` ... ``corner 4``) and the ``keypoint`` flag.
Corners go clockwise from the top-left one. Exports read the (possibly hand-adjusted)
points back with :func:`object_keypoints`.
"""

import cv2
import numpy as np

from easy_mask.core.rasterize import shape_to_mask
from easy_mask.core.shapes import Shape, ShapeType
from easy_mask.core.video import ObjectKey, object_key

KEYPOINT_FLAG = "keypoint"
CENTROID = "centroid"
CORNERS = tuple(f"corner {n}" for n in range(1, 5))
ROLES = (CENTROID, *CORNERS)
Corners = list[list[float]]


def is_keypoint(shape: Shape) -> bool:
    """True for a key point shape (as made by :func:`with_keypoints`)."""
    return shape.shape_type == ShapeType.POINT and bool(shape.flags.get(KEYPOINT_FLAG))


def centroid(full: np.ndarray) -> tuple[float, float]:
    """The mean ``(x, y)`` of a mask's pixels."""
    ys, xs = np.nonzero(full)
    return float(xs.mean()), float(ys.mean())


def clockwise_from_top_left(points: np.ndarray) -> Corners:
    """Four corner points ordered top-left, top-right, bottom-right, bottom-left."""
    middle = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - middle[1], points[:, 0] - middle[0])
    ordered = points[np.argsort(angles)]  # clockwise on screen (y grows downwards)
    ordered = np.roll(ordered, -int(np.argmin(ordered.sum(axis=1))), axis=0)
    return [[round(float(x), 2), round(float(y), 2)] for x, y in ordered]


def rotated_corners(full: np.ndarray) -> Corners:
    """Corners of the smallest rotated rectangle around a mask's pixels."""
    contours, _ = cv2.findContours(full.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    points = np.concatenate([c.reshape(-1, 2) for c in contours]).astype(np.float32)
    return clockwise_from_top_left(cv2.boxPoints(cv2.minAreaRect(points)))


def mask_keypoints(full: np.ndarray) -> dict[str, tuple[float, float]]:
    """``{role: (x, y)}`` for a non-empty mask: its centroid and four rotated corners."""
    cx, cy = centroid(full)
    points = {CENTROID: (round(cx, 2), round(cy, 2))}
    for role, (x, y) in zip(CORNERS, rotated_corners(full), strict=True):
        points[role] = (x, y)
    return points


def with_keypoints(
    shapes: list[Shape], height: int, width: int, keys: set[ObjectKey] | None = None
) -> list[Shape]:
    """``shapes`` with fresh key points for every area object (or only ``keys``).

    Earlier key points of those objects are replaced; other shapes are kept as they are.
    """
    full_by_key: dict[ObjectKey, np.ndarray] = {}
    for shape in shapes:
        key = object_key(shape)
        if shape.shape_type.is_area and (keys is None or key in keys):
            full = shape_to_mask(shape, height, width)
            full_by_key[key] = full_by_key[key] | full if key in full_by_key else full
    kept = [s for s in shapes if not (is_keypoint(s) and object_key(s) in full_by_key)]
    for (label, group_id), full in full_by_key.items():
        if not full.any():
            continue
        for role, point in mask_keypoints(full).items():
            kept.append(
                Shape(
                    label,
                    ShapeType.POINT,
                    [point],
                    group_id=group_id,
                    description=role,
                    flags={KEYPOINT_FLAG: True},
                )
            )
    return kept


def without_keypoints(shapes: list[Shape]) -> list[Shape]:
    """``shapes`` with every key point removed."""
    return [s for s in shapes if not is_keypoint(s)]


def object_keypoints(shapes: list[Shape]) -> dict[str, tuple[float, float]]:
    """``{role: (x, y)}`` of the key points among ``shapes`` (one object's shapes)."""
    return {s.description: s.points[0] for s in shapes if is_keypoint(s)}
