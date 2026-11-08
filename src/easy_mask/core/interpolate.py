"""Keyframe interpolation: fill the frames between two annotated keyframes, no model needed.

An *object* is identified across frames by ``(label, group_id)``. For each object drawn on
both keyframes, the in-between frames get a blend of the two shapes:

* rectangles and circles blend their defining points directly;
* any other pair of vector area shapes (polygons, or a mix) blends outlines by vertex
  correspondence: both outlines are resampled to the same number of evenly spaced points,
  put in the same winding order, and the start point is rotated to the best match;
* if either keyframe holds a mask, both are rasterized, moved to the interpolated centroid
  and blended through their signed distance fields. Centring first makes a moving object
  slide instead of shrinking away where the two masks do not overlap.
"""

import cv2
import numpy as np

from easy_mask.core import masks
from easy_mask.core.geometry import Point
from easy_mask.core.rasterize import shape_to_mask
from easy_mask.core.shapes import Shape, ShapeType
from easy_mask.core.video import INTERPOLATED_FLAG, ObjectKey, object_key

OUTLINE_POINTS = 64  # resampled vertices per outline: smooth enough, small in JSON


# ---------------------------------------------------------------------- outlines


def resample_closed(points: list[Point], n: int) -> np.ndarray:
    """Resample a closed outline to ``n`` points evenly spaced along its perimeter.

    Returns:
        ``(n, 2)`` float array starting at ``points[0]``.
    """
    ring = np.asarray([*points, points[0]], dtype=float)
    lengths = np.hypot(*np.diff(ring, axis=0).T)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    if cumulative[-1] == 0:  # degenerate outline: all points coincide
        return np.repeat(ring[:1], n, axis=0)
    targets = np.linspace(0, cumulative[-1], n, endpoint=False)
    return np.stack(
        [np.interp(targets, cumulative, ring[:, 0]), np.interp(targets, cumulative, ring[:, 1])],
        axis=1,
    )


def _signed_area(ring: np.ndarray) -> float:
    x, y = ring[:, 0], ring[:, 1]
    return float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)) / 2


def align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Reorder ``b`` (same length as ``a``) so ``b[i]`` corresponds to ``a[i]``.

    Matches the winding direction, then picks the cyclic shift with the smallest total
    squared distance to ``a``.
    """
    if np.sign(_signed_area(a)) != np.sign(_signed_area(b)):
        b = b[::-1]
    costs = [np.sum((a - np.roll(b, -shift, axis=0)) ** 2) for shift in range(len(b))]
    return np.roll(b, -int(np.argmin(costs)), axis=0)


def interpolate_outline(a: list[Point], b: list[Point], t: float) -> list[Point]:
    """Blend two closed outlines; ``t=0`` gives ``a``'s shape, ``t=1`` gives ``b``'s."""
    ra = resample_closed(a, OUTLINE_POINTS)
    rb = align(ra, resample_closed(b, OUTLINE_POINTS))
    blended = (1 - t) * ra + t * rb
    return [(float(x), float(y)) for x, y in blended]


# ---------------------------------------------------------------------- masks


def signed_distance(full: np.ndarray) -> np.ndarray:
    """Distance to the mask boundary: positive inside, negative outside."""
    as_u8 = full.astype(np.uint8)
    inside = cv2.distanceTransform(as_u8, cv2.DIST_L2, 5)
    outside = cv2.distanceTransform(1 - as_u8, cv2.DIST_L2, 5)
    return np.asarray(inside - outside, dtype=np.float32)


def _centroid(full: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(full)
    return np.array([xs.mean(), ys.mean()])


def _shift(full: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Translate a mask by ``(dx, dy)`` pixels; parts moved off the image are lost."""
    height, width = full.shape
    matrix = np.array([[1, 0, offset[0]], [0, 1, offset[1]]], dtype=np.float32)
    moved = cv2.warpAffine(full.astype(np.uint8), matrix, (width, height), flags=cv2.INTER_NEAREST)
    return np.asarray(moved, dtype=bool)


def interpolate_mask(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Blend two same-size boolean masks; see the module docstring for the method."""
    if not a.any() or not b.any():  # an empty mask has no boundary to morph from
        return (a if t < 0.5 else b).copy()
    ca, cb = _centroid(a), _centroid(b)
    centre = (1 - t) * ca + t * cb
    field = (1 - t) * signed_distance(_shift(a, centre - ca)) + t * signed_distance(
        _shift(b, centre - cb)
    )
    return np.asarray(field > 0)


def _lerp_points(a: list[Point], b: list[Point], t: float) -> list[Point]:
    return [
        ((1 - t) * xa + t * xb, (1 - t) * ya + t * yb)
        for (xa, ya), (xb, yb) in zip(a, b, strict=True)
    ]


def interpolate_shape(a: Shape, b: Shape, t: float, height: int, width: int) -> Shape | None:
    """Blend two area shapes of one object at ``0 <= t <= 1``.

    The result keeps ``a``'s label, group id and description and is flagged
    ``interpolated``. Returns None if a mask blend comes out empty.
    """
    result: Shape | None
    if a.shape_type == b.shape_type and a.shape_type in (ShapeType.RECTANGLE, ShapeType.CIRCLE):
        result = Shape(a.label, a.shape_type, _lerp_points(a.points, b.points, t))
    elif ShapeType.MASK not in (a.shape_type, b.shape_type):
        result = Shape(a.label, ShapeType.POLYGON, interpolate_outline(a.outline(), b.outline(), t))
    else:
        full = interpolate_mask(shape_to_mask(a, height, width), shape_to_mask(b, height, width), t)
        result = masks.to_shape(a.label, full)
        if result is None:
            return None
    result.group_id, result.description = a.group_id, a.description
    result.flags = {INTERPOLATED_FLAG: True}
    return result


def _single_area_shapes(shapes: list[Shape]) -> dict[ObjectKey, Shape]:
    """Area shapes keyed by object, skipping objects drawn as several shapes on a frame."""
    by_key: dict[ObjectKey, list[Shape]] = {}
    for shape in shapes:
        if shape.shape_type.is_area:
            by_key.setdefault(object_key(shape), []).append(shape)
    return {key: group[0] for key, group in by_key.items() if len(group) == 1}


def interpolate_frames(
    start: list[Shape], end: list[Shape], gap: int, height: int, width: int
) -> list[list[Shape]]:
    """Shapes for the ``gap - 1`` frames strictly between two keyframes ``gap`` apart.

    Only objects with exactly one area shape on both keyframes are interpolated.

    Returns:
        One list per in-between frame, in order.
    """
    first, last = _single_area_shapes(start), _single_area_shapes(end)
    common = [key for key in first if key in last]
    frames = []
    for step in range(1, gap):
        t = step / gap
        blended = [interpolate_shape(first[k], last[k], t, height, width) for k in common]
        frames.append([shape for shape in blended if shape is not None])
    return frames
