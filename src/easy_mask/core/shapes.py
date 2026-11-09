"""The ``Shape`` model: one annotated object drawn on an image.

A shape is a label plus a list of points whose meaning depends on its type:

==========  ==============================================
type        points
==========  ==============================================
polygon     3+ vertices, implicitly closed
rectangle   2 opposite corners
circle      centre, then any point on the rim
line        2 end points
linestrip   2+ vertices, open
point       1 point
mask        2 inclusive bbox corners + a boolean ``mask`` crop
==========  ==============================================

This mirrors the labelme JSON format so files are interchangeable.
"""

import base64
import copy
import io
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
from PIL import Image

from easy_mask.core import geometry
from easy_mask.core.geometry import Point


class ShapeType(StrEnum):
    """The kinds of shape a user can draw."""

    POLYGON = "polygon"
    RECTANGLE = "rectangle"
    CIRCLE = "circle"
    LINE = "line"
    LINESTRIP = "linestrip"
    POINT = "point"
    MASK = "mask"

    @property
    def min_points(self) -> int:
        """Fewest points needed for a valid shape of this type."""
        return {ShapeType.POLYGON: 3, ShapeType.POINT: 1}.get(self, 2)

    @property
    def fixed_points(self) -> int | None:
        """Exact number of points for fixed-size types, or None if open-ended."""
        if self in (ShapeType.POLYGON, ShapeType.LINESTRIP):
            return None
        return self.min_points

    @property
    def is_area(self) -> bool:
        """True for closed shapes that cover pixels (and so become masks)."""
        return self in (ShapeType.POLYGON, ShapeType.RECTANGLE, ShapeType.CIRCLE, ShapeType.MASK)


@dataclass
class Shape:
    """One labelled object on an image.

    Attributes:
        label: Class name, e.g. ``"cat"``.
        shape_type: Geometry kind; decides how ``points`` are read.
        points: Vertices in image pixel coordinates.
        group_id: Optional id tying several shapes into one object instance.
        description: Free-text note.
        flags: Per-shape boolean attributes.
        mask: For ``mask`` shapes only: boolean pixels inside the bounding box
            ``points``; see :mod:`easy_mask.core.masks`.
    """

    label: str
    shape_type: ShapeType
    points: list[Point]
    group_id: int | None = None
    description: str = ""
    flags: dict[str, bool] = field(default_factory=dict)
    mask: np.ndarray | None = field(default=None, repr=False)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Shape):
            return NotImplemented
        same_mask = (self.mask is None and other.mask is None) or (
            self.mask is not None
            and other.mask is not None
            and np.array_equal(self.mask, other.mask)
        )
        return same_mask and (
            self.label,
            self.shape_type,
            self.points,
            self.group_id,
            self.description,
            self.flags,
        ) == (
            other.label,
            other.shape_type,
            other.points,
            other.group_id,
            other.description,
            other.flags,
        )

    # ------------------------------------------------------------------ geometry

    def outline(self) -> list[Point]:
        """Return the closed outline of an area shape as polygon vertices.

        Raises:
            ValueError: If the shape is not an area shape (line, point, ...).
        """
        if self.shape_type == ShapeType.POLYGON:
            return list(self.points)
        if self.shape_type == ShapeType.RECTANGLE:
            (x1, y1), (x2, y2) = self.points
            return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        if self.shape_type == ShapeType.CIRCLE:
            return geometry.circle_to_polygon(self.points[0], self.radius())
        raise ValueError(f"{self.shape_type} has no single outline polygon")

    def radius(self) -> float:
        """Radius of a circle shape (distance from centre to rim point)."""
        return geometry.distance(self.points[0], self.points[1])

    def bounding_box(self) -> tuple[float, float, float, float]:
        """Return ``(x_min, y_min, x_max, y_max)`` covering the whole shape."""
        if self.shape_type == ShapeType.CIRCLE:
            (cx, cy), r = self.points[0], self.radius()
            return cx - r, cy - r, cx + r, cy + r
        return geometry.bounding_box(self.points)

    def hit(self, p: Point, tolerance: float) -> bool:
        """Return True if ``p`` is on or inside the shape, within ``tolerance`` pixels."""
        if self.shape_type == ShapeType.MASK:
            return self._mask_pixel(p)
        if self.shape_type.is_area:
            outline = self.outline()
            if geometry.point_in_polygon(p, outline):
                return True
            return geometry.distance_to_polyline(p, outline, closed=True) <= tolerance
        return geometry.distance_to_polyline(p, self.points, closed=False) <= tolerance

    def _mask_pixel(self, p: Point) -> bool:
        """True if ``p`` falls on a set pixel of this mask shape."""
        if self.mask is None:
            return False
        col = int(p[0] - round(self.points[0][0]))
        row = int(p[1] - round(self.points[0][1]))
        height, width = self.mask.shape
        return 0 <= row < height and 0 <= col < width and bool(self.mask[row, col])

    def nearest_vertex(self, p: Point, tolerance: float) -> int | None:
        """Return the index of the vertex closest to ``p`` within ``tolerance``, if any.

        Mask shapes have no editable vertices (their points are just the bounding box).
        """
        if self.shape_type == ShapeType.MASK:
            return None
        best, best_dist = None, tolerance
        for i, vertex in enumerate(self.points):
            d = geometry.distance(p, vertex)
            if d <= best_dist:
                best, best_dist = i, d
        return best

    def translate(self, dx: float, dy: float) -> None:
        """Move every point by ``(dx, dy)`` in place."""
        self.points = [(x + dx, y + dy) for x, y in self.points]

    def copy(self) -> "Shape":
        """Return an independent deep copy."""
        return copy.deepcopy(self)

    # ------------------------------------------------------------------ serialization

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a labelme-compatible dict."""
        return {
            "label": self.label,
            "points": [[float(x), float(y)] for x, y in self.points],
            "group_id": self.group_id,
            "description": self.description,
            "shape_type": str(self.shape_type),
            "flags": dict(self.flags),
            "mask": None if self.mask is None else encode_mask(self.mask),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Shape":
        """Build a shape from a labelme-compatible dict."""
        return cls(
            label=data["label"],
            shape_type=ShapeType(data.get("shape_type") or "polygon"),
            points=[(float(x), float(y)) for x, y in data["points"]],
            group_id=data.get("group_id"),
            description=data.get("description") or "",
            flags=dict(data.get("flags") or {}),
            mask=decode_mask(data["mask"]) if data.get("mask") else None,
        )


def encode_mask(mask: np.ndarray) -> str:
    """Encode a boolean array as a base64 PNG (labelme-compatible, values 0/1)."""
    buffer = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_mask(data: str) -> np.ndarray:
    """Decode a base64 PNG back into a boolean array (any non-zero pixel is True)."""
    with Image.open(io.BytesIO(base64.b64decode(data))) as image:
        return np.asarray(image.convert("L")) > 0
