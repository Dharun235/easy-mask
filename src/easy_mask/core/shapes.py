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
==========  ==============================================

This mirrors the labelme JSON format so files are interchangeable.
"""

import copy
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

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
        return self in (ShapeType.POLYGON, ShapeType.RECTANGLE, ShapeType.CIRCLE)


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
    """

    label: str
    shape_type: ShapeType
    points: list[Point]
    group_id: int | None = None
    description: str = ""
    flags: dict[str, bool] = field(default_factory=dict)

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
        raise ValueError(f"{self.shape_type} has no area outline")

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
        if self.shape_type.is_area:
            outline = self.outline()
            if geometry.point_in_polygon(p, outline):
                return True
            return geometry.distance_to_polyline(p, outline, closed=True) <= tolerance
        return geometry.distance_to_polyline(p, self.points, closed=False) <= tolerance

    def nearest_vertex(self, p: Point, tolerance: float) -> int | None:
        """Return the index of the vertex closest to ``p`` within ``tolerance``, if any."""
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
        )
