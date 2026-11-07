"""Small, dependency-free 2D geometry helpers.

All functions work in image pixel space with ``(x, y)`` float tuples.
"""

import itertools
import math

Point = tuple[float, float]


def distance(a: Point, b: Point) -> float:
    """Return the Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def distance_to_segment(p: Point, a: Point, b: Point) -> float:
    """Return the shortest distance from point ``p`` to the segment ``a-b``."""
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return distance(p, a)
    # Project p onto the segment, clamped to its end points.
    t = ((p[0] - ax) * dx + (p[1] - ay) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    return distance(p, (ax + t * dx, ay + t * dy))


def distance_to_polyline(p: Point, points: list[Point], closed: bool) -> float:
    """Return the distance from ``p`` to a polyline (or polygon outline if ``closed``)."""
    if len(points) == 1:
        return distance(p, points[0])
    segments = list(itertools.pairwise(points))
    if closed:
        segments.append((points[-1], points[0]))
    return min(distance_to_segment(p, a, b) for a, b in segments)


def point_in_polygon(p: Point, polygon: list[Point]) -> bool:
    """Return True if ``p`` lies inside ``polygon`` (even-odd ray casting)."""
    x, y = p
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        crosses = (y1 > y) != (y2 > y)
        if crosses and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def polygon_area(polygon: list[Point]) -> float:
    """Return the absolute area of a simple polygon (shoelace formula)."""
    n = len(polygon)
    twice_area = sum(
        polygon[i][0] * polygon[(i + 1) % n][1] - polygon[(i + 1) % n][0] * polygon[i][1]
        for i in range(n)
    )
    return abs(twice_area) / 2.0


def bounding_box(points: list[Point]) -> tuple[float, float, float, float]:
    """Return ``(x_min, y_min, x_max, y_max)`` of a non-empty point list."""
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    return min(xs), min(ys), max(xs), max(ys)


def circle_to_polygon(center: Point, radius: float, segments: int = 64) -> list[Point]:
    """Approximate a circle with a regular polygon of ``segments`` vertices."""
    cx, cy = center
    return [
        (
            cx + radius * math.cos(2 * math.pi * i / segments),
            cy + radius * math.sin(2 * math.pi * i / segments),
        )
        for i in range(segments)
    ]
