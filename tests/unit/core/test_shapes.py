import pytest

from easy_mask.core.shapes import Shape, ShapeType


def test_shape_type_point_rules():
    assert ShapeType.POLYGON.min_points == 3 and ShapeType.POLYGON.fixed_points is None
    assert ShapeType.RECTANGLE.fixed_points == 2
    assert ShapeType.POINT.fixed_points == 1
    assert {t for t in ShapeType if t.is_area} == {
        ShapeType.POLYGON,
        ShapeType.RECTANGLE,
        ShapeType.CIRCLE,
        ShapeType.MASK,
    }


def test_rectangle_outline_and_bbox_ignore_corner_order():
    rect = Shape("box", ShapeType.RECTANGLE, [(30, 40), (10, 20)])
    assert len(rect.outline()) == 4
    assert rect.bounding_box() == (10, 20, 30, 40)


def test_circle_bbox_and_radius():
    circle = Shape("ball", ShapeType.CIRCLE, [(50, 50), (53, 54)])
    assert circle.radius() == 5
    assert circle.bounding_box() == (45, 45, 55, 55)


def test_outline_rejects_open_shapes():
    with pytest.raises(ValueError):
        Shape("edge", ShapeType.LINE, [(0, 0), (1, 1)]).outline()


def test_hit_inside_near_edge_and_outside(square):
    assert square.hit((20, 20), tolerance=2)  # inside
    assert square.hit((31.5, 20), tolerance=2)  # just outside the edge
    assert not square.hit((40, 20), tolerance=2)


def test_hit_line_uses_distance_only():
    line = Shape("edge", ShapeType.LINE, [(0, 0), (10, 0)])
    assert line.hit((5, 1), tolerance=2)
    assert not line.hit((5, 5), tolerance=2)


def test_nearest_vertex(square):
    assert square.nearest_vertex((29, 29), tolerance=3) == 2
    assert square.nearest_vertex((20, 20), tolerance=3) is None


def test_translate_and_copy_are_independent(square):
    clone = square.copy()
    clone.translate(5, -5)
    assert clone.points[0] == (15, 5)
    assert square.points[0] == (10, 10)


def test_dict_roundtrip(square):
    square.group_id = 3
    square.flags = {"occluded": True}
    assert Shape.from_dict(square.to_dict()) == square


def test_from_dict_accepts_labelme_nulls():
    shape = Shape.from_dict(
        {
            "label": "a",
            "points": [[1, 2]],
            "shape_type": "point",
            "description": None,
            "flags": None,
        }
    )
    assert shape.description == "" and shape.flags == {}


def test_from_dict_rejects_unknown_type():
    with pytest.raises(ValueError):
        Shape.from_dict({"label": "a", "points": [[1, 2]], "shape_type": "hexagon"})
