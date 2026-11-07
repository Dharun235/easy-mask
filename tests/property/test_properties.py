"""Property-based tests: invariants that must hold for *any* input Hypothesis can find.

Example-based unit tests check cases we thought of; these search for the ones we didn't.
"""

import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from easy_mask.core import geometry
from easy_mask.core.annotation import Annotation
from easy_mask.core.history import History
from easy_mask.core.labels import label_color
from easy_mask.core.rasterize import shape_to_mask
from easy_mask.core.shapes import Shape, ShapeType

coord = st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False)
point = st.tuples(coord, coord)
label = st.text(min_size=1, max_size=20)


@st.composite
def shapes(draw) -> Shape:
    """Any valid shape: right number of points for its type."""
    shape_type = draw(st.sampled_from(list(ShapeType)))
    n = shape_type.fixed_points or draw(st.integers(shape_type.min_points, 12))
    return Shape(
        label=draw(label),
        shape_type=shape_type,
        points=draw(st.lists(point, min_size=n, max_size=n)),
        group_id=draw(st.none() | st.integers(0, 10_000)),
        description=draw(st.text(max_size=30)),
        flags=draw(st.dictionaries(st.text(min_size=1, max_size=8), st.booleans(), max_size=3)),
    )


@given(shapes())
def test_shape_dict_roundtrip(shape):
    assert Shape.from_dict(shape.to_dict()) == shape


@given(st.lists(shapes(), max_size=5), st.dictionaries(label, st.booleans(), max_size=3))
def test_annotation_dict_roundtrip(shape_list, flags):
    ann = Annotation("a/b.png", 640, 480, shape_list, flags)
    assert Annotation.from_dict(ann.to_dict()) == ann


@given(shapes(), coord, coord)
def test_translate_moves_bbox_exactly(shape, dx, dy):
    x0, y0, x1, y1 = shape.bounding_box()
    moved = shape.copy()
    moved.translate(dx, dy)
    tol = 1e-6 * (1 + abs(dx) + abs(dy) + max(map(abs, (x0, y0, x1, y1))))
    expected = (x0 + dx, y0 + dy, x1 + dx, y1 + dy)
    for got, want in zip(moved.bounding_box(), expected, strict=True):
        assert math.isclose(got, want, abs_tol=tol)


@given(st.lists(point, min_size=3, max_size=10), coord, coord, st.integers(0, 9))
def test_polygon_area_invariant_to_translation_and_start_vertex(poly, dx, dy, k):
    area = geometry.polygon_area(poly)
    shifted = [(x + dx, y + dy) for x, y in poly]
    k %= len(poly)
    rotated = poly[k:] + poly[:k]
    assert area >= 0
    assert math.isclose(geometry.polygon_area(shifted), area, rel_tol=1e-6, abs_tol=1e-2)
    assert math.isclose(geometry.polygon_area(rotated), area, rel_tol=1e-9, abs_tol=1e-6)


@given(point, point, point)
def test_segment_distance_never_exceeds_endpoint_distance(p, a, b):
    d = geometry.distance_to_segment(p, a, b)
    assert d <= min(geometry.distance(p, a), geometry.distance(p, b)) + 1e-6


@given(st.integers(0, 60), st.integers(0, 60), st.integers(1, 40), st.integers(1, 40))
@settings(max_examples=50)
def test_rectangle_mask_area_close_to_geometric_area(x, y, w, h):
    rect = Shape("r", ShapeType.RECTANGLE, [(x, y), (x + w, y + h)])
    mask = shape_to_mask(rect, 100, 100)
    # Pillow fills inclusive pixel edges, so the count is between w*h and (w+1)*(h+1).
    assert w * h <= mask.sum() <= (w + 1) * (h + 1)
    assert mask.dtype == np.bool_


@given(shapes())
@settings(max_examples=50)
def test_every_shape_rasterizes_without_error(shape):
    mask = shape_to_mask(shape, 64, 64)
    assert mask.shape == (64, 64)


@given(st.lists(st.sampled_from(["push", "undo", "redo"]), max_size=40))
def test_history_matches_simple_model(ops):
    """Compare History against an obviously-correct list-and-index model."""
    history, states, index = History(), [[]], 0
    history.reset([])
    for step, op in enumerate(ops):
        if op == "push":
            new = [Shape(str(step), ShapeType.POINT, [(step, step)])]
            history.push(new)
            states, index = [*states[: index + 1], new], index + 1
        elif op == "undo":
            result = history.undo()
            assert (result is None) == (index == 0)
            index = max(index - 1, 0)
            assert result is None or result == states[index]
        else:
            result = history.redo()
            assert (result is None) == (index == len(states) - 1)
            index = min(index + 1, len(states) - 1)
            assert result is None or result == states[index]


@given(label, st.lists(label, max_size=5))
def test_label_color_is_deterministic_and_valid(name, labels):
    color = label_color(name, labels)
    assert color == label_color(name, labels)
    assert len(color) == 3 and all(0 <= c <= 255 for c in color)
