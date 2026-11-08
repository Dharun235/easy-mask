import numpy as np
import pytest

from easy_mask.core import masks
from easy_mask.core.interpolate import (
    align,
    interpolate_frames,
    interpolate_mask,
    interpolate_outline,
    interpolate_shape,
    resample_closed,
    signed_distance,
)
from easy_mask.core.shapes import Shape, ShapeType
from easy_mask.core.video import INTERPOLATED_FLAG

SQUARE = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]


def disc(cx, cy, r, h=60, w=80):
    yy, xx = np.mgrid[:h, :w]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


def test_resample_closed_spaces_points_evenly_along_perimeter():
    ring = resample_closed(SQUARE, 8)
    assert ring.shape == (8, 2)
    steps = np.hypot(*np.diff(np.vstack([ring, ring[:1]]), axis=0).T)
    assert steps == pytest.approx(np.full(8, 5.0))
    assert tuple(ring[0]) == (0.0, 0.0)


def test_resample_degenerate_outline_repeats_the_point():
    ring = resample_closed([(3.0, 4.0), (3.0, 4.0), (3.0, 4.0)], 5)
    assert (ring == [3.0, 4.0]).all()


def test_align_undoes_rotation_and_reversed_winding():
    a = resample_closed(SQUARE, 16)
    b = np.roll(a, 5, axis=0)[::-1]
    assert np.allclose(align(a, b), a)


def test_interpolate_outline_endpoints_and_midpoint():
    shifted = [(x + 20, y) for x, y in SQUARE]
    start = interpolate_outline(SQUARE, shifted, 0.0)
    middle = interpolate_outline(SQUARE, shifted, 0.5)
    assert np.allclose(start, resample_closed(SQUARE, len(start)))
    assert np.allclose(np.mean(middle, axis=0), [15.0, 5.0])


def test_signed_distance_sign():
    full = disc(40, 30, 10)
    field = signed_distance(full)
    assert field[30, 40] > 0 > field[0, 0]


def test_interpolate_mask_moves_and_grows_a_disc():
    middle = interpolate_mask(disc(20, 30, 6), disc(60, 30, 14), 0.5)
    _, xs = np.nonzero(middle)
    assert xs.mean() == pytest.approx(40, abs=1)
    assert masks.iou(middle, disc(40, 30, 10)) > 0.9


def test_interpolate_mask_with_empty_side_switches_halfway():
    a, empty = disc(20, 30, 6), np.zeros((60, 80), bool)
    assert (interpolate_mask(a, empty, 0.4) == a).all()
    assert not interpolate_mask(a, empty, 0.6).any()


def test_rectangles_blend_corners():
    a = Shape("car", ShapeType.RECTANGLE, [(0, 0), (10, 10)], group_id=2)
    b = Shape("car", ShapeType.RECTANGLE, [(20, 10), (40, 30)], group_id=2)
    mid = interpolate_shape(a, b, 0.5, 60, 80)
    assert mid.shape_type == ShapeType.RECTANGLE
    assert mid.points == [(10, 5), (25, 20)]
    assert (mid.group_id, mid.flags) == (2, {INTERPOLATED_FLAG: True})


def test_circles_blend_centre_and_rim():
    a = Shape("ball", ShapeType.CIRCLE, [(10, 10), (15, 10)])
    b = Shape("ball", ShapeType.CIRCLE, [(30, 10), (45, 10)])
    mid = interpolate_shape(a, b, 0.5, 60, 80)
    assert mid.shape_type == ShapeType.CIRCLE and mid.radius() == pytest.approx(10)


def test_polygon_and_rectangle_blend_to_polygon():
    a = Shape("box", ShapeType.POLYGON, SQUARE)
    b = Shape("box", ShapeType.RECTANGLE, [(20, 0), (30, 10)])
    mid = interpolate_shape(a, b, 0.5, 60, 80)
    assert mid.shape_type == ShapeType.POLYGON
    assert np.mean(mid.points, axis=0) == pytest.approx([15, 5], abs=0.5)


def test_mask_and_polygon_blend_to_mask():
    a = masks.to_shape("blob", disc(20, 30, 8), description="note")
    b = Shape("blob", ShapeType.POLYGON, [(50, 22), (66, 22), (66, 38), (50, 38)])
    mid = interpolate_shape(a, b, 0.5, 60, 80)
    assert mid.shape_type == ShapeType.MASK and mid.description == "note"
    assert mid.flags == {INTERPOLATED_FLAG: True}


def test_mask_blend_slides_disjoint_objects_instead_of_vanishing():
    a = masks.to_shape("blob", disc(15, 30, 4))
    b = masks.to_shape("blob", disc(65, 30, 4))
    mid = interpolate_shape(a, b, 0.5, 60, 80)
    assert masks.iou(masks.to_full(mid, 60, 80), disc(40, 30, 4)) > 0.8


def test_mask_blend_that_comes_out_empty_returns_none(monkeypatch):
    from easy_mask.core import interpolate

    monkeypatch.setattr(interpolate, "interpolate_mask", lambda a, b, t: np.zeros_like(a))
    a = masks.to_shape("blob", disc(15, 30, 4))
    assert interpolate_shape(a, a, 0.5, 60, 80) is None


def test_interpolate_frames_matches_objects_by_label_and_group():
    start = [
        Shape("car", ShapeType.RECTANGLE, [(0, 0), (10, 10)], group_id=1),
        Shape("car", ShapeType.RECTANGLE, [(50, 0), (60, 10)], group_id=2),
        Shape("tree", ShapeType.RECTANGLE, [(0, 40), (5, 50)]),  # not on the end frame
        Shape("road", ShapeType.LINE, [(0, 0), (9, 9)]),  # not an area: ignored
    ]
    end = [
        Shape("car", ShapeType.RECTANGLE, [(52, 0), (62, 10)], group_id=2),
        Shape("car", ShapeType.RECTANGLE, [(4, 0), (14, 10)], group_id=1),
    ]
    frames = interpolate_frames(start, end, 4, 60, 80)
    assert len(frames) == 3
    first = {s.group_id: s.points for s in frames[0]}
    assert first == {1: [(1, 0), (11, 10)], 2: [(50.5, 0), (60.5, 10)]}


def test_interpolate_frames_skips_objects_split_into_several_shapes():
    part = Shape("dog", ShapeType.RECTANGLE, [(0, 0), (5, 5)], group_id=1)
    other = Shape("dog", ShapeType.RECTANGLE, [(9, 0), (12, 5)], group_id=1)
    assert interpolate_frames([part, other], [part], 3, 60, 80) == [[], []]


def test_adjacent_keyframes_have_nothing_between():
    shape = Shape("dog", ShapeType.RECTANGLE, [(0, 0), (5, 5)])
    assert interpolate_frames([shape], [shape], 1, 60, 80) == []
