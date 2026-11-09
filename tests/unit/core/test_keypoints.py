import numpy as np
import pytest

from easy_mask.core.keypoints import (
    CENTROID,
    CORNERS,
    ROLES,
    is_keypoint,
    mask_keypoints,
    object_keypoints,
    with_keypoints,
    without_keypoints,
)
from easy_mask.core.shapes import Shape, ShapeType


def rect(label="dog", x0=10, y0=10, x1=29, y1=19, group_id=None):
    return Shape(label, ShapeType.RECTANGLE, [(x0, y0), (x1, y1)], group_id=group_id)


def test_mask_keypoints_of_an_upright_box():
    full = np.zeros((40, 50), bool)
    full[10:20, 10:30] = True
    points = mask_keypoints(full)
    assert list(points) == list(ROLES)
    assert points[CENTROID] == (19.5, 14.5)
    xs = sorted(x for x, _ in (points[c] for c in CORNERS))
    assert xs[0] == pytest.approx(10, abs=0.6) and xs[-1] == pytest.approx(29, abs=0.6)
    assert points["corner 1"][0] < points["corner 2"][0]  # top-left, then top-right


def test_with_keypoints_adds_editable_points_and_replaces_old_ones():
    shapes = with_keypoints([rect()], 40, 50)
    points = [s for s in shapes if is_keypoint(s)]
    assert [p.description for p in points] == list(ROLES)
    assert all(p.label == "dog" and p.shape_type == ShapeType.POINT for p in points)
    again = with_keypoints(shapes, 40, 50)
    assert len(again) == len(shapes)  # replaced, not doubled
    assert without_keypoints(again) == [rect()]
    assert object_keypoints(again)[CENTROID] == pytest.approx((19.5, 14.5), abs=0.6)


def test_with_keypoints_only_for_chosen_objects():
    shapes = [rect("dog"), rect("cat", 30, 25, 45, 35)]
    result = with_keypoints(shapes, 40, 50, keys={("cat", None)})
    assert {s.label for s in result if is_keypoint(s)} == {"cat"}


def test_hand_drawn_point_is_not_a_keypoint():
    assert not is_keypoint(Shape("dog", ShapeType.POINT, [(1, 1)]))
