import pytest

from easy_mask.core import geometry


def test_distance_to_segment_projects_and_clamps():
    assert geometry.distance_to_segment((5, 3), (0, 0), (10, 0)) == 3
    assert geometry.distance_to_segment((-3, 4), (0, 0), (10, 0)) == 5  # beyond the end
    assert geometry.distance_to_segment((3, 4), (0, 0), (0, 0)) == 5  # degenerate segment


def test_point_in_polygon():
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert geometry.point_in_polygon((5, 5), square)
    assert not geometry.point_in_polygon((15, 5), square)


def test_polygon_area_is_orientation_independent():
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert geometry.polygon_area(square) == 100
    assert geometry.polygon_area(square[::-1]) == 100


def test_circle_to_polygon_points_lie_on_circle():
    for p in geometry.circle_to_polygon((5, 5), 2, segments=16):
        assert geometry.distance(p, (5, 5)) == pytest.approx(2)
