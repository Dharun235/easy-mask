import numpy as np

from easy_mask.core.rasterize import instance_mask, semantic_mask, shape_to_mask
from easy_mask.core.shapes import Shape, ShapeType


def test_polygon_mask_area(square):
    mask = shape_to_mask(square, 50, 50)
    assert mask.dtype == bool and mask.shape == (50, 50)
    assert 400 <= mask.sum() <= 450  # 20x20 interior plus outline pixels
    assert mask[20, 20] and not mask[5, 5]


def test_rectangle_and_circle_masks():
    rect = Shape("r", ShapeType.RECTANGLE, [(30, 30), (10, 10)])
    assert shape_to_mask(rect, 50, 50)[20, 20]
    circle = Shape("c", ShapeType.CIRCLE, [(25, 25), (35, 25)])
    mask = shape_to_mask(circle, 50, 50)
    assert mask[25, 25] and not mask[25, 40]
    assert abs(mask.sum() - np.pi * 100) < 40


def test_line_and_point_produce_pixels():
    line = Shape("l", ShapeType.LINE, [(0, 10), (49, 10)])
    point = Shape("p", ShapeType.POINT, [(25, 25)])
    assert shape_to_mask(line, 50, 50)[10].sum() == 50
    assert shape_to_mask(point, 50, 50)[25, 25]


def test_semantic_mask_later_shapes_win(square):
    dog = Shape("dog", ShapeType.RECTANGLE, [(20, 20), (40, 40)])
    mask = semantic_mask([square, dog], 50, 50, {"cat": 1, "dog": 2})
    assert mask[15, 15] == 1
    assert mask[25, 25] == 2  # overlap painted by the later shape
    assert mask[45, 5] == 0


def test_semantic_mask_skips_unknown_labels(square):
    assert semantic_mask([square], 50, 50, {"dog": 1}).max() == 0


def test_instance_mask_merges_groups(square):
    a = Shape("dog", ShapeType.RECTANGLE, [(35, 35), (40, 40)], group_id=7)
    b = Shape("dog", ShapeType.RECTANGLE, [(42, 42), (48, 48)], group_id=7)
    mask, instances = instance_mask([square, a, b], 50, 50)
    assert instances == [(1, "cat"), (2, "dog")]
    assert mask[20, 20] == 1 and mask[37, 37] == 2 and mask[45, 45] == 2


def test_shape_to_mask_is_writable(square):
    mask = shape_to_mask(square, 50, 50)
    mask[0, 0] = True  # brush and combine edit masks in place
    assert mask[0, 0]
