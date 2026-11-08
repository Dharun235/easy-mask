import numpy as np

from easy_mask.core import masks, operations
from easy_mask.core.rasterize import shape_to_mask
from easy_mask.core.shapes import Shape, ShapeType

H, W = 50, 50


def rect(label, x0, y0, x1, y1, **kw):
    return Shape(label, ShapeType.RECTANGLE, [(x0, y0), (x1, y1)], **kw)


def test_union_replaces_inputs_with_one_mask_at_target_position():
    shapes = [
        rect("bg", 40, 40, 45, 45),
        rect("cat", 0, 0, 10, 10, group_id=2),
        rect("dog", 5, 0, 20, 10),
    ]
    result, index = operations.combine(shapes, [1, 2], "union", H, W)
    assert index == 1 and len(result) == 2 and result[0] is shapes[0]
    merged = result[1]
    assert (merged.label, merged.group_id, merged.shape_type) == ("cat", 2, ShapeType.MASK)
    assert merged.mask.sum() == shape_to_mask(rect("x", 0, 0, 20, 10), H, W).sum()


def test_subtract_and_intersect():
    a, b = rect("a", 0, 0, 20, 20), rect("b", 10, 0, 30, 20)
    sub, _ = operations.combine([a, b], [0, 1], "subtract", H, W)
    inter, _ = operations.combine([a, b], [0, 1], "intersect", H, W)
    full_sub = masks.to_full(sub[0], H, W)
    assert full_sub[5, 5] and not full_sub[5, 15]
    full_inter = masks.to_full(inter[0], H, W)
    assert full_inter[5, 15] and not full_inter[5, 5]


def test_subtract_everything_removes_target():
    a = rect("a", 0, 0, 10, 10)
    result, index = operations.combine([a, rect("b", 0, 0, 20, 20)], [0, 1], "subtract", H, W)
    assert result == [] and index is None


def test_combine_does_not_mutate_input():
    shapes = [rect("a", 0, 0, 10, 10), rect("b", 5, 5, 15, 15)]
    before = [s.copy() for s in shapes]
    operations.combine(shapes, [0, 1], "union", H, W)
    assert shapes == before


def test_map_pixels_only_touches_selected_area_shapes():
    line = Shape("edge", ShapeType.LINE, [(0, 0), (10, 10)])
    shapes = [rect("a", 10, 10, 20, 20), line, rect("b", 30, 30, 40, 40)]
    result = operations.map_pixels(shapes, [0, 1], lambda f: masks.grow(f, 1), H, W)
    assert result[0].shape_type == ShapeType.MASK and result[1] is line and result[2] is shapes[2]


def test_map_pixels_drops_shapes_that_become_empty():
    result = operations.map_pixels([rect("a", 0, 0, 2, 2)], [0], lambda f: np.zeros_like(f), H, W)
    assert result == []


def test_to_masks_then_to_polygons_roundtrip():
    shapes = [rect("a", 10, 10, 30, 20, group_id=1, description="note")]
    as_mask = operations.to_masks(shapes, [0], H, W)
    assert as_mask[0].shape_type == ShapeType.MASK
    back = operations.to_polygons(as_mask, [0], H, W, tolerance=0.5)
    assert len(back) == 1 and back[0].shape_type == ShapeType.POLYGON
    assert (back[0].label, back[0].group_id, back[0].description) == ("a", 1, "note")
    assert masks.iou(shape_to_mask(back[0], H, W), shape_to_mask(shapes[0], H, W)) > 0.95


def test_to_polygons_splits_blobs_and_skips_non_masks():
    full = np.zeros((H, W), bool)
    full[0:5, 0:5] = full[20:30, 20:30] = True
    shapes = [masks.to_shape("m", full), rect("r", 0, 0, 3, 3)]
    result = operations.to_polygons(shapes, [0, 1], H, W)
    assert [s.shape_type for s in result] == [
        ShapeType.POLYGON,
        ShapeType.POLYGON,
        ShapeType.RECTANGLE,
    ]
