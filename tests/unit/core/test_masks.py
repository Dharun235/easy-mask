import numpy as np
import pytest

from easy_mask.core import masks
from easy_mask.core.shapes import Shape, ShapeType, decode_mask, encode_mask


def blob(height=40, width=50, box=(10, 5, 20, 15)):
    """Full mask with a filled rectangle x0,y0,x1,y1 (exclusive)."""
    full = np.zeros((height, width), dtype=bool)
    x0, y0, x1, y1 = box
    full[y0:y1, x0:x1] = True
    return full


def test_encode_decode_roundtrip():
    mask = np.random.default_rng(0).random((7, 9)) > 0.5
    assert np.array_equal(decode_mask(encode_mask(mask)), mask)


def test_to_shape_crops_to_inclusive_bbox():
    shape = masks.to_shape("cat", blob(), group_id=3)
    assert shape.shape_type == ShapeType.MASK and shape.group_id == 3
    assert shape.points == [(10, 5), (19, 14)]
    assert shape.mask.shape == (10, 10) and shape.mask.all()


def test_to_shape_of_empty_mask_is_none():
    assert masks.to_shape("cat", np.zeros((5, 5), bool)) is None


def test_to_full_roundtrip_and_clipping():
    full = blob()
    shape = masks.to_shape("cat", full)
    assert np.array_equal(masks.to_full(shape, 40, 50), full)
    shape.translate(-15, 0)  # half off the left edge
    moved = masks.to_full(shape, 40, 50)
    assert moved.sum() == 5 * 10 and moved[5:15, 0:5].all()
    shape.translate(-100, 0)  # completely off the image
    assert not masks.to_full(shape, 40, 50).any()


def test_mask_shape_hit_and_no_vertices():
    shape = masks.to_shape("cat", blob())
    assert shape.hit((12, 7), tolerance=0)
    assert not shape.hit((2, 2), tolerance=0)
    assert shape.nearest_vertex((10, 5), tolerance=5) is None


def test_mask_shape_json_roundtrip_is_labelme_style():
    shape = masks.to_shape("cat", blob())
    data = shape.to_dict()
    assert data["shape_type"] == "mask" and isinstance(data["mask"], str)
    assert data["points"] == [[10.0, 5.0], [19.0, 14.0]]
    assert Shape.from_dict(data) == shape


def test_vector_shapes_serialize_mask_as_null(square):
    assert square.to_dict()["mask"] is None


def test_shape_equality_compares_mask_pixels():
    a = masks.to_shape("cat", blob())
    b = a.copy()
    assert a == b
    b.mask[0, 0] = False
    assert a != b


def test_paint_and_erase_stroke():
    full = np.zeros((30, 30), bool)
    masks.paint_stroke(full, (5, 15), (25, 15), radius=2, value=True)
    assert full[15, 5:26].all() and not full[0].any()
    masks.paint_stroke(full, (15, 10), (15, 20), radius=1, value=False)
    assert not full[15, 15]


def test_fill_holes():
    full = blob(box=(5, 5, 25, 25))
    full[10:15, 10:15] = False  # hole
    open_notch = full.copy()
    open_notch[10:15, 5:15] = False  # touches the outside: not a hole
    assert masks.fill_holes(full)[12, 12]
    assert not masks.fill_holes(open_notch)[12, 7]


def test_remove_small_keeps_big_blobs():
    full = blob(box=(5, 5, 25, 25))
    full[35, 45] = True  # 1-pixel speck
    cleaned = masks.remove_small(full, min_area=10)
    assert cleaned.sum() == 400 and not cleaned[35, 45]


def test_grow_shrink_smooth():
    full = blob(box=(10, 10, 20, 20))
    assert masks.grow(full, 2).sum() > full.sum()
    assert masks.shrink(full, 2).sum() < full.sum()
    jagged = full.copy()
    jagged[5, 5] = True  # isolated speck disappears when smoothing
    assert not masks.smooth(jagged, 1)[5, 5]


def test_to_polygons_one_per_blob():
    full = blob(box=(2, 2, 12, 12)) | blob(box=(20, 20, 30, 30))
    polygons = masks.to_polygons(full, tolerance=0.5)
    assert len(polygons) == 2
    assert all(len(p) == 4 for p in polygons)  # rectangles simplify to 4 corners


def test_shape_polygons_offsets_mask_contours(square):
    shape = masks.to_shape("cat", blob(box=(10, 5, 20, 15)))
    (polygon,) = masks.shape_polygons(shape)
    xs = [x for x, _ in polygon]
    assert min(xs) == 10 and max(xs) == 19
    assert masks.shape_polygons(square) == [square.outline()]


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [((0, 0, 10, 10), (0, 0, 10, 10), 1.0), ((0, 0, 10, 10), (5, 0, 15, 10), 1 / 3)],
)
def test_iou(a, b, expected):
    assert masks.iou(blob(box=a), blob(box=b)) == pytest.approx(expected)
    assert masks.iou(np.zeros((3, 3), bool), np.zeros((3, 3), bool)) == 1.0


def test_keep_regions_containing_drops_stray_blobs():
    full = blob(box=(5, 5, 15, 15)) | blob(box=(30, 20, 40, 30))
    kept = masks.keep_regions_containing(full, [(10, 10)])
    assert kept[10, 10] and not kept[25, 35]
    both = masks.keep_regions_containing(full, [(10, 10), (35, 25)])
    assert np.array_equal(both, full)


def test_keep_regions_containing_keeps_mask_when_no_point_hits():
    full = blob(box=(5, 5, 15, 15)) | blob(box=(30, 20, 40, 30))
    assert np.array_equal(masks.keep_regions_containing(full, [(0, 39)]), full)
    single = blob(box=(5, 5, 15, 15))
    assert masks.keep_regions_containing(single, [(0, 0)]) is single  # one blob: untouched
