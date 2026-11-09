import json

import numpy as np
import pytest

from easy_mask.core import annotation as ann_io
from easy_mask.core import masks, rle
from easy_mask.core.annotation import Annotation
from easy_mask.core.shapes import Shape, ShapeType
from easy_mask.io import export


def ring(height=80, width=100):
    """A square ring (mask with a hole) that polygons cannot represent exactly."""
    full = np.zeros((height, width), bool)
    full[10:50, 20:60] = True
    full[20:40, 30:50] = False
    return full


@pytest.fixture
def items(image_dir, square):
    mask_shape = masks.to_shape("donut", ring())
    speck = masks.to_shape("speck", np.pad(np.ones((1, 1), bool), ((70, 9), (90, 9))))
    shapes = [square, mask_shape, speck]
    ann_io.save(Annotation("img1.png", 100, 80, shapes), image_dir / "img1.json")
    return export.collect(image_dir)


def annotations(out):
    coco = json.loads((out / "annotations.json").read_text())
    names = {c["id"]: c["name"] for c in coco["categories"]}
    return {names[a["category_id"]]: a for a in coco["annotations"]}


def test_rle_export_keeps_mask_pixels_exactly(items, tmp_path):
    export.export_coco(items, tmp_path / "out", rle=True)
    donut = annotations(tmp_path / "out")["donut"]
    assert donut["segmentation"]["size"] == [80, 100]
    np.testing.assert_array_equal(rle.decode(donut["segmentation"]), ring())
    assert donut["area"] == int(ring().sum())
    assert donut["bbox"] == [20, 10, 40, 40]


def test_rle_export_keeps_vector_shapes_as_polygons(items, tmp_path):
    export.export_coco(items, tmp_path / "out", rle=True)
    assert isinstance(annotations(tmp_path / "out")["cat"]["segmentation"], list)


def test_rle_export_keeps_single_pixel_masks(items, tmp_path):
    export.export_coco(items, tmp_path / "out", rle=True)
    speck = annotations(tmp_path / "out")["speck"]
    assert speck["area"] == 1 and speck["bbox"] == [90, 70, 1, 1]


def test_polygon_export_unchanged_by_default(items, tmp_path):
    export.export_coco(items, tmp_path / "out")
    found = annotations(tmp_path / "out")
    assert isinstance(found["donut"]["segmentation"], list)
    assert "speck" not in found  # a 1-pixel blob traces to no polygon


def test_grouped_mask_and_polygon_become_one_rle(image_dir, square):
    blob = masks.to_shape("cat", ring(), group_id=3)
    square.group_id = 3
    ann_io.save(Annotation("img1.png", 100, 80, [square, blob]), image_dir / "img1.json")
    export.EXPORTERS["coco-rle"](export.collect(image_dir), image_dir / "out")
    (cat,) = annotations(image_dir / "out").values()
    decoded = rle.decode(cat["segmentation"])
    assert decoded[20, 20] and decoded[15, 25]  # square pixel and ring pixel
    assert cat["area"] == int(decoded.sum())


def test_empty_mask_shape_is_skipped(image_dir):
    empty = Shape("void", ShapeType.MASK, [(0, 0), (1, 1)], mask=np.zeros((2, 2), bool))
    ann_io.save(Annotation("img1.png", 100, 80, [empty]), image_dir / "img1.json")
    export.export_coco(export.collect(image_dir), image_dir / "out", rle=True)
    assert annotations(image_dir / "out") == {}
