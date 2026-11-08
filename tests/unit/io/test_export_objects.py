import csv
import json

import numpy as np
import pytest
from PIL import Image

from easy_mask.core import annotation as ann_io
from easy_mask.core import masks
from easy_mask.core.annotation import Annotation
from easy_mask.core.shapes import Shape, ShapeType
from easy_mask.io import export
from easy_mask.io.export.objects import group_objects, safe_name


@pytest.fixture
def scene(image_dir):
    """img1 (100x80, coloured): a cat mask, a two-part dog (group 1) and a point."""
    rgb = np.zeros((80, 100, 3), np.uint8)
    rgb[..., 0] = np.arange(100)[None, :]  # red ramp so cutout pixels are checkable
    rgb[..., 1] = np.arange(80)[:, None]
    Image.fromarray(rgb).save(image_dir / "img1.png")
    cat = np.zeros((80, 100), bool)
    cat[10:20, 10:30] = True
    shapes = [
        masks.to_shape("cat", cat),
        Shape("dog", ShapeType.RECTANGLE, [(50, 10), (60, 20)], group_id=1),
        Shape("dog", ShapeType.RECTANGLE, [(70, 10), (80, 20)], group_id=1),
        Shape("nose / tip", ShapeType.POINT, [(40, 60)]),
    ]
    ann_io.save(Annotation("img1.png", 100, 80, shapes), image_dir / "img1.json")
    return export.collect(image_dir), rgb, cat


def test_all_parts(scene, tmp_path):
    items, rgb, cat = scene
    export.export_objects(items, tmp_path)
    names = sorted(p.name for p in (tmp_path / "masks").iterdir())
    assert names == ["img1_01_cat.png", "img1_02_dog.png", "img1_03_nose-tip.png"]

    mask = np.array(Image.open(tmp_path / "masks" / "img1_01_cat.png"))
    assert mask.shape == (80, 100) and set(np.unique(mask)) == {0, 255}
    assert np.array_equal(mask == 255, cat)  # binary mask is pixel-exact

    cutout = np.array(Image.open(tmp_path / "cutouts" / "img1_01_cat.png"))
    assert cutout.shape == (10, 20, 4)  # cropped to the object
    assert np.array_equal(cutout[..., :3], rgb[10:20, 10:30])  # original pixels
    assert (cutout[..., 3] == 255).all()

    records = json.loads((tmp_path / "objects.json").read_text())
    cat_rec, dog_rec, nose_rec = records
    assert cat_rec["bbox"] == [10, 10, 29, 19] and cat_rec["area"] == 200
    assert dog_rec["group_id"] == 1 and len(dog_rec["polygons"]) == 2  # one object, two parts
    assert nose_rec["polygons"] == [[[40.0, 60.0]]] and nose_rec["shape_types"] == ["point"]
    assert cat_rec["mask_file"] == "masks/img1_01_cat.png"

    rows = list(csv.DictReader((tmp_path / "objects.csv").open()))
    assert [r["label"] for r in rows] == ["cat", "dog", "nose / tip"]
    assert rows[1]["group_id"] == "1" and rows[0]["group_id"] == ""
    assert rows[0]["x_min"] == "10" and rows[0]["polygon"].count(",") >= 4


def test_cutout_is_transparent_outside_the_object(scene, tmp_path):
    items, _, _ = scene
    export.export_objects(items, tmp_path, parts=("cutouts",))
    dog = np.array(Image.open(tmp_path / "cutouts" / "img1_02_dog.png"))
    assert dog[..., 3][5, 5] == 255 and dog[..., 3][5, 15] == 0  # the gap between parts


def test_only_chosen_parts_are_written(scene, tmp_path):
    items, _, _ = scene
    export.export_objects(items, tmp_path, parts=("coordinates",))
    assert (tmp_path / "objects.json").exists() and not (tmp_path / "masks").exists()
    records = json.loads((tmp_path / "objects.json").read_text())
    assert records[0]["mask_file"] == "" and records[0]["cutout_file"] == ""


@pytest.mark.parametrize("parts", [(), ("masks", "nope")])
def test_invalid_parts_rejected(scene, tmp_path, parts):
    with pytest.raises(ValueError, match="parts"):
        export.export_objects(scene[0], tmp_path, parts=parts)


def test_off_image_shapes_are_skipped(image_dir, tmp_path):
    shape = Shape("ghost", ShapeType.RECTANGLE, [(-50, -50), (-10, -10)])
    ann_io.save(Annotation("img1.png", 100, 80, [shape]), image_dir / "img1.json")
    export.export_objects(export.collect(image_dir), tmp_path)
    assert json.loads((tmp_path / "objects.json").read_text()) == []


def test_group_objects_and_safe_name():
    a = Shape("x", ShapeType.POINT, [(1, 1)], group_id=2)
    b = Shape("x", ShapeType.POINT, [(2, 2)])
    c = Shape("x", ShapeType.POINT, [(3, 3)], group_id=2)
    assert [len(o.shapes) for o in group_objects([a, b, c])] == [2, 1]
    assert safe_name("my cat!") == "my-cat" and safe_name("***") == "object"
