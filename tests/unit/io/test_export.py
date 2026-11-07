import json

import numpy as np
import pytest
from PIL import Image

from easy_mask.core import annotation as ann_io
from easy_mask.core.annotation import Annotation
from easy_mask.core.shapes import Shape, ShapeType
from easy_mask.io import export


@pytest.fixture
def annotated(image_dir, square):
    """img1 has a cat polygon, a grouped two-part dog and a line; img2 is unannotated."""
    shapes = [
        square,
        Shape("dog", ShapeType.RECTANGLE, [(50, 10), (60, 20)], group_id=1),
        Shape("dog", ShapeType.RECTANGLE, [(70, 10), (80, 20)], group_id=1),
        Shape("edge", ShapeType.LINE, [(0, 70), (90, 70)]),
    ]
    ann_io.save(Annotation("img1.png", 100, 80, shapes), image_dir / "img1.json")
    return export.collect(image_dir)


def test_collect_skips_unannotated(annotated):
    assert [item.image_path.name for item in annotated] == ["img1.png"]


def test_export_masks(annotated, tmp_path):
    out = tmp_path / "out"
    export.export_masks(annotated, out)
    assert (out / "class_names.txt").read_text().split() == ["_background_", "cat", "dog", "edge"]
    semantic = np.array(Image.open(out / "semantic" / "img1.png"))
    assert semantic[20, 20] == 1 and semantic[15, 55] == 2 and semantic[70, 50] == 3
    instance = np.array(Image.open(out / "instance" / "img1.png"))
    assert instance[15, 55] == instance[15, 75] == 2  # grouped dog parts share an id
    assert (out / "images" / "img1.png").exists()


def test_export_masks_respects_label_order(annotated, tmp_path):
    export.export_masks(annotated, tmp_path / "out", labels=["dog", "cat"])
    semantic = np.array(Image.open(tmp_path / "out" / "semantic" / "img1.png"))
    assert semantic[15, 55] == 1 and semantic[20, 20] == 2 and semantic[70, 50] == 0


def test_export_coco(annotated, tmp_path):
    export.export_coco(annotated, tmp_path / "out")
    coco = json.loads((tmp_path / "out" / "annotations.json").read_text())
    assert [c["name"] for c in coco["categories"]] == ["cat", "dog", "edge"]
    assert coco["images"][0]["file_name"] == "images/img1.png"
    # cat + one merged dog; the line is not an area shape
    assert len(coco["annotations"]) == 2
    cat, dog = coco["annotations"]
    assert cat["area"] == 400 and cat["bbox"] == [10, 10, 20, 20]
    assert len(dog["segmentation"]) == 2 and dog["bbox"] == [50, 10, 30, 10]


def test_export_yolo(annotated, tmp_path):
    export.export_yolo(annotated, tmp_path / "out")
    lines = (tmp_path / "out" / "labels" / "img1.txt").read_text().splitlines()
    assert len(lines) == 3  # cat + two dog rectangles
    values = lines[0].split()
    assert values[0] == "0" and len(values) == 1 + 2 * 4
    assert float(values[1]) == pytest.approx(0.1) and float(values[2]) == pytest.approx(0.125)
    assert "names:\n  0: cat" in (tmp_path / "out" / "data.yaml").read_text()


def test_masks_reject_more_than_255_classes(annotated, tmp_path):
    with pytest.raises(ValueError, match="255"):
        export.export_masks(annotated, tmp_path / "out", labels=[f"c{i}" for i in range(256)])


def test_coco_skips_labels_outside_class_list(annotated, tmp_path):
    export.export_coco(annotated, tmp_path / "out", labels=["dog"])
    coco = json.loads((tmp_path / "out" / "annotations.json").read_text())
    assert [a["category_id"] for a in coco["annotations"]] == [1]
