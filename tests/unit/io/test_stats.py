import json

import numpy as np
import pytest

from easy_mask.core import annotation as ann_io
from easy_mask.core import masks
from easy_mask.core.annotation import REVIEWED_FLAG, Annotation
from easy_mask.core.shapes import Shape, ShapeType
from easy_mask.io import stats


@pytest.fixture
def folder(image_dir, square):
    """img1: cat square + two-part dog + a line; img2: empty but reviewed; img10: none."""
    shapes = [
        square,
        Shape("dog", ShapeType.RECTANGLE, [(50, 10), (60, 20)], group_id=1),
        Shape("dog", ShapeType.RECTANGLE, [(70, 10), (80, 20)], group_id=1),
        Shape("edge", ShapeType.LINE, [(0, 70), (90, 70)]),
    ]
    ann_io.save(Annotation("img1.png", 100, 80, shapes), image_dir / "img1.json")
    ann_io.save(
        Annotation("img2.png", 100, 80, flags={REVIEWED_FLAG: True}), image_dir / "img2.json"
    )
    return image_dir


def test_image_counts(folder):
    result = stats.compute(folder)
    assert (result.images, result.annotated, result.unannotated) == (3, 2, 1)
    assert result.reviewed == 1


def test_grouped_shapes_count_as_one_object(folder):
    result = stats.compute(folder)
    assert (result.objects, result.shapes) == (3, 4)
    assert (result.labels["dog"].objects, result.labels["dog"].shapes) == (1, 2)


def test_per_label_area_and_images(folder):
    labels = stats.compute(folder).labels
    assert list(labels) == ["cat", "dog", "edge"]
    assert labels["cat"].area == pytest.approx(400)
    assert labels["dog"].area == pytest.approx(200)
    assert labels["edge"].area == 0
    assert labels["cat"].images == 1


def test_mask_area_counts_pixels(image_dir):
    full = np.zeros((80, 100), bool)
    full[0:3, 0:5] = True
    shape = masks.to_shape("blob", full)
    ann_io.save(Annotation("img1.png", 100, 80, [shape]), image_dir / "img1.json")
    assert stats.compute(image_dir).labels["blob"].area == 15


def test_unknown_labels_reported_only_with_label_list(folder):
    assert stats.compute(folder).unknown_labels == {}
    result = stats.compute(folder, allowed_labels=["cat"])
    assert result.unknown_labels == {"dog": ["img1.png"], "edge": ["img1.png"]}


def test_annotation_dir(image_dir, square, tmp_path):
    labels_dir = tmp_path / "labels"
    ann_io.save(Annotation("x", 100, 80, [square]), labels_dir / "img2.json")
    assert stats.compute(image_dir).annotated == 0
    assert stats.compute(image_dir, labels_dir).annotated == 1


def test_format_table(folder):
    text = stats.format_table(stats.compute(folder, allowed_labels=["cat", "dog"]), validated=True)
    assert "annotated 2, unannotated 1, reviewed 1" in text
    assert text.splitlines()[4].split() == ["cat", "1", "1", "1", "400"]
    assert "edge: img1.png" in text


def test_format_table_all_valid(folder):
    result = stats.compute(folder, allowed_labels=["cat", "dog", "edge"])
    assert "All labels are in the label list." in stats.format_table(result, validated=True)
    assert "label list" not in stats.format_table(result)


def test_to_dict_is_json_ready(folder):
    data = json.loads(json.dumps(stats.compute(folder).to_dict()))
    assert data["unannotated"] == 1
    assert data["labels"]["dog"]["objects"] == 1
