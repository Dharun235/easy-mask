import numpy as np
import pytest
from PIL import Image

from easy_mask.core import annotation as ann_io
from easy_mask.core.annotation import Annotation
from easy_mask.core.shapes import Shape, ShapeType
from easy_mask.io import export
from easy_mask.io.export.davis import davis_palette, object_ids


def rect(label, x, group_id=None):
    return Shape(label, ShapeType.RECTANGLE, [(x, 10), (x + 9, 19)], group_id=group_id)


@pytest.fixture
def video_items(image_dir):
    """img1: car g1 + person; img2: person + car g2 + car g1 (new object appears)."""
    frames = {
        "img1": [rect("car", 0, 1), rect("person", 40)],
        "img2": [rect("person", 42), rect("car", 70, 2), rect("car", 2, 1)],
    }
    for stem, shapes in frames.items():
        shapes.append(Shape("lane", ShapeType.LINE, [(0, 70), (90, 70)]))  # not exported
        ann_io.save(Annotation(f"{stem}.png", 100, 80, shapes), image_dir / f"{stem}.json")
    return export.collect(image_dir)


def test_davis_palette_matches_pascal_voc():
    palette = davis_palette()
    assert len(palette) == 768
    assert palette[:12] == [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0]


def test_object_ids_follow_first_appearance(video_items):
    assert object_ids(video_items) == {("car", 1): 1, ("person", None): 2, ("car", 2): 3}


def test_too_many_objects_rejected(image_dir):
    shapes = [rect("o", 0, group_id=i) for i in range(256)]
    ann_io.save(Annotation("img1.png", 100, 80, shapes), image_dir / "img1.json")
    with pytest.raises(ValueError, match="255"):
        object_ids(export.collect(image_dir))


def test_export_davis_writes_consistent_indexed_pngs(video_items, image_dir, tmp_path):
    out = tmp_path / "davis"
    export.EXPORTERS["davis"](video_items, out)
    sequence = image_dir.name
    assert sorted(p.name for p in (out / "JPEGImages" / sequence).iterdir()) == [
        "img1.png",
        "img2.png",
    ]
    first = Image.open(out / "Annotations" / sequence / "img1.png")
    second = np.array(Image.open(out / "Annotations" / sequence / "img2.png"))
    assert first.mode == "P" and first.getpalette()[:6] == [0, 0, 0, 128, 0, 0]
    first = np.array(first)
    assert (first[15, 5], first[15, 45], first[75, 5]) == (1, 2, 0)
    assert (second[15, 7], second[15, 47], second[15, 75]) == (1, 2, 3)
    assert (out / "objects.txt").read_text().splitlines() == ["1 car 1", "2 person", "3 car 2"]


def test_export_davis_with_nothing_is_a_no_op(tmp_path):
    export.export_davis([], tmp_path / "out")
    assert not (tmp_path / "out").exists()
