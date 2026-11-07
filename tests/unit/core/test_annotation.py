import json

from easy_mask.core import annotation as ann_io
from easy_mask.core.annotation import Annotation


def test_save_load_roundtrip(tmp_path, square):
    annotation = Annotation("img.png", 100, 80, [square], {"blurry": True})
    path = tmp_path / "sub" / "img.json"
    ann_io.save(annotation, path)
    assert ann_io.load(path) == annotation
    assert not path.with_suffix(".json.tmp").exists()


def test_file_is_labelme_compatible(tmp_path, square):
    path = tmp_path / "img.json"
    ann_io.save(Annotation("img.png", 100, 80, [square]), path)
    data = json.loads(path.read_text())
    assert {
        "version",
        "flags",
        "shapes",
        "imagePath",
        "imageData",
        "imageHeight",
        "imageWidth",
    } <= set(data)
    assert data["shapes"][0]["shape_type"] == "polygon"


def test_loads_labelme_file_with_image_data(tmp_path):
    path = tmp_path / "x.json"
    path.write_text(
        json.dumps(
            {
                "version": "5.0.1",
                "flags": {},
                "shapes": [{"label": "dog", "points": [[1, 1], [5, 5]], "shape_type": "rectangle"}],
                "imagePath": "x.jpg",
                "imageData": "aGVsbG8=",
                "imageHeight": 10,
                "imageWidth": 10,
            }
        )
    )
    annotation = ann_io.load(path)
    assert annotation.shapes[0].label == "dog"


def test_annotation_path_for(tmp_path):
    image = tmp_path / "images" / "cat.jpg"
    assert ann_io.annotation_path_for(image) == tmp_path / "images" / "cat.json"
    assert ann_io.annotation_path_for(image, tmp_path / "out") == tmp_path / "out" / "cat.json"


def test_relative_image_path(tmp_path):
    image = tmp_path / "images" / "cat.jpg"
    image.parent.mkdir()
    image.touch()
    assert (
        ann_io.relative_image_path(image, tmp_path / "labels" / "cat.json") == "../images/cat.jpg"
    )
