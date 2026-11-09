"""Interoperability with labelme files, using a checked-in sample in labelme 5.x format."""

import json
import shutil

from easy_mask.core import annotation as ann_io
from easy_mask.core.shapes import ShapeType


def test_reads_labelme_file(data_dir):
    ann = ann_io.load(data_dir / "labelme_v5_sample.json")
    assert (ann.image_width, ann.image_height, ann.image_path) == (160, 120, "scene.jpg")
    assert ann.flags == {"night": True}
    assert [s.shape_type for s in ann.shapes] == [
        ShapeType.POLYGON,
        ShapeType.RECTANGLE,
        ShapeType.CIRCLE,
        ShapeType.LINESTRIP,
        ShapeType.LINE,
        ShapeType.POINT,
    ]
    assert ann.shapes[1].description == "" and ann.shapes[1].flags == {"occluded": True}


def test_resave_keeps_every_shape_and_drops_image_data(data_dir, tmp_path):
    src = tmp_path / "scene.json"
    shutil.copy(data_dir / "labelme_v5_sample.json", src)
    ann_io.save(ann_io.load(src), src)

    original = json.loads((data_dir / "labelme_v5_sample.json").read_text())
    resaved = json.loads(src.read_text())
    assert resaved["imageData"] is None
    for before, after in zip(original["shapes"], resaved["shapes"], strict=True):
        assert after["label"] == before["label"]
        assert after["shape_type"] == before["shape_type"]
        assert after["points"] == [[float(x), float(y)] for x, y in before["points"]]
        assert after["group_id"] == before["group_id"]


def test_reads_labelme_mask_shape_with_0_255_png(tmp_path):
    """labelme writes mask crops as PNG; accept both 0/1 and 0/255 pixel values."""
    import base64
    import io

    import numpy as np
    from PIL import Image

    crop = np.zeros((4, 6), np.uint8)
    crop[1:3, 2:5] = 255
    buffer = io.BytesIO()
    Image.fromarray(crop).save(buffer, format="PNG")
    data = {
        "imagePath": "x.png",
        "imageHeight": 10,
        "imageWidth": 10,
        "flags": {},
        "shapes": [
            {
                "label": "m",
                "shape_type": "mask",
                "points": [[3, 2], [8, 5]],
                "mask": base64.b64encode(buffer.getvalue()).decode(),
                "group_id": None,
            }
        ],
    }
    path = tmp_path / "x.json"
    path.write_text(json.dumps(data))
    (shape,) = ann_io.load(path).shapes
    assert shape.shape_type == ShapeType.MASK
    assert shape.mask.sum() == 6 and shape.mask[1, 2]
