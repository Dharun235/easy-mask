"""End-to-end data pipeline without the GUI: annotate -> save -> export -> re-read.

The formats are checked against each other, so a bug in one exporter shows up as a
disagreement with the others.
"""

import json

import numpy as np
from PIL import Image, ImageDraw

from easy_mask.cli import main
from easy_mask.core import annotation as ann_io
from easy_mask.core.annotation import Annotation
from easy_mask.core.shapes import Shape, ShapeType


def annotate(image_dir):
    shapes = [
        Shape("cat", ShapeType.POLYGON, [(10, 10), (40, 12), (35, 50), (8, 45)]),
        Shape("dog", ShapeType.CIRCLE, [(70, 40), (80, 40)]),
        Shape("dog", ShapeType.RECTANGLE, [(50, 60), (95, 75)]),
    ]
    ann_io.save(Annotation("img1.png", 100, 80, shapes), image_dir / "img1.json")
    ann_io.save(Annotation("img2.png", 100, 80, shapes[:1]), image_dir / "img2.json")


def rasterize_polygons(polygons, width, height):
    canvas = Image.new("1", (width, height))
    draw = ImageDraw.Draw(canvas)
    for flat in polygons:
        draw.polygon(list(zip(flat[::2], flat[1::2], strict=True)), fill=1, outline=1)
    return np.asarray(canvas, dtype=bool)


def iou(a, b):
    return (a & b).sum() / (a | b).sum()


def test_all_formats_agree(image_dir, tmp_path):
    annotate(image_dir)
    for fmt in ("masks", "coco", "yolo"):
        assert main(["export", str(image_dir), "--format", fmt, "--out", str(tmp_path / fmt)]) == 0

    class_names = (tmp_path / "masks" / "class_names.txt").read_text().split()
    semantic = np.array(Image.open(tmp_path / "masks" / "semantic" / "img1.png"))
    coco = json.loads((tmp_path / "coco" / "annotations.json").read_text())
    yolo_lines = (tmp_path / "yolo" / "labels" / "img1.txt").read_text().splitlines()
    yolo_names = [n for n in class_names[1:]]

    for name in ("cat", "dog"):
        from_masks = semantic == class_names.index(name)
        cat_id = next(c["id"] for c in coco["categories"] if c["name"] == name)
        polys = [
            p
            for a in coco["annotations"]
            if a["image_id"] == 1 and a["category_id"] == cat_id
            for p in a["segmentation"]
        ]
        from_coco = rasterize_polygons(polys, 100, 80)
        yolo_polys = [
            [float(v) * (100 if i % 2 == 0 else 80) for i, v in enumerate(line.split()[1:])]
            for line in yolo_lines
            if int(line.split()[0]) == yolo_names.index(name)
        ]
        from_yolo = rasterize_polygons(yolo_polys, 100, 80)
        # Circles become 64-gons in COCO/YOLO, so allow a little disagreement.
        assert iou(from_masks, from_coco) > 0.97, name
        assert iou(from_coco, from_yolo) > 0.99, name


def test_every_annotated_image_is_exported_once(image_dir, tmp_path):
    annotate(image_dir)
    main(["export", str(image_dir), "--format", "coco", "--out", str(tmp_path / "c")])
    coco = json.loads((tmp_path / "c" / "annotations.json").read_text())
    assert sorted(i["file_name"] for i in coco["images"]) == ["images/img1.png", "images/img2.png"]
    assert sorted(p.name for p in (tmp_path / "c" / "images").iterdir()) == ["img1.png", "img2.png"]


def test_separate_annotation_folder(image_dir, tmp_path):
    labels = tmp_path / "labels"
    ann_io.save(
        Annotation(
            "../images/img1.png", 100, 80, [Shape("cat", ShapeType.RECTANGLE, [(0, 0), (9, 9)])]
        ),
        labels / "img1.json",
    )
    out = tmp_path / "out"
    args = [
        "export",
        str(image_dir),
        "--format",
        "yolo",
        "--out",
        str(out),
        "--annotations",
        str(labels),
    ]
    assert main(args) == 0
    assert (out / "labels" / "img1.txt").read_text().startswith("0 ")


def test_labels_file_controls_class_order(image_dir, tmp_path):
    annotate(image_dir)
    order = tmp_path / "labels.txt"
    order.write_text("dog\ncat\n")
    main(
        [
            "export",
            str(image_dir),
            "--format",
            "masks",
            "--out",
            str(tmp_path / "m"),
            "--labels",
            str(order),
        ]
    )
    assert (tmp_path / "m" / "class_names.txt").read_text().split() == [
        "_background_",
        "dog",
        "cat",
    ]
