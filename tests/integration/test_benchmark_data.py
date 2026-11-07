"""Run the pipeline on the real interactive-segmentation benchmarks.

Needs the datasets in ``data/benchmarks`` (see docs/benchmarks.md); skipped otherwise,
e.g. in CI. Real data catches what synthetic images don't: odd sizes, ``.JPG`` and
``.bmp`` files, grayscale images, thin structures.

Run just these with ``uv run pytest -m benchmark``.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from PySide6.QtGui import QImage

from easy_mask.core import annotation as ann_io
from easy_mask.core.annotation import Annotation
from easy_mask.core.shapes import Shape, ShapeType
from easy_mask.io import export
from easy_mask.io.images import list_images

ROOT = Path(__file__).parents[2] / "data" / "benchmarks"

pytestmark = [pytest.mark.benchmark, pytest.mark.slow]


@dataclass(frozen=True)
class Dataset:
    name: str
    images: str  # folder of images, relative to ROOT
    masks: str  # folder of ground-truth masks, relative to ROOT
    mask_suffix: str
    expected_count: int


DATASETS = [
    Dataset("GrabCut", "GrabCut/GrabCut/data_GT", "GrabCut/GrabCut/boundary_GT", ".bmp", 50),
    Dataset("Berkeley", "Berkeley/Berkeley/images", "Berkeley/Berkeley/masks", ".png", 100),
    Dataset("DAVIS", "DAVIS/DAVIS345/img", "DAVIS/DAVIS345/gt", ".png", 345),
]


def foreground(mask_path: Path) -> np.ndarray:
    """Ground-truth object pixels, whatever the dataset's encoding.

    GrabCut: 0 / 128 (uncertain boundary band, excluded) / 255. Berkeley: RGB 0 / 255.
    DAVIS: palette image of object ids (0 = background); any id counts, as in RITM.
    """
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask > 128 if mask.max() == 255 else mask > 0


@pytest.fixture(params=DATASETS, ids=lambda d: d.name)
def dataset(request) -> Dataset:
    ds = request.param
    if not (ROOT / ds.images).is_dir():
        pytest.skip(f"{ds.name} not downloaded to {ROOT} (see docs/benchmarks.md)")
    return ds


def test_every_image_is_found_and_has_a_mask(dataset):
    images = list_images(ROOT / dataset.images)
    assert len(images) == dataset.expected_count
    missing = [
        p.name
        for p in images
        if not (ROOT / dataset.masks / (p.stem + dataset.mask_suffix)).exists()
    ]
    assert missing == []


def test_qt_and_pillow_agree_on_image_size(dataset, qapp):
    """The GUI (Qt) and exporters (Pillow) must see the same pixel grid."""
    mismatched = []
    for path in list_images(ROOT / dataset.images):
        qimage = QImage(str(path))
        assert not qimage.isNull(), f"Qt cannot read {path.name}"
        with Image.open(path) as image:
            if (qimage.width(), qimage.height()) != image.size:
                mismatched.append(path.name)
    assert mismatched == []


def test_bbox_annotations_export_correctly(dataset, tmp_path):
    """Box every ground-truth object, export, and check COCO and mask outputs line up."""
    images = list_images(ROOT / dataset.images)[:10]
    ann_dir = tmp_path / "labels"
    boxes = {}
    for path in images:
        fg = foreground(ROOT / dataset.masks / (path.stem + dataset.mask_suffix))
        ys, xs = np.nonzero(fg)
        box = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
        boxes[path.name] = box
        height, width = fg.shape
        shape = Shape("object", ShapeType.RECTANGLE, [box[:2], box[2:]])
        ann_io.save(Annotation(path.name, width, height, [shape]), ann_dir / f"{path.stem}.json")

    items = export.collect(ROOT / dataset.images, ann_dir)
    assert len(items) == len(images)
    export.export_coco(items, tmp_path / "coco")
    export.export_masks(items, tmp_path / "masks")

    coco = json.loads((tmp_path / "coco" / "annotations.json").read_text())
    by_id = {img["id"]: img for img in coco["images"]}
    for ann in coco["annotations"]:
        name = Path(by_id[ann["image_id"]]["file_name"]).name
        x0, y0, x1, y1 = boxes[name]
        assert ann["bbox"] == pytest.approx([x0, y0, x1 - x0, y1 - y0])

    for item in items:
        semantic = Image.open(tmp_path / "masks" / "semantic" / f"{item.image_path.stem}.png")
        with Image.open(item.image_path) as image:
            assert semantic.size == image.size


def test_box_iou_ceiling_is_reported(dataset):
    """Sanity check on the data: a tight box covers its object but is a poor mask.

    This is why easy-mask focuses on masks: on these benchmarks a perfect box reaches
    well under 90% IoU, the NoC@90 target (see docs/benchmarks.md).
    """
    ious = []
    for path in list_images(ROOT / dataset.images)[:20]:
        fg = foreground(ROOT / dataset.masks / (path.stem + dataset.mask_suffix))
        ys, xs = np.nonzero(fg)
        box = np.zeros_like(fg)
        box[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1] = True
        ious.append((fg & box).sum() / (fg | box).sum())
    assert 0.2 < float(np.mean(ious)) < 0.9
