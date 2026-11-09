"""COCO export: RLE vs traced polygons on real ground-truth masks.

Every ground-truth mask becomes one mask shape; the folder is exported with
``easy-mask``'s COCO exporter twice (polygons, then ``rle=True``). Each exported
segmentation is decoded back to pixels (RLE with :mod:`easy_mask.core.rle`, polygons by
rasterizing) and compared with the original mask. Reports mean pixel IoU, the share of
masks recovered exactly, and ``annotations.json`` size per object.

Usage::

    uv run python -m benchmarks.coco_rle [--limit N]
"""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from benchmarks.datasets import available
from easy_mask.core import masks, rle
from easy_mask.core.annotation import Annotation
from easy_mask.core.rasterize import shape_to_mask
from easy_mask.core.shapes import Shape, ShapeType
from easy_mask.io.export import AnnotatedImage, export_coco


def decode(segmentation: list | dict, height: int, width: int) -> np.ndarray:
    """Pixels of one COCO segmentation (RLE dict or list of flat polygons)."""
    if isinstance(segmentation, dict):
        return rle.decode(segmentation)
    full = np.zeros((height, width), dtype=bool)
    for flat in segmentation:
        points = list(zip(flat[::2], flat[1::2], strict=True))
        full |= shape_to_mask(Shape("", ShapeType.POLYGON, points), height, width)
    return full


def run(items: list[AnnotatedImage], truths: list[np.ndarray], use_rle: bool) -> dict[str, float]:
    """Export ``items`` once and score every decoded segmentation against its mask."""
    with tempfile.TemporaryDirectory() as tmp:
        export_coco(items, Path(tmp), rle=use_rle)
        text = (Path(tmp) / "annotations.json").read_text(encoding="utf-8")
    coco = json.loads(text)
    by_image = {a["image_id"]: a for a in coco["annotations"]}
    ious = []
    for image_id, truth in enumerate(truths, start=1):
        ann = by_image.get(image_id)
        got = np.zeros_like(truth) if ann is None else decode(ann["segmentation"], *truth.shape)
        ious.append(masks.iou(got, truth))
    return {
        "iou": float(np.mean(ious)),
        "exact": float(np.mean([iou == 1.0 for iou in ious])),
        "kb": len(text.encode()) / 1024 / len(truths),
        "compact_kb": len(json.dumps(coco).encode()) / 1024 / len(truths),
    }


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark and print a markdown table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, help="max images per dataset")
    args = parser.parse_args(argv)

    datasets = available()
    if not datasets:
        print("No datasets in data/benchmarks; see docs/benchmarks.md")
        return 1
    header = ["dataset", "n", "format", "mean IoU", "exact", "KB / object", "KB / object (compact)"]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for dataset in datasets:
        items, truths = [], []
        for image_path, truth in dataset.samples(limit=args.limit):
            shape = masks.to_shape("object", truth)
            assert shape is not None
            height, width = truth.shape
            annotation = Annotation(image_path.name, width, height, [shape])
            items.append(AnnotatedImage(image_path, annotation))
            truths.append(truth)
        for name, use_rle in (("polygon", False), ("RLE", True)):
            r = run(items, truths, use_rle)
            cells = [dataset.name, str(len(truths)), name, f"{r['iou']:.4f}", f"{r['exact']:.0%}"]
            cells += [f"{r['kb']:.1f}", f"{r['compact_kb']:.1f}"]
            print("| " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
