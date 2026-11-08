"""Export annotated folders to training-ready dataset formats.

Each exporter takes the same inputs (see :func:`collect`) and writes a self-contained
dataset folder with a copy of every annotated image.
"""

from easy_mask.io.export.coco import export_coco, export_coco_rle
from easy_mask.io.export.common import AnnotatedImage, Exporter, collect
from easy_mask.io.export.davis import export_davis
from easy_mask.io.export.masks import export_masks
from easy_mask.io.export.yolo import export_yolo

# "coco-rle" is COCO with mask shapes as exact RLE (CLI: `--format coco --rle`).
EXPORTERS: dict[str, Exporter] = {
    "masks": export_masks,
    "coco": export_coco,
    "coco-rle": export_coco_rle,
    "yolo": export_yolo,
    "davis": export_davis,
}

__all__ = [
    "EXPORTERS",
    "AnnotatedImage",
    "collect",
    "export_coco",
    "export_coco_rle",
    "export_davis",
    "export_masks",
    "export_yolo",
]
