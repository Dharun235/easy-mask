"""Export annotated folders to training-ready dataset formats.

Each exporter takes the same inputs (see :func:`collect`) and writes a self-contained
dataset folder with a copy of every annotated image.
"""

from easy_mask.io.export.coco import export_coco, export_coco_rle
from easy_mask.io.export.common import AnnotatedImage, Exporter, collect
from easy_mask.io.export.davis import export_davis
from easy_mask.io.export.masks import export_masks
from easy_mask.io.export.objects import PARTS, export_objects
from easy_mask.io.export.yolo import export_yolo

# "coco-rle" is COCO with mask shapes as exact RLE (CLI: `--format coco --rle`).
EXPORTERS: dict[str, Exporter] = {
    "masks": export_masks,
    "coco": export_coco,
    "coco-rle": export_coco_rle,
    "yolo": export_yolo,
    "davis": export_davis,
    "objects": export_objects,
}

# What the File > Export menu shows for each format.
EXPORT_NAMES = {
    "objects": "Objects (choose what to save below)",
    "masks": "Semantic + instance mask PNGs",
    "coco": "COCO (polygons)",
    "coco-rle": "COCO (exact RLE masks)",
    "yolo": "YOLO segmentation",
    "davis": "DAVIS video masks",
}

__all__ = [
    "EXPORTERS",
    "EXPORT_NAMES",
    "PARTS",
    "AnnotatedImage",
    "collect",
    "export_coco",
    "export_coco_rle",
    "export_davis",
    "export_masks",
    "export_objects",
    "export_yolo",
]
