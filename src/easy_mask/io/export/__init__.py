"""Export annotated folders to training-ready dataset formats.

Each exporter takes the same inputs (see :func:`collect`) and writes a self-contained
dataset folder with a copy of every annotated image.
"""

from easy_mask.io.export.coco import export_coco
from easy_mask.io.export.common import AnnotatedImage, collect
from easy_mask.io.export.davis import export_davis
from easy_mask.io.export.masks import export_masks
from easy_mask.io.export.yolo import export_yolo

EXPORTERS = {"masks": export_masks, "coco": export_coco, "yolo": export_yolo, "davis": export_davis}

__all__ = [
    "EXPORTERS",
    "AnnotatedImage",
    "collect",
    "export_coco",
    "export_davis",
    "export_masks",
    "export_yolo",
]
