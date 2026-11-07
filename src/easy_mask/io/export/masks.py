"""Export PNG masks: one semantic (class) mask and one instance mask per image.

Output layout::

    out/
      images/              copies of the source images
      semantic/<stem>.png  palette PNG, pixel value = class index (0 = background)
      instance/<stem>.png  16-bit PNG, pixel value = instance id (0 = background)
      class_names.txt      line i = name of class index i
"""

from pathlib import Path

import numpy as np
from PIL import Image

from easy_mask.core.labels import label_color
from easy_mask.core.rasterize import instance_mask, semantic_mask
from easy_mask.io.export.common import AnnotatedImage, copy_images, resolve_labels

BACKGROUND = "_background_"


def _palette(class_names: list[str]) -> list[int]:
    """Flat RGB palette: black background, then each label's display colour."""
    palette = [0, 0, 0]
    for name in class_names[1:]:
        palette.extend(label_color(name))
    return palette


def export_masks(
    items: list[AnnotatedImage], out_dir: Path, labels: list[str] | None = None
) -> None:
    """Write semantic and instance mask PNGs for every annotated image.

    Args:
        items: Annotated images from :func:`easy_mask.io.export.collect`.
        out_dir: Destination folder (created if missing).
        labels: Class order; defaults to all labels seen, sorted. At most 255 classes.
    """
    class_names = [BACKGROUND, *resolve_labels(items, labels)]
    if len(class_names) > 256:
        raise ValueError("Semantic PNG masks support at most 255 classes")
    class_ids = {name: i for i, name in enumerate(class_names) if i > 0}

    (out_dir / "semantic").mkdir(parents=True, exist_ok=True)
    (out_dir / "instance").mkdir(parents=True, exist_ok=True)
    copy_images(items, out_dir / "images")
    (out_dir / "class_names.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")

    palette = _palette(class_names)
    for item in items:
        ann, stem = item.annotation, item.image_path.stem
        h, w = ann.image_height, ann.image_width

        semantic = Image.fromarray(semantic_mask(ann.shapes, h, w, class_ids), mode="P")
        semantic.putpalette(palette)
        semantic.save(out_dir / "semantic" / f"{stem}.png")

        instances, _ = instance_mask(ann.shapes, h, w)
        Image.fromarray(instances.astype(np.uint16)).save(out_dir / "instance" / f"{stem}.png")
