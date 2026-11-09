"""Export a YOLO segmentation dataset (Ultralytics format).

Output layout::

    out/
      images/<name>      copies of the source images
      labels/<stem>.txt  one line per object: ``class x1 y1 x2 y2 ...`` normalized to 0-1
      data.yaml          class names

Only area shapes (polygon, rectangle, circle, mask) are exported; a mask with
several blobs gives one line per blob.
"""

from pathlib import Path

from easy_mask.core import masks
from easy_mask.io.export.common import (
    AnnotatedImage,
    ProgressFn,
    copy_images,
    resolve_labels,
    tracked,
)


def export_yolo(
    items: list[AnnotatedImage],
    out_dir: Path,
    labels: list[str] | None = None,
    progress: ProgressFn | None = None,
) -> None:
    """Write YOLO-seg label files and ``data.yaml``.

    Args:
        items: Annotated images from :func:`easy_mask.io.export.collect`.
        out_dir: Destination folder (created if missing).
        labels: Class order; defaults to all labels seen, sorted.
        progress: Called with ``(images done, total)`` after each image.
    """
    class_names = resolve_labels(items, labels)
    class_ids = {name: i for i, name in enumerate(class_names)}

    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    copy_images(items, out_dir / "images")

    for item in tracked(items, progress):
        ann = item.annotation
        w, h = ann.image_width, ann.image_height
        lines = []
        for shape in ann.shapes:
            if not shape.shape_type.is_area or shape.label not in class_ids:
                continue
            for polygon in masks.shape_polygons(shape):
                coords = []
                for x, y in polygon:
                    coords += [min(max(x / w, 0.0), 1.0), min(max(y / h, 0.0), 1.0)]
                lines.append(" ".join([str(class_ids[shape.label])] + [f"{c:.6f}" for c in coords]))
        label_file = out_dir / "labels" / f"{item.image_path.stem}.txt"
        label_file.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(class_names))
    yaml = f"path: .\ntrain: images\nval: images\nnames:\n{names}\n"
    (out_dir / "data.yaml").write_text(yaml, encoding="utf-8")
