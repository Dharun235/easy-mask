"""Export a COCO instance-segmentation dataset.

Output layout::

    out/
      images/            copies of the source images
      annotations.json   COCO format with polygon segmentations

Only area shapes (polygon, rectangle, circle) are exported. Shapes sharing a
``(label, group_id)`` become one COCO annotation with several polygons.
"""

import json
from pathlib import Path

from easy_mask.core import geometry
from easy_mask.core.shapes import Shape
from easy_mask.io.export.common import AnnotatedImage, copy_images, resolve_labels


def _coco_object(shapes: list[Shape]) -> dict:
    """Polygons, bbox and area for one object made of one or more area shapes."""
    outlines = [s.outline() for s in shapes]
    all_points = [p for outline in outlines for p in outline]
    x_min, y_min, x_max, y_max = geometry.bounding_box(all_points)
    return {
        "segmentation": [[c for x, y in outline for c in (x, y)] for outline in outlines],
        "area": sum(geometry.polygon_area(o) for o in outlines),
        "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
        "iscrowd": 0,
    }


def _group_objects(shapes: list[Shape]) -> list[list[Shape]]:
    """Group area shapes into objects by ``(label, group_id)``; ungrouped stay alone."""
    objects: list[list[Shape]] = []
    by_key: dict[tuple[str, int], list[Shape]] = {}
    for shape in shapes:
        if not shape.shape_type.is_area:
            continue
        if shape.group_id is None:
            objects.append([shape])
            continue
        key = (shape.label, shape.group_id)
        if key not in by_key:
            by_key[key] = []
            objects.append(by_key[key])
        by_key[key].append(shape)
    return objects


def export_coco(
    items: list[AnnotatedImage], out_dir: Path, labels: list[str] | None = None
) -> None:
    """Write ``annotations.json`` in COCO format plus an ``images/`` folder.

    Args:
        items: Annotated images from :func:`easy_mask.io.export.collect`.
        out_dir: Destination folder (created if missing).
        labels: Category order; defaults to all labels seen, sorted.
    """
    class_names = resolve_labels(items, labels)
    category_ids = {name: i + 1 for i, name in enumerate(class_names)}
    coco: dict = {
        "info": {"description": "Exported by easy-mask"},
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": n, "supercategory": ""} for n, i in category_ids.items()],
    }

    for image_id, item in enumerate(items, start=1):
        ann = item.annotation
        coco["images"].append(
            {
                "id": image_id,
                "file_name": f"images/{item.image_path.name}",
                "width": ann.image_width,
                "height": ann.image_height,
            }
        )
        for obj in _group_objects(ann.shapes):
            category_id = category_ids.get(obj[0].label)
            if category_id is None:
                continue
            coco["annotations"].append(
                {
                    "id": len(coco["annotations"]) + 1,
                    "image_id": image_id,
                    "category_id": category_id,
                    **_coco_object(obj),
                }
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    copy_images(items, out_dir / "images")
    (out_dir / "annotations.json").write_text(json.dumps(coco, indent=2), encoding="utf-8")
