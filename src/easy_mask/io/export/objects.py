"""Export every annotated object on its own: binary mask, cut-out pixels, coordinates.

Output layout (each part optional)::

    out/
      masks/<image>_<NN>_<label>.png    full-size binary mask: 255 = object, 0 = background
      cutouts/<image>_<NN>_<label>.png  the object's original pixels, cropped to its box,
                                        transparent outside the object (RGBA)
      objects.json                      one record per object (see below)
      objects.csv                       the same, one row per object

An *object* is every shape sharing ``(label, group_id)`` when ``group_id`` is set (one
object in several parts), otherwise a single shape. ``NN`` numbers objects per image in
drawing order. Each record has: image, object, label, group_id, shape_types, bbox
``[x_min, y_min, x_max, y_max]`` (inclusive pixels), area (pixels), polygons (outlines in
image coordinates; points / lines give their own points) and the mask / cutout file names.
"""

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from easy_mask.core import masks
from easy_mask.core.geometry import Point
from easy_mask.core.rasterize import shape_to_mask
from easy_mask.core.shapes import Shape
from easy_mask.io.export.common import AnnotatedImage, ProgressFn, tracked

PARTS = ("masks", "cutouts", "coordinates")
CSV_FIELDS = [
    "image",
    "object",
    "label",
    "group_id",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "area",
    "shape_types",
    "polygon",
    "mask_file",
    "cutout_file",
]


@dataclass
class _Object:
    label: str
    group_id: int | None
    shapes: list[Shape]


def group_objects(shapes: list[Shape]) -> list[_Object]:
    """Objects in drawing order: shapes sharing ``(label, group_id)`` merge into one."""
    objects: list[_Object] = []
    by_key: dict[tuple[str, int], _Object] = {}
    for shape in shapes:
        if shape.group_id is None:
            objects.append(_Object(shape.label, None, [shape]))
            continue
        key = (shape.label, shape.group_id)
        if key not in by_key:
            by_key[key] = _Object(shape.label, shape.group_id, [])
            objects.append(by_key[key])
        by_key[key].shapes.append(shape)
    return objects


def safe_name(text: str) -> str:
    """``text`` made safe for a file name (letters, digits, ``-`` and ``_``)."""
    return re.sub(r"[^A-Za-z0-9_-]+", "-", text).strip("-") or "object"


def _polygons(shape: Shape) -> list[list[Point]]:
    if shape.shape_type.is_area:
        return masks.shape_polygons(shape)
    return [list(shape.points)]  # points and lines: their own coordinates


def export_objects(
    items: list[AnnotatedImage],
    out_dir: Path,
    labels: list[str] | None = None,
    progress: ProgressFn | None = None,
    parts: tuple[str, ...] = PARTS,
) -> None:
    """Write the chosen ``parts`` for every object of every annotated image.

    Args:
        items: Annotated images from :func:`easy_mask.io.export.collect`.
        out_dir: Destination folder (created if missing).
        labels: Unused (every label is exported); accepted so every exporter has the same
            signature.
        progress: Called with ``(images done, total)`` after each image.
        parts: Any of ``"masks"``, ``"cutouts"``, ``"coordinates"``.

    Raises:
        ValueError: If ``parts`` is empty or names an unknown part.
    """
    unknown = set(parts) - set(PARTS)
    if not parts or unknown:
        raise ValueError(f"parts must be a non-empty subset of {PARTS}, got {parts}")
    for part in ("masks", "cutouts"):
        if part in parts:
            (out_dir / part).mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for item in tracked(items, progress):
        ann = item.annotation
        height, width = ann.image_height, ann.image_width
        rgb = None
        if "cutouts" in parts:
            with Image.open(item.image_path) as image:
                rgb = np.asarray(image.convert("RGB"))
        for number, obj in enumerate(group_objects(ann.shapes), start=1):
            full = np.zeros((height, width), dtype=bool)
            for shape in obj.shapes:
                full |= shape_to_mask(shape, height, width)
            ys, xs = np.nonzero(full)
            if len(xs) == 0:
                continue  # degenerate or off-image shape
            x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            name = f"{item.image_path.stem}_{number:02d}_{safe_name(obj.label)}.png"
            mask_file = cutout_file = ""
            if "masks" in parts:
                mask_file = f"masks/{name}"
                Image.fromarray(full.astype(np.uint8) * 255).save(out_dir / mask_file)
            if rgb is not None:
                cutout_file = f"cutouts/{name}"
                crop = np.zeros((y1 - y0 + 1, x1 - x0 + 1, 4), dtype=np.uint8)
                crop[..., :3] = rgb[y0 : y1 + 1, x0 : x1 + 1]
                crop[..., 3] = full[y0 : y1 + 1, x0 : x1 + 1] * 255
                Image.fromarray(crop, mode="RGBA").save(out_dir / cutout_file)
            records.append(
                {
                    "image": item.image_path.name,
                    "object": number,
                    "label": obj.label,
                    "group_id": obj.group_id,
                    "shape_types": sorted({str(s.shape_type) for s in obj.shapes}),
                    "bbox": [x0, y0, x1, y1],
                    "area": int(full.sum()),
                    "polygons": [
                        [[round(x, 2), round(y, 2)] for x, y in polygon]
                        for shape in obj.shapes
                        for polygon in _polygons(shape)
                    ],
                    "mask_file": mask_file,
                    "cutout_file": cutout_file,
                }
            )

    if "coordinates" in parts:
        (out_dir / "objects.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        with (out_dir / "objects.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for r in records:
                x0, y0, x1, y1 = r["bbox"]
                first = r["polygons"][0] if r["polygons"] else []
                writer.writerow(
                    {
                        "image": r["image"],
                        "object": r["object"],
                        "label": r["label"],
                        "group_id": "" if r["group_id"] is None else r["group_id"],
                        "x_min": x0,
                        "y_min": y0,
                        "x_max": x1,
                        "y_max": y1,
                        "area": r["area"],
                        "shape_types": " ".join(r["shape_types"]),
                        "polygon": " ".join(f"{x},{y}" for x, y in first),
                        "mask_file": r["mask_file"],
                        "cutout_file": r["cutout_file"],
                    }
                )
