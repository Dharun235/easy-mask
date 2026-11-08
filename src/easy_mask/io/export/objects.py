"""Export every annotated object on its own: binary mask, cut-out pixels, coordinates.

Output layout (each part optional)::

    out/
      masks/<image>_<NN>_<label>.png    full-size binary mask: 255 = object, 0 = background
      cutouts/<image>_<NN>_<label>.png  the object's original pixels, cropped to its box,
                                        transparent outside the object (RGBA)
      overlays/<image>.png              the frame with its masks and points drawn on
      objects.json                      one record per object (see below)
      objects.csv                       the same, one row per object

``objects.json`` / ``.csv`` are written when any of coordinates, centroid, corners or points
is chosen, with the fields of the chosen ones.

An *object* is every shape sharing ``(label, group_id)`` when ``group_id`` is set (one
object in several parts), otherwise a single shape. ``NN`` numbers objects per image in
drawing order. Each record has: image, object, label, group_id, shape_types, bbox
``[x_min, y_min, x_max, y_max]`` (inclusive pixels), area (pixels), polygons (outlines in
image coordinates; points / lines give their own points) and the mask / cutout file names.
*centroid* adds ``centroid`` ``[x, y]``, the mean of the object's pixels. *corners* adds
``box_corners`` (the box above as four points) and ``rotated_corners`` (the four corners of
the smallest rotated rectangle around the object); both start top-left and go clockwise.
When the object has key points (:mod:`easy_mask.core.keypoints`, possibly moved by hand),
the centroid and rotated corners are read from them instead of computed. *points* adds
``points``: ``{role: [x, y]}`` for every key point of the object.
"""

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from easy_mask.core import masks
from easy_mask.core.geometry import Point
from easy_mask.core.keypoints import (
    CENTROID,
    CORNERS,
    Corners,
    is_keypoint,
    object_keypoints,
    rotated_corners,
)
from easy_mask.core.labels import label_color
from easy_mask.core.rasterize import shape_to_mask
from easy_mask.core.shapes import Shape
from easy_mask.io.export.common import AnnotatedImage, ProgressFn, tracked

PARTS = ("masks", "cutouts", "coordinates", "centroid", "corners", "points", "overlays")
TABLE_PARTS = ("coordinates", "centroid", "corners", "points")  # in objects.json / .csv
OVERLAY_ALPHA = 0.45  # mask opacity on overlay images
POINT_RADIUS = 5
CSV_FIELDS = {
    None: ["image", "object", "label", "group_id"],
    "coordinates": ["x_min", "y_min", "x_max", "y_max", "area", "shape_types", "polygon"],
    "centroid": ["centroid_x", "centroid_y"],
    "corners": ["box_corners", "rotated_corners"],
    "points": ["points"],
}
FILE_FIELDS = ["mask_file", "cutout_file"]


@dataclass
class _Object:
    label: str
    group_id: int | None
    shapes: list[Shape]


def group_objects(shapes: list[Shape]) -> list[_Object]:
    """Objects in drawing order: shapes sharing ``(label, group_id)`` merge into one.

    Key points join the object they belong to (same label and group id) instead of
    being objects of their own; key points without such an object are dropped.
    """
    objects: list[_Object] = []
    by_key: dict[tuple[str, int], _Object] = {}
    for shape in shapes:
        if is_keypoint(shape):
            continue
        if shape.group_id is None:
            objects.append(_Object(shape.label, None, [shape]))
            continue
        key = (shape.label, shape.group_id)
        if key not in by_key:
            by_key[key] = _Object(shape.label, shape.group_id, [])
            objects.append(by_key[key])
        by_key[key].shapes.append(shape)
    for shape in shapes:
        if is_keypoint(shape):
            owner = next(
                (o for o in objects if (o.label, o.group_id) == (shape.label, shape.group_id)),
                None,
            )
            if owner is not None:
                owner.shapes.append(shape)
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
        labels: The label list, for the overlay colours (every label is exported).
        progress: Called with ``(images done, total)`` after each image.
        parts: Any of :data:`PARTS`.

    Raises:
        ValueError: If ``parts`` is empty or names an unknown part.
    """
    unknown = set(parts) - set(PARTS)
    if not parts or unknown:
        raise ValueError(f"parts must be a non-empty subset of {PARTS}, got {parts}")
    for part in ("masks", "cutouts", "overlays"):
        if part in parts:
            (out_dir / part).mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for item in tracked(items, progress):
        ann = item.annotation
        height, width = ann.image_height, ann.image_width
        rgb = None
        if "cutouts" in parts or "overlays" in parts:
            with Image.open(item.image_path) as image:
                rgb = np.asarray(image.convert("RGB"))
        drawn: list[tuple[str, np.ndarray, dict[str, tuple[float, float]]]] = []
        for number, obj in enumerate(group_objects(ann.shapes), start=1):
            own = [s for s in obj.shapes if not is_keypoint(s)]
            points = object_keypoints(obj.shapes)
            full = np.zeros((height, width), dtype=bool)
            for shape in own:
                full |= shape_to_mask(shape, height, width)
            ys, xs = np.nonzero(full)
            if len(xs) == 0:
                continue  # degenerate or off-image shape
            x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            name = f"{item.image_path.stem}_{number:02d}_{safe_name(obj.label)}.png"
            record: dict = {
                "image": item.image_path.name,
                "object": number,
                "label": obj.label,
                "group_id": obj.group_id,
            }
            if "coordinates" in parts:
                record["shape_types"] = sorted({str(s.shape_type) for s in own})
                record["bbox"] = [x0, y0, x1, y1]
                record["area"] = int(full.sum())
                record["polygons"] = [
                    [[round(x, 2), round(y, 2)] for x, y in polygon]
                    for shape in own
                    for polygon in _polygons(shape)
                ]
            if "centroid" in parts:
                cx, cy = points.get(CENTROID, (float(xs.mean()), float(ys.mean())))
                record["centroid"] = [round(cx, 2), round(cy, 2)]
            if "corners" in parts:
                record["box_corners"] = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
                if all(role in points for role in CORNERS):
                    record["rotated_corners"] = [
                        [round(points[r][0], 2), round(points[r][1], 2)] for r in CORNERS
                    ]
                else:
                    record["rotated_corners"] = rotated_corners(full)
            if "points" in parts:
                record["points"] = {
                    role: [round(x, 2), round(y, 2)] for role, (x, y) in points.items()
                }
            drawn.append((obj.label, full, points))
            record["mask_file"] = record["cutout_file"] = ""
            if "masks" in parts:
                record["mask_file"] = f"masks/{name}"
                Image.fromarray(full.astype(np.uint8) * 255).save(out_dir / record["mask_file"])
            if rgb is not None and "cutouts" in parts:
                record["cutout_file"] = f"cutouts/{name}"
                crop = np.zeros((y1 - y0 + 1, x1 - x0 + 1, 4), dtype=np.uint8)
                crop[..., :3] = rgb[y0 : y1 + 1, x0 : x1 + 1]
                crop[..., 3] = full[y0 : y1 + 1, x0 : x1 + 1] * 255
                Image.fromarray(crop, mode="RGBA").save(out_dir / record["cutout_file"])
            records.append(record)
        if rgb is not None and "overlays" in parts:
            overlay = draw_overlay(rgb, drawn, labels or [])
            overlay.save(out_dir / "overlays" / f"{item.image_path.stem}.png")

    tables = [part for part in TABLE_PARTS if part in parts]
    if tables:
        _write_tables(records, out_dir, tables)


def draw_overlay(
    rgb: np.ndarray,
    objects: list[tuple[str, np.ndarray, dict[str, tuple[float, float]]]],
    labels: list[str],
) -> Image.Image:
    """The frame with each object's mask tinted in its label colour and its points drawn.

    The centroid is a filled dot; corners are rings joined by the rotated box's outline.
    """
    canvas = rgb.astype(np.float32)
    for label, full, _points in objects:
        tint = np.array(label_color(label, labels), dtype=np.float32)
        canvas[full] = canvas[full] * (1 - OVERLAY_ALPHA) + tint * OVERLAY_ALPHA
    image = Image.fromarray(canvas.astype(np.uint8))
    draw = ImageDraw.Draw(image)
    r = POINT_RADIUS
    for label, _full, points in objects:
        colour = label_color(label, labels)
        corners = [points[c] for c in CORNERS if c in points]
        if len(corners) == len(CORNERS):
            draw.line([*corners, corners[0]], fill=colour, width=2)
        for role, (x, y) in points.items():
            box = (x - r, y - r, x + r, y + r)
            if role == CENTROID:
                draw.ellipse(box, fill=colour, outline=(255, 255, 255), width=2)
            else:
                draw.ellipse(box, fill=(255, 255, 255), outline=colour, width=3)
    return image


def _points(points: Corners) -> str:
    return " ".join(f"{x},{y}" for x, y in points)


def _write_tables(records: list[dict], out_dir: Path, tables: list[str]) -> None:
    """``objects.json`` with every record, ``objects.csv`` with one row per object."""
    (out_dir / "objects.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    fields = CSV_FIELDS[None] + [f for part in tables for f in CSV_FIELDS[part]] + FILE_FIELDS
    with (out_dir / "objects.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in records:
            row = {key: r[key] for key in (*CSV_FIELDS[None], *FILE_FIELDS)}
            row["group_id"] = "" if r["group_id"] is None else r["group_id"]
            if "coordinates" in tables:
                x0, y0, x1, y1 = r["bbox"]
                first = r["polygons"][0] if r["polygons"] else []
                row.update(
                    x_min=x0,
                    y_min=y0,
                    x_max=x1,
                    y_max=y1,
                    area=r["area"],
                    shape_types=" ".join(r["shape_types"]),
                    polygon=_points(first),
                )
            if "centroid" in tables:
                row["centroid_x"], row["centroid_y"] = r["centroid"]
            if "corners" in tables:
                row["box_corners"] = _points(r["box_corners"])
                row["rotated_corners"] = _points(r["rotated_corners"])
            if "points" in tables:
                row["points"] = " ".join(f"{k}:{x},{y}" for k, (x, y) in r["points"].items())
            writer.writerow(row)
