"""Export annotated objects: binary masks, original pixels, overlays, corners and points.

Choose any of :data:`PARTS`:

* ``masks``: binary mask, 255 = object, 0 = background.
* ``cutouts``: the object's original pixels, transparent (PNG) or black (video) elsewhere.
* ``overlays``: the frame with its masks and points drawn on.
* ``corners``: box corners, rotated-box corners and centroid, in ``objects.json`` / ``.csv``.
* ``points``: every key point as placed or moved by hand, in the same tables.

Images go out as PNG frames or as MP4 videos (``video=True``), one file per object or, with
``combine=True``, all objects together in one image per frame::

    out/
      masks/<image>_<NN>_<label>.png    per object (cutouts are cropped to the object)
      masks/<image>.png                 combine: every object of the frame in one image
      masks.mp4, cutouts.mp4            video + combine
      masks/<label>_<NN>.mp4            video, per object (black where it is absent)
      overlays/<image>.png, overlays.mp4
      objects.json, objects.csv         one record per object per frame

An *object* is every shape sharing ``(label, group_id)`` when ``group_id`` is set,
otherwise a single shape. ``NN`` numbers objects per image in drawing order (in videos,
the n-th object with that label and group). ``only_labels`` keeps just those labels.
Centroid and rotated corners come from the object's key points
(:mod:`easy_mask.core.keypoints`) when it has them, else they are computed. Corners start
top-left and go clockwise.
"""

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from easy_mask.core.keypoints import CENTROID, CORNERS, Corners, is_keypoint, object_keypoints
from easy_mask.core.keypoints import rotated_corners as compute_rotated_corners
from easy_mask.core.labels import label_color
from easy_mask.core.rasterize import shape_to_mask
from easy_mask.core.shapes import Shape
from easy_mask.io.export.common import AnnotatedImage, ProgressFn, tracked

PARTS = ("masks", "cutouts", "overlays", "corners", "points")
IMAGE_PARTS = ("masks", "cutouts", "overlays")
TABLE_PARTS = ("corners", "points")  # written to objects.json / .csv
FPS = 25
OVERLAY_ALPHA = 0.45  # mask opacity on overlay images
POINT_RADIUS = 5
CSV_FIELDS = {
    None: ["image", "object", "label", "group_id"],
    "corners": ["centroid_x", "centroid_y", "box_corners", "rotated_corners"],
    "points": ["points"],
}
FILE_FIELDS = ["mask_file", "cutout_file"]
Points = dict[str, tuple[float, float]]


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


def _video_keys(objects: list[_Object]) -> list[tuple[str, int | None, int]]:
    """A stable name for each object across frames: label, group id, n-th of those."""
    seen: dict[tuple[str, int | None], int] = {}
    keys = []
    for obj in objects:
        n = seen.get((obj.label, obj.group_id), 0) + 1
        seen[(obj.label, obj.group_id)] = n
        keys.append((obj.label, obj.group_id, n))
    return keys


def _video_name(key: tuple[str, int | None, int]) -> str:
    label, group_id, n = key
    group = "" if group_id is None else f"_g{group_id}"
    return f"{safe_name(label)}{group}_{n:02d}.mp4"


class _Videos:
    """MP4 writers by file name; every video gets one frame per exported image."""

    def __init__(self, out_dir: Path, size: tuple[int, int]):
        self.out_dir = out_dir
        self.width, self.height = (size[0] + 1) // 2 * 2, (size[1] + 1) // 2 * 2  # even
        self.writers: dict[str, cv2.VideoWriter] = {}

    def write(self, name: str, rgb: np.ndarray) -> None:
        if name not in self.writers:
            path = self.out_dir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            self.writers[name] = _open_writer(path, (self.width, self.height))
        frame = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        self.writers[name].write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        for writer in self.writers.values():
            writer.release()


def _open_writer(path: Path, size: tuple[int, int]) -> cv2.VideoWriter:
    """H.264 (plays everywhere) where OpenCV has it, else MPEG-4."""
    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*codec), FPS, size)
        if writer.isOpened():
            return writer
    raise OSError(f"Could not write the video {path}")


def export_objects(
    items: list[AnnotatedImage],
    out_dir: Path,
    labels: list[str] | None = None,
    progress: ProgressFn | None = None,
    parts: tuple[str, ...] = PARTS,
    only_labels: set[str] | None = None,
    combine: bool = False,
    video: bool = False,
) -> None:
    """Write the chosen ``parts`` for the objects of every annotated image.

    Args:
        items: Annotated images from :func:`easy_mask.io.export.collect`, in frame order.
        out_dir: Destination folder (created if missing).
        labels: The label list, for the overlay colours.
        progress: Called with ``(images done, total)`` after each image.
        parts: Any of :data:`PARTS`.
        only_labels: Export only objects with these labels (None: all).
        combine: All objects of a frame in one mask / cutout image instead of one each.
        video: Write MP4 videos instead of PNG frames.

    Raises:
        ValueError: If ``parts`` is empty or names an unknown part.
    """
    unknown = set(parts) - set(PARTS)
    if not parts or unknown:
        raise ValueError(f"parts must be a non-empty subset of {PARTS}, got {parts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    if not video:
        for part in IMAGE_PARTS:
            if part in parts:
                (out_dir / part).mkdir(exist_ok=True)
    videos = None
    all_keys: dict[tuple[str, int | None, int], None] = {}
    if video and items:
        first = items[0].annotation
        videos = _Videos(out_dir, (first.image_width, first.image_height))
        if not combine:  # one video per object: every object needs a frame in every image
            all_keys = dict.fromkeys(
                key
                for item in items
                for key in _video_keys([o for _, o in _wanted(item.annotation.shapes, only_labels)])
            )

    records: list[dict] = []
    try:
        for item in tracked(items, progress):
            ann = item.annotation
            height, width = ann.image_height, ann.image_width
            rgb = None
            if "cutouts" in parts or "overlays" in parts:
                with Image.open(item.image_path) as image:
                    rgb = np.asarray(image.convert("RGB"))
            objects = _wanted(ann.shapes, only_labels)
            stem = item.image_path.stem
            drawn: list[tuple[str, np.ndarray, Points]] = []
            frame_masks: dict[tuple[str, int | None, int], np.ndarray] = {}
            union = np.zeros((height, width), dtype=bool)
            keys = _video_keys([obj for _, obj in objects])
            for (number, obj), key in zip(objects, keys, strict=True):
                points = object_keypoints(obj.shapes)
                full = np.zeros((height, width), dtype=bool)
                for shape in obj.shapes:
                    if not is_keypoint(shape):
                        full |= shape_to_mask(shape, height, width)
                if not full.any():
                    continue  # degenerate or off-image shape
                union |= full
                frame_masks[key] = full
                drawn.append((obj.label, full, points))
                name = f"{stem}_{number:02d}_{safe_name(obj.label)}.png"
                record = _record(item, number, obj, full, points, parts)
                if "masks" in parts:
                    record["mask_file"] = _mask_file(name, stem, key, combine, video)
                if "cutouts" in parts:
                    record["cutout_file"] = _mask_file(name, stem, key, combine, video).replace(
                        "masks", "cutouts", 1
                    )
                records.append(record)
                if not video and not combine:
                    if "masks" in parts:
                        _save_mask(full, out_dir / "masks" / name)
                    if rgb is not None and "cutouts" in parts:
                        _save_cutout(rgb, full, out_dir / "cutouts" / name, crop=True)

            if not video and combine:
                if "masks" in parts:
                    _save_mask(union, out_dir / "masks" / f"{stem}.png")
                if rgb is not None and "cutouts" in parts:
                    _save_cutout(rgb, union, out_dir / "cutouts" / f"{stem}.png", crop=False)
            overlay = None
            if rgb is not None and "overlays" in parts:
                overlay = draw_overlay(rgb, drawn, labels or [])
                if not video:
                    overlay.save(out_dir / "overlays" / f"{stem}.png")
            if videos is not None:
                _write_video_frame(
                    videos, parts, combine, rgb, union, frame_masks, overlay, all_keys
                )
    finally:
        if videos is not None:
            videos.close()

    tables = [part for part in TABLE_PARTS if part in parts]
    if tables:
        _write_tables(records, out_dir, tables)


def _wanted(shapes: list[Shape], only_labels: set[str] | None) -> list[tuple[int, _Object]]:
    """``(number, object)`` pairs, numbered in drawing order before filtering by label."""
    numbered = enumerate(group_objects(shapes), start=1)
    return [(n, o) for n, o in numbered if only_labels is None or o.label in only_labels]


def _mask_file(
    name: str, stem: str, key: tuple[str, int | None, int], combine: bool, video: bool
) -> str:
    """Where an object's mask went (cutouts: the same with ``cutouts``)."""
    if video:
        return "masks.mp4" if combine else f"masks/{_video_name(key)}"
    return f"masks/{stem}.png" if combine else f"masks/{name}"


def _record(
    item: AnnotatedImage,
    number: int,
    obj: _Object,
    full: np.ndarray,
    points: Points,
    parts: tuple[str, ...],
) -> dict:
    record: dict = {
        "image": item.image_path.name,
        "object": number,
        "label": obj.label,
        "group_id": obj.group_id,
        "mask_file": "",
        "cutout_file": "",
    }
    if "corners" in parts:
        ys, xs = np.nonzero(full)
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        cx, cy = points.get(CENTROID, (float(xs.mean()), float(ys.mean())))
        record["centroid"] = [round(cx, 2), round(cy, 2)]
        record["box_corners"] = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        if all(role in points for role in CORNERS):
            record["rotated_corners"] = [
                [round(points[r][0], 2), round(points[r][1], 2)] for r in CORNERS
            ]
        else:
            record["rotated_corners"] = compute_rotated_corners(full)
    if "points" in parts:
        record["points"] = {role: [round(x, 2), round(y, 2)] for role, (x, y) in points.items()}
    return record


def _save_mask(full: np.ndarray, path: Path) -> None:
    Image.fromarray(full.astype(np.uint8) * 255).save(path)


def _save_cutout(rgb: np.ndarray, full: np.ndarray, path: Path, crop: bool) -> None:
    """The original pixels under ``full``, transparent elsewhere; cropped to it if ``crop``."""
    rgba = np.zeros((*full.shape, 4), dtype=np.uint8)
    rgba[..., :3] = rgb
    rgba[..., 3] = full * 255
    if crop:
        ys, xs = np.nonzero(full)
        rgba = rgba[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    Image.fromarray(rgba, mode="RGBA").save(path)


def _write_video_frame(
    videos: _Videos,
    parts: tuple[str, ...],
    combine: bool,
    rgb: np.ndarray | None,
    union: np.ndarray,
    frame_masks: dict[tuple[str, int | None, int], np.ndarray],
    overlay: Image.Image | None,
    all_keys: dict[tuple[str, int | None, int], None],
) -> None:
    """Add this image to every video: masks white on black, cutouts on black."""

    def as_mask(full: np.ndarray) -> np.ndarray:
        return np.repeat(full[..., None].astype(np.uint8) * 255, 3, axis=2)

    def as_cutout(full: np.ndarray) -> np.ndarray:
        assert rgb is not None
        return np.where(full[..., None], rgb, 0).astype(np.uint8)

    if combine:
        if "masks" in parts:
            videos.write("masks.mp4", as_mask(union))
        if rgb is not None and "cutouts" in parts:
            videos.write("cutouts.mp4", as_cutout(union))
    else:
        empty = np.zeros_like(union)
        for key in all_keys:
            full = frame_masks.get(key, empty)
            if "masks" in parts:
                videos.write(f"masks/{_video_name(key)}", as_mask(full))
            if rgb is not None and "cutouts" in parts:
                videos.write(f"cutouts/{_video_name(key)}", as_cutout(full))
    if overlay is not None:
        videos.write("overlays.mp4", np.asarray(overlay))


def draw_overlay(
    rgb: np.ndarray, objects: list[tuple[str, np.ndarray, Points]], labels: list[str]
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
            if "corners" in tables:
                row["centroid_x"], row["centroid_y"] = r["centroid"]
                row["box_corners"] = _points(r["box_corners"])
                row["rotated_corners"] = _points(r["rotated_corners"])
            if "points" in tables:
                row["points"] = " ".join(f"{k}:{x},{y}" for k, (x, y) in r["points"].items())
            writer.writerow(row)
