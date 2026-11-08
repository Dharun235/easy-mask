"""Export a video (folder of frames) as a DAVIS-style object-segmentation sequence.

Output layout (``<seq>`` is the frame folder's name)::

    out/
      JPEGImages/<seq>/<stem>.<ext>   copies of the annotated frames
      Annotations/<seq>/<stem>.png    palette PNG, pixel value = object id (0 = background)
      objects.txt                     line ``id label group_id`` per object

Object ids are the same in every frame: each ``(label, group_id)`` gets the next id in
order of first appearance (frames in order, shapes in drawing order). The palette is the
PASCAL VOC / DAVIS colour map, so the PNGs open in the official DAVIS tools.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from easy_mask.core.rasterize import shape_to_mask
from easy_mask.core.video import ObjectKey, object_key
from easy_mask.io.export.common import AnnotatedImage, ProgressFn, copy_images, tracked

MAX_OBJECTS = 255


def davis_palette() -> list[int]:
    """The 256-colour PASCAL VOC palette used by DAVIS (flat RGB list)."""
    palette: list[int] = []
    for index in range(256):
        r = g = b = 0
        value = index
        for bit in range(8):
            r |= ((value >> 0) & 1) << (7 - bit)
            g |= ((value >> 1) & 1) << (7 - bit)
            b |= ((value >> 2) & 1) << (7 - bit)
            value >>= 3
        palette.extend((r, g, b))
    return palette


def object_ids(items: list[AnnotatedImage]) -> dict[ObjectKey, int]:
    """Assign ids 1, 2, ... to objects in order of first appearance across frames."""
    ids: dict[ObjectKey, int] = {}
    for item in items:
        for shape in item.annotation.shapes:
            if shape.shape_type.is_area:
                ids.setdefault(object_key(shape), len(ids) + 1)
    if len(ids) > MAX_OBJECTS:
        raise ValueError(f"DAVIS PNGs support at most {MAX_OBJECTS} objects, got {len(ids)}")
    return ids


def export_davis(
    items: list[AnnotatedImage],
    out_dir: Path,
    labels: list[str] | None = None,
    progress: ProgressFn | None = None,
) -> None:
    """Write DAVIS-style indexed PNGs for every annotated frame.

    Args:
        items: Annotated frames from :func:`easy_mask.io.export.collect`, in frame order.
        out_dir: Destination folder (created if missing).
        labels: Unused (object ids come from the frames); accepted so every exporter has
            the same signature.
        progress: Called with ``(images done, total)`` after each image.
    """
    if not items:
        return
    sequence = items[0].image_path.parent.name
    annotations_dir = out_dir / "Annotations" / sequence
    annotations_dir.mkdir(parents=True, exist_ok=True)
    copy_images(items, out_dir / "JPEGImages" / sequence)

    ids = object_ids(items)
    lines = [
        f"{i} {label} {'' if group is None else group}".rstrip()
        for (label, group), i in ids.items()
    ]
    (out_dir / "objects.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    palette = davis_palette()
    for item in tracked(items, progress):
        ann = item.annotation
        frame = np.zeros((ann.image_height, ann.image_width), dtype=np.uint8)
        for shape in ann.shapes:  # later shapes paint on top, as on the canvas
            if shape.shape_type.is_area:
                frame[shape_to_mask(shape, ann.image_height, ann.image_width)] = ids[
                    object_key(shape)
                ]
        image = Image.fromarray(frame, mode="P")
        image.putpalette(palette)
        image.save(annotations_dir / f"{item.image_path.stem}.png")
