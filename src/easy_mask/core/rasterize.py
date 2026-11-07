"""Turn vector shapes into pixel masks.

This is the heart of "easy-mask": whatever the user draws ends up as a numpy array that a
segmentation model can train on.
"""

import numpy as np
from PIL import Image, ImageDraw

from easy_mask.core.shapes import Shape, ShapeType


def shape_to_mask(
    shape: Shape, height: int, width: int, line_width: int = 3, point_radius: int = 3
) -> np.ndarray:
    """Rasterize one shape into a boolean ``(height, width)`` mask.

    Area shapes are filled. Lines and line strips are drawn ``line_width`` pixels thick
    and points as discs of ``point_radius``, so every shape type yields some pixels.
    """
    canvas = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    points = [(float(x), float(y)) for x, y in shape.points]

    if shape.shape_type == ShapeType.CIRCLE:
        (cx, cy), r = points[0], shape.radius()
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=1, outline=1)
    elif shape.shape_type.is_area:
        draw.polygon(shape.outline(), fill=1, outline=1)
    elif shape.shape_type in (ShapeType.LINE, ShapeType.LINESTRIP):
        draw.line(points, fill=1, width=line_width)
    elif shape.shape_type == ShapeType.POINT:
        (x, y), r = points[0], point_radius
        draw.ellipse([x - r, y - r, x + r, y + r], fill=1, outline=1)

    return np.asarray(canvas, dtype=bool)


def semantic_mask(
    shapes: list[Shape], height: int, width: int, class_ids: dict[str, int]
) -> np.ndarray:
    """Build a class-index mask (``uint8``, 0 = background).

    Shapes are painted in order, so later shapes overwrite earlier ones where they overlap.
    Shapes whose label is not in ``class_ids`` are skipped.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    for shape in shapes:
        class_id = class_ids.get(shape.label)
        if class_id is not None:
            mask[shape_to_mask(shape, height, width)] = class_id
    return mask


def instance_mask(
    shapes: list[Shape], height: int, width: int
) -> tuple[np.ndarray, list[tuple[int, str]]]:
    """Build an instance-id mask (``int32``, 0 = background).

    Shapes sharing the same ``(label, group_id)`` with a non-null ``group_id`` form one
    instance (useful for objects split by occlusion). Every other shape is its own instance.

    Returns:
        The mask and a list of ``(instance_id, label)`` in id order.
    """
    mask = np.zeros((height, width), dtype=np.int32)
    instances: list[tuple[int, str]] = []
    group_to_id: dict[tuple[str, int], int] = {}
    for shape in shapes:
        key = (shape.label, shape.group_id) if shape.group_id is not None else None
        if key is not None and key in group_to_id:
            instance_id = group_to_id[key]
        else:
            instance_id = len(instances) + 1
            instances.append((instance_id, shape.label))
            if key is not None:
                group_to_id[key] = instance_id
        mask[shape_to_mask(shape, height, width)] = instance_id
    return mask, instances
