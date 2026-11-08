"""Edits that turn one list of shapes into another: combine, clean up, convert.

Every function is pure: it returns a new list and never mutates its input, so the GUI
can record undo history by simply keeping the old list.
"""

from collections.abc import Callable
from typing import Literal

import numpy as np

from easy_mask.core import masks
from easy_mask.core.rasterize import shape_to_mask
from easy_mask.core.shapes import Shape, ShapeType

CombineOp = Literal["union", "subtract", "intersect"]
MaskFn = Callable[[np.ndarray], np.ndarray]


def with_pixels(shape: Shape, full: np.ndarray) -> Shape | None:
    """A mask shape with ``full``'s pixels and ``shape``'s label, group, note and flags.

    Returns None if ``full`` is empty.
    """
    return masks.to_shape(
        shape.label,
        full,
        group_id=shape.group_id,
        description=shape.description,
        flags=dict(shape.flags),
    )


def combine(
    shapes: list[Shape], selection: list[int], op: CombineOp, height: int, width: int
) -> tuple[list[Shape], int | None]:
    """Boolean-combine selected shapes into one mask.

    The first selected shape is the *target*: it keeps its label and position in the list,
    and ``subtract`` removes the others from it. All inputs are replaced by the result.

    Returns:
        The new shape list and the index of the result (None if it came out empty).
    """
    target_index = selection[0]
    full = shape_to_mask(shapes[target_index], height, width)
    for index in selection[1:]:
        other = shape_to_mask(shapes[index], height, width)
        if op == "union":
            full |= other
        elif op == "subtract":
            full &= ~other
        else:
            full &= other
    result_shape = with_pixels(shapes[target_index], full)

    result: list[Shape] = []
    result_index = None
    for index, shape in enumerate(shapes):
        if index == target_index:
            if result_shape is not None:
                result_index = len(result)
                result.append(result_shape)
        elif index not in selection:
            result.append(shape)
    return result, result_index


def map_pixels(
    shapes: list[Shape], indices: list[int], fn: MaskFn, height: int, width: int
) -> list[Shape]:
    """Apply a mask operation (fill holes, grow, ...) to each selected area shape.

    Shapes become masks; any that end up empty are removed.
    """
    result: list[Shape] = []
    for index, shape in enumerate(shapes):
        if index in indices and shape.shape_type.is_area:
            changed = with_pixels(shape, fn(shape_to_mask(shape, height, width)))
            if changed is not None:
                result.append(changed)
        else:
            result.append(shape)
    return result


def to_masks(shapes: list[Shape], indices: list[int], height: int, width: int) -> list[Shape]:
    """Convert selected polygons / rectangles / circles to mask shapes."""
    return map_pixels(shapes, indices, lambda full: full, height, width)


def to_polygons(
    shapes: list[Shape], indices: list[int], height: int, width: int, tolerance: float = 1.0
) -> list[Shape]:
    """Convert selected mask shapes to polygons, one per blob (holes are dropped)."""
    result: list[Shape] = []
    for index, shape in enumerate(shapes):
        if index not in indices or shape.shape_type != ShapeType.MASK:
            result.append(shape)
            continue
        for polygon in masks.to_polygons(shape_to_mask(shape, height, width), tolerance):
            result.append(
                Shape(
                    shape.label,
                    ShapeType.POLYGON,
                    polygon,
                    group_id=shape.group_id,
                    description=shape.description,
                    flags=dict(shape.flags),
                )
            )
    return result
