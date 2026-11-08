"""Objects across video frames: identity, automatic vs. manual shapes, merging results.

A video is a folder of frames with one annotation per frame. An *object* is identified
across frames by ``(label, group_id)``; give objects that share a label different group
ids. Shapes created by interpolation or propagation carry a flag (``interpolated`` or
``propagated``) so users and tools can tell them from hand-drawn ones: automatic shapes
are replaced when an object is re-propagated, manual ones are never overwritten.
"""

from typing import Literal

import numpy as np

from easy_mask.core.operations import with_pixels
from easy_mask.core.rasterize import shape_to_mask
from easy_mask.core.shapes import Shape

ObjectKey = tuple[str, int | None]

PROPAGATED_FLAG = "propagated"
INTERPOLATED_FLAG = "interpolated"
AUTO_FLAGS = (PROPAGATED_FLAG, INTERPOLATED_FLAG)


def object_key(shape: Shape) -> ObjectKey:
    """The identity of the object a shape belongs to across frames."""
    return shape.label, shape.group_id


def is_auto(shape: Shape) -> bool:
    """True if the shape was made by interpolation or propagation, not by hand."""
    return any(shape.flags.get(flag) for flag in AUTO_FLAGS)


def confirm(shapes: list[Shape]) -> list[Shape]:
    """Copies of ``shapes`` with the automatic flags removed (the user vouches for them)."""
    confirmed = []
    for shape in shapes:
        copy = shape.copy()
        for flag in AUTO_FLAGS:
            copy.flags.pop(flag, None)
        confirmed.append(copy)
    return confirmed


def object_masks(shapes: list[Shape], height: int, width: int) -> dict[ObjectKey, np.ndarray]:
    """One full-size boolean mask per object: the union of its area shapes."""
    result: dict[ObjectKey, np.ndarray] = {}
    for shape in shapes:
        if shape.shape_type.is_area:
            full = shape_to_mask(shape, height, width)
            key = object_key(shape)
            result[key] = result[key] | full if key in result else full
    return result


def manual_objects(shapes: list[Shape]) -> set[ObjectKey]:
    """Objects that have at least one hand-drawn area shape."""
    return {object_key(s) for s in shapes if s.shape_type.is_area and not is_auto(s)}


def merge_auto(shapes: list[Shape], objects: set[ObjectKey], new: list[Shape]) -> list[Shape]:
    """Replace a frame's automatic shapes of ``objects`` with ``new``.

    Objects the user drew by hand on this frame are left alone, and ``new`` shapes for
    them are dropped. An object missing from ``new`` (e.g. it left the view) loses its old
    automatic shapes.
    """
    manual = manual_objects(shapes)
    replaced = objects - manual
    kept = [s for s in shapes if not (is_auto(s) and object_key(s) in replaced)]
    return kept + [s for s in new if object_key(s) in replaced]


def edit_object(
    shapes: list[Shape],
    key: ObjectKey,
    region: np.ndarray,
    height: int,
    width: int,
    op: Literal["add", "subtract"],
    template: Shape | None = None,
) -> list[Shape]:
    """Add ``region``'s pixels to object ``key``, or remove them from it.

    The object's area shapes become one mask shape (keeping the first one's label, group,
    note and flags) at the first one's position; if nothing is left, the object is
    removed. If the object is absent, ``add`` creates it from ``template`` (when given).
    Returns ``shapes`` unchanged when nothing would change.
    """
    targets = [i for i, s in enumerate(shapes) if s.shape_type.is_area and object_key(s) == key]
    if not targets:
        if op == "add" and template is not None and region.any():
            created = with_pixels(template, region)
            return [*shapes, created] if created is not None else shapes
        return shapes
    full = np.zeros((height, width), dtype=bool)
    for i in targets:
        full |= shape_to_mask(shapes[i], height, width)
    edited = full | region if op == "add" else full & ~region
    if np.array_equal(edited, full):
        return shapes
    replacement = with_pixels(shapes[targets[0]], edited)
    result: list[Shape] = []
    for i, shape in enumerate(shapes):
        if i == targets[0]:
            if replacement is not None:
                result.append(replacement)
        elif i not in targets:
            result.append(shape)
    return result
