"""Objects across video frames: identity, automatic vs. manual shapes, merging results.

A video is a folder of frames with one annotation per frame. An *object* is identified
across frames by ``(label, group_id)``; new objects that share a label get different
group ids automatically (:func:`free_group_id`). Shapes created by interpolation or
propagation carry a flag (``interpolated`` or ``propagated``) so users and tools can tell
them from hand-drawn ones: automatic shapes are replaced when an object is re-propagated,
manual ones are never overwritten.
"""

import numpy as np

from easy_mask.core.operations import with_pixels
from easy_mask.core.rasterize import shape_to_mask
from easy_mask.core.shapes import Shape

ObjectKey = tuple[str, int | None]

PROPAGATED_FLAG = "propagated"
INTERPOLATED_FLAG = "interpolated"
AUTO_FLAGS = (PROPAGATED_FLAG, INTERPOLATED_FLAG)
ERASE_LABEL = "(erase)"  # a shape with this label is never kept: it cuts the masks under it


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


def cut_region(
    shapes: list[Shape],
    region: np.ndarray,
    height: int,
    width: int,
    keys: set[ObjectKey] | None = None,
) -> list[Shape]:
    """Remove ``region``'s pixels from every area shape under it (or only from ``keys``).

    Edited shapes become masks that keep their label, group, note and flags; a shape cut
    away completely is dropped. Returns ``shapes`` itself when nothing is under the region.
    """
    result: list[Shape] = []
    changed = False
    for shape in shapes:
        if not shape.shape_type.is_area or (keys is not None and object_key(shape) not in keys):
            result.append(shape)
            continue
        full = shape_to_mask(shape, height, width)
        if not (full & region).any():
            result.append(shape)
            continue
        changed = True
        edited = with_pixels(shape, full & ~region)
        if edited is not None:
            result.append(edited)
    return result if changed else shapes


def free_group_id(shapes: list[Shape], label: str) -> int | None:
    """A group id that makes a new ``label`` object distinct from those in ``shapes``.

    None while no area shape has ``label`` with no group id (labelme's default), else one
    more than the largest group id used with ``label``, so two objects with the same label
    are never mistaken for one when propagating or exporting.
    """
    same = [s.group_id for s in shapes if s.shape_type.is_area and s.label == label]
    if None not in same:
        return None
    return max((g for g in same if g is not None), default=0) + 1
