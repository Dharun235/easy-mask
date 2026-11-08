"""Run a mask-propagation model over a run of frames and turn its output into shapes.

The model side is the small :class:`Propagator` protocol (frames + prompt masks in,
masks per frame out), so the GUI and tests can swap the real SAM 2-family backend for a
fake. :func:`track` does the bookkeeping around it: which shapes are prompts, object ids
<-> ``(label, group_id)``, and flagging every result as ``propagated``.
"""

from collections.abc import Generator, Iterator
from pathlib import Path
from typing import Protocol

import numpy as np

from easy_mask.core import masks
from easy_mask.core.shapes import Shape
from easy_mask.core.video import PROPAGATED_FLAG, ObjectKey, is_auto, object_masks

# frame position -> object id -> full-size boolean mask
FrameMasks = dict[int, dict[int, np.ndarray]]


class Propagator(Protocol):
    """A video object segmentation model with memory (SAM 2 and friends)."""

    def propagate(self, frames: list[Path], prompts: FrameMasks) -> Iterator[FrameMasks]:
        """Track prompted objects through ``frames``, in list order.

        Args:
            frames: Image files in tracking order (reverse them to track backwards).
            prompts: Known masks by frame position; position 0 is always present.

        Yields:
            ``{position: {object_id: mask}}`` for one frame at a time, in order, for every
            frame after position 0. Stopping the iteration early cancels the run.
        """
        ...


def track(
    propagator: Propagator,
    frames: list[Path],
    frame_shapes: list[list[Shape]],
    height: int,
    width: int,
) -> Generator[tuple[int, list[Shape]], None, None]:
    """Propagate the objects on ``frames[0]`` through the rest of ``frames``.

    Every area shape on the first frame is a prompt (polygons and boxes are rasterized).
    Hand-drawn shapes of the same objects on later frames are extra prompts, so a user's
    corrections steer the model.

    Args:
        propagator: The model.
        frames: Image files in tracking order.
        frame_shapes: The current shapes of each frame, same order as ``frames``.
        height: Frame height in pixels.
        width: Frame width in pixels.

    Yields:
        ``(position, shapes)``: the propagated mask shapes for each later frame.
    """
    start = object_masks(frame_shapes[0], height, width)
    keys: list[ObjectKey] = list(start)
    if not keys:
        return
    ids = {key: number for number, key in enumerate(keys, start=1)}
    prompts: FrameMasks = {0: {ids[key]: full for key, full in start.items()}}
    for position, shapes in enumerate(frame_shapes[1:], start=1):
        manual = [s for s in shapes if not is_auto(s)]
        known = {ids[k]: m for k, m in object_masks(manual, height, width).items() if k in ids}
        if known:
            prompts[position] = known

    for result in propagator.propagate(frames, prompts):
        for position, by_id in result.items():
            yield position, _to_shapes(by_id, keys)


def _to_shapes(by_id: dict[int, np.ndarray], keys: list[ObjectKey]) -> list[Shape]:
    shapes = []
    for object_id, full in sorted(by_id.items()):
        label, group_id = keys[object_id - 1]
        shape = masks.to_shape(label, full, group_id=group_id, flags={PROPAGATED_FLAG: True})
        if shape is not None:
            shapes.append(shape)
    return shapes
