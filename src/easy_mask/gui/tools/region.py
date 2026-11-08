"""Magic wand: click to select a region of similar colour.

Like every region tool (see :func:`combine_with_selection`): a plain click makes a new mask,
Shift adds to the selected shape, Alt subtracts from it. For a box around an object, use AI
Segment (drag a box), which is faster and more accurate than classic GrabCut.
"""

from typing import TYPE_CHECKING

from PySide6.QtGui import QInputEvent, QKeyEvent

from easy_mask.core import pixel_ops
from easy_mask.core.geometry import Point
from easy_mask.gui.tools.base import CanvasTool, combine_with_selection
from easy_mask.gui.tools.brush import StrokeCallback

if TYPE_CHECKING:
    from easy_mask.gui.canvas import Canvas

PREVIEW_RGB = (0, 200, 255)


class MagicWandTool(CanvasTool):
    """Click to flood-fill a region of similar colour. ``[`` / ``]`` change the tolerance."""

    def __init__(self, on_result: StrokeCallback, tolerance: int = 32):
        self.on_result = on_result
        self.tolerance = tolerance

    def press(self, canvas: "Canvas", p: Point, event: QInputEvent) -> None:
        """Select the region around the clicked pixel."""
        region = pixel_ops.magic_wand(canvas.image_rgb(), p, self.tolerance)
        edit = combine_with_selection(canvas, region, event.modifiers())
        if edit is not None:
            self.on_result(*edit)

    def key(self, canvas: "Canvas", event: QKeyEvent) -> bool:
        """``[`` / ``]`` lower / raise the colour tolerance."""
        if event.text() not in ("[", "]"):
            return False
        step = 4 if event.text() == "]" else -4
        self.tolerance = min(max(self.tolerance + step, 0), 255)
        canvas.status_changed.emit(f"wand tolerance {self.tolerance}")
        return True
