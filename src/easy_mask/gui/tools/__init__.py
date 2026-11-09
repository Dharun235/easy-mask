"""Pixel tools that plug into the canvas (brush, eraser, ...).

Vector shapes (polygon, rectangle, ...) are drawn by the canvas itself. Tools that work
on pixels implement :class:`~easy_mask.gui.tools.base.CanvasTool` and are activated with
``Canvas.set_tool``; the canvas forwards mouse, key and paint events to them.
"""
