"""Undo/redo across every frame of a folder, by snapshots.

One history covers the whole folder: an edit on one image, then a move to another and an
edit there, are two steps, and undoing walks back through both. A step records, for each
frame it touched, the frame's shapes before and after (``None``: the frame had no
annotation file). An edit is a one-frame step; a video run (propagate, erase, clear,
interpolate) is grouped into one step over all the frames it changed.

Annotations are small (a few hundred shapes at most), so snapshots are cheap and far
simpler than per-operation commands.
"""

from easy_mask.core.shapes import Shape

FrameState = list[Shape] | None  # a frame's shapes; None: no annotation file
Step = dict[int, tuple[FrameState, FrameState]]  # frame index -> (before, after)


def _copy(state: FrameState) -> FrameState:
    return None if state is None else [s.copy() for s in state]


class History:
    """A linear undo/redo stack of multi-frame steps.

    Usage: :meth:`record` every change with the frame's state before and after it, and
    apply the states :meth:`undo` / :meth:`redo` return. Wrap a run that changes many
    frames in :meth:`begin_group` / :meth:`end_group` so it undoes in one go.
    """

    def __init__(self, limit: int = 200):
        self._limit = limit
        self._steps: list[Step] = []
        self._index = 0  # steps[:index] are done, steps[index:] can be redone
        self._group: Step | None = None

    def clear(self) -> None:
        """Forget everything (a different folder was opened)."""
        self._steps, self._index, self._group = [], 0, None

    def record(self, frame: int, before: FrameState, after: FrameState) -> None:
        """Record that ``frame`` changed from ``before`` to ``after``."""
        if self._group is not None:
            first = self._group.get(frame, (before, after))[0]  # keep the earliest before
            self._group[frame] = (first, _copy(after))
            return
        self._push({frame: (_copy(before), _copy(after))})

    def begin_group(self) -> None:
        """Start collecting changes into one step (until :meth:`end_group`)."""
        self._group = {}

    def end_group(self) -> int:
        """Close the group; returns how many frames it changed (0: no step was added)."""
        group, self._group = self._group, None
        if group:
            self._push(group)
        return len(group or {})

    def _push(self, step: Step) -> None:
        self._steps = self._steps[: self._index]
        self._steps.append(step)
        if len(self._steps) > self._limit:
            self._steps.pop(0)
        self._index = len(self._steps)

    @property
    def can_undo(self) -> bool:
        """True if there is a step to undo."""
        return self._index > 0

    @property
    def can_redo(self) -> bool:
        """True if an undone step can be redone."""
        return self._index < len(self._steps)

    def undo(self) -> dict[int, FrameState] | None:
        """Step back: each touched frame's state before the step, or None at the start."""
        if not self.can_undo:
            return None
        self._index -= 1
        return {frame: _copy(before) for frame, (before, _) in self._steps[self._index].items()}

    def redo(self) -> dict[int, FrameState] | None:
        """Step forward: each touched frame's state after the step, or None at the end."""
        if not self.can_redo:
            return None
        step = self._steps[self._index]
        self._index += 1
        return {frame: _copy(after) for frame, (_, after) in step.items()}
