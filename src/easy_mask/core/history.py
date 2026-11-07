"""Undo/redo by snapshots.

Every edit pushes a full copy of the shape list. Annotations are small (a few hundred
shapes at most), so snapshots are cheap and far simpler than per-operation commands.
"""

from easy_mask.core.shapes import Shape


def _snapshot(shapes: list[Shape]) -> list[Shape]:
    return [s.copy() for s in shapes]


class History:
    """A linear undo/redo stack of shape-list snapshots.

    Usage: call :meth:`reset` when an image is opened, :meth:`push` after every edit,
    and apply whatever :meth:`undo` / :meth:`redo` return.
    """

    def __init__(self, limit: int = 200):
        self._limit = limit
        self._states: list[list[Shape]] = [[]]
        self._index = 0

    def reset(self, shapes: list[Shape]) -> None:
        """Forget all history and start from ``shapes``."""
        self._states = [_snapshot(shapes)]
        self._index = 0

    def push(self, shapes: list[Shape]) -> None:
        """Record ``shapes`` as the newest state, discarding any redo states."""
        self._states = self._states[: self._index + 1]
        self._states.append(_snapshot(shapes))
        if len(self._states) > self._limit:
            self._states.pop(0)
        self._index = len(self._states) - 1

    @property
    def can_undo(self) -> bool:
        """True if there is an earlier state."""
        return self._index > 0

    @property
    def can_redo(self) -> bool:
        """True if an undone state can be restored."""
        return self._index < len(self._states) - 1

    def undo(self) -> list[Shape] | None:
        """Step back and return a copy of that state, or None if at the start."""
        if not self.can_undo:
            return None
        self._index -= 1
        return _snapshot(self._states[self._index])

    def redo(self) -> list[Shape] | None:
        """Step forward and return a copy of that state, or None if at the end."""
        if not self.can_redo:
            return None
        self._index += 1
        return _snapshot(self._states[self._index])
