"""The interface every click-to-segment backend implements.

A segmenter works in two steps, like SAM:

1. :meth:`Segmenter.set_image` runs the heavy image encoder once (hundreds of ms on CPU)
   and keeps the embedding, cached per image key so revisiting an image is instant.
2. :meth:`Segmenter.predict` turns clicks and an optional box into masks using that
   embedding (tens of ms), so it can run on every click.

Prompt conventions (shared with SAM / EfficientSAM): points are ``(x, y)`` in original
image pixels; label 1 is foreground, 0 is background; a box ``(x0, y0, x1, y1)`` is sent
as two extra points labelled 2 (top-left) and 3 (bottom-right).
"""

import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

import numpy as np

from easy_mask.core.geometry import Point

Box = tuple[float, float, float, float]  # x0, y0, x1, y1 in image pixels
Prediction = tuple[np.ndarray, float]  # (H x W bool mask, predicted IoU score)

POSITIVE, NEGATIVE = 1, 0
BOX_TOP_LEFT, BOX_BOTTOM_RIGHT = 2, 3


def encode_prompts(
    points: Sequence[Point], labels: Sequence[int], box: Box | None, max_points: int
) -> tuple[np.ndarray, np.ndarray]:
    """Turn clicks and a box into the ``(N, 2)`` coordinates and ``(N,)`` labels a decoder takes.

    Decoders accept at most ``max_points`` points in total. The box (two corner points)
    is always kept because it frames the object; the remaining slots go to the *most
    recent* clicks, since later clicks correct earlier ones.

    Raises:
        ValueError: If ``points`` and ``labels`` differ in length.
    """
    if len(points) != len(labels):
        raise ValueError(f"{len(points)} points but {len(labels)} labels")
    coords: list[Point] = []
    codes: list[int] = []
    if box is not None:
        x0, y0, x1, y1 = box
        coords += [(min(x0, x1), min(y0, y1)), (max(x0, x1), max(y0, y1))]
        codes += [BOX_TOP_LEFT, BOX_BOTTOM_RIGHT]
    room = max_points - len(coords)
    if room > 0:
        coords += list(points)[-room:]
        codes += list(labels)[-room:]
    return np.array(coords, dtype=np.float32).reshape(-1, 2), np.array(codes, dtype=np.float32)


class Segmenter(ABC):
    """Base class: embedding cache and prompt handling; subclasses run the model.

    Thread safety: :meth:`set_image` may run on a worker thread (the GUI does this so
    encoding never freezes the window) while the cache is read elsewhere; cache access is
    locked. Call :meth:`predict` from one thread at a time.

    Ranking: candidates are ordered by predicted IoU, except for a lone click, where the
    largest candidate (the whole object rather than a part) comes first. A single point is
    ambiguous and EfficientSAM's IoU head then favours small parts: on 25 images each of
    GrabCut / Berkeley / DAVIS the largest mask scores mIoU@1 0.89 / 0.76 / 0.60 against
    0.57 / 0.54 / 0.45 for the top-scored one. With more prompts the scores are reliable.

    Attributes:
        max_points: Most prompt points (box corners included) the decoder accepts.
    """

    max_points = 6

    def __init__(self, cache_size: int = 8):
        self._cache: OrderedDict[str, tuple[Any, tuple[int, int]]] = OrderedDict()
        self._cache_size = cache_size
        self._lock = threading.Lock()
        self._current: tuple[Any, tuple[int, int]] | None = None

    # ------------------------------------------------------------------ public API

    def set_image(self, rgb: np.ndarray, key: str | None = None) -> None:
        """Make ``rgb`` the image that :meth:`predict` segments.

        Args:
            rgb: ``(H, W, 3)`` uint8 RGB image.
            key: Identifies the image (e.g. its path). An embedding cached under the same
                key is reused instead of re-encoding; None disables caching.
        """
        if key is not None:
            with self._lock:
                cached = self._cache.get(key)
                if cached is not None:
                    self._cache.move_to_end(key)
                    self._current = cached
                    return
        entry = (self._encode(rgb), (rgb.shape[0], rgb.shape[1]))
        with self._lock:
            if key is not None:
                self._cache[key] = entry
                while len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)  # least recently used
            self._current = entry

    def precompute(self, rgb: np.ndarray, key: str) -> None:
        """Encode ``rgb`` into the cache under ``key`` without making it the current image.

        Used to prepare the next images in the background so switching to them is instant.
        """
        if self.is_cached(key):
            return
        entry = (self._encode(rgb), (rgb.shape[0], rgb.shape[1]))
        with self._lock:
            self._cache[key] = entry
            self._cache.move_to_end(key, last=False)  # evict prefetched images first
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)

    def is_cached(self, key: str) -> bool:
        """True if an embedding for ``key`` is cached (``set_image`` will be instant)."""
        with self._lock:
            return key in self._cache

    def predict(
        self, points: Sequence[Point], labels: Sequence[int], box: Box | None = None
    ) -> list[Prediction]:
        """Masks for the given prompts, best first (see the class docstring for the ranking).

        Args:
            points: Clicks ``(x, y)`` in image pixels. Only the most recent ones are used
                if there are more than the model accepts (see :func:`encode_prompts`).
            labels: :data:`POSITIVE` (object) or :data:`NEGATIVE` (background) per point.
            box: Optional ``(x0, y0, x1, y1)`` box around the object.

        Returns:
            ``(mask, score)`` pairs; empty if there is no prompt.

        Raises:
            RuntimeError: If no image was set.
        """
        if self._current is None:
            raise RuntimeError("call set_image() before predict()")
        if not points and box is None:
            return []
        coords, codes = encode_prompts(points, labels, box, self.max_points)
        embedding, size = self._current
        predictions = self._decode(embedding, size, coords, codes)
        if len(codes) == 1:  # a single click: prefer the whole object (see class docstring)
            return sorted(predictions, key=lambda p: int(p[0].sum()), reverse=True)
        return sorted(predictions, key=lambda p: p[1], reverse=True)

    # ------------------------------------------------------------------ backend hooks

    @abstractmethod
    def _encode(self, rgb: np.ndarray) -> Any:
        """Run the image encoder; return whatever :meth:`_decode` needs."""

    @abstractmethod
    def _decode(
        self, embedding: Any, size: tuple[int, int], coords: np.ndarray, labels: np.ndarray
    ) -> list[Prediction]:
        """Run the prompt decoder.

        Args:
            embedding: Result of :meth:`_encode` for the current image.
            size: Image ``(height, width)``; masks must have this shape.
            coords: ``(N, 2)`` float32 point coordinates (box corners first).
            labels: ``(N,)`` float32 point labels (0, 1, 2, 3).
        """
