"""Shared fixtures and test-suite configuration.

Tests are grouped by *type* in folders; each test is auto-marked with its folder name
(``unit``, ``integration``, ``gui``, ...). See ``docs/testing.md`` for the strategy.
"""

import os
import threading
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from easy_mask.ai.segmenter import Segmenter
from easy_mask.core.shapes import Shape, ShapeType

# GUI tests must run without a display (CI, SSH). Set before Qt is imported anywhere.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

TESTS_DIR = Path(__file__).parent
DATA_DIR = TESTS_DIR / "data"
TYPE_FOLDERS = {"unit", "integration", "gui", "property", "architecture", "smoke"}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark every test with the type folder it lives in."""
    for item in items:
        relative = Path(str(item.path)).relative_to(TESTS_DIR)
        folder = relative.parts[0]
        if folder in TYPE_FOLDERS:
            item.add_marker(getattr(pytest.mark, folder))


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path_factory, monkeypatch):
    """Never read or write the developer's real easy-mask settings (subprocesses inherit)."""
    monkeypatch.setenv("EASY_MASK_CONFIG_DIR", str(tmp_path_factory.mktemp("config")))


@pytest.fixture
def data_dir() -> Path:
    """Folder of small checked-in test files (golden outputs, sample annotations)."""
    return DATA_DIR


@pytest.fixture
def image_dir(tmp_path: Path) -> Path:
    """A folder with three small RGB images: img1.png, img2.png, img10.png."""
    folder = tmp_path / "images"
    folder.mkdir()
    for name in ("img10.png", "img2.png", "img1.png"):
        Image.new("RGB", (100, 80), (120, 120, 120)).save(folder / name)
    (folder / "notes.txt").write_text("not an image")
    return folder


@pytest.fixture
def square() -> Shape:
    """A 20x20 polygon square from (10,10) to (30,30) labelled 'cat'."""
    return Shape("cat", ShapeType.POLYGON, [(10, 10), (30, 10), (30, 30), (10, 30)])


class FakeSegmenter(Segmenter):
    """Deterministic stand-in for a real model: discs around clicks, clipped to the box.

    Positive points add a disc of radius ``RADIUS``, negative points cut one out; a box
    fills the box (positive points inside it add nothing). ``calls`` records what reached
    the decoder.
    """

    RADIUS = 10

    def __init__(self, cache_size: int = 8):
        super().__init__(cache_size)
        self.encoded: list[tuple[int, ...]] = []
        self.calls: list[tuple[np.ndarray, np.ndarray]] = []
        self.gate: threading.Event | None = None  # set to make encoding wait

    def _encode(self, rgb):
        if self.gate is not None:
            assert self.gate.wait(10), "test never released the encoder"
        self.encoded.append(rgb.shape)
        return rgb.shape

    def _decode(self, embedding, size, coords, labels):
        self.calls.append((coords, labels))
        height, width = size
        ys, xs = np.mgrid[:height, :width]
        mask = np.zeros(size, dtype=bool)
        corners = coords[labels >= 2]
        if len(corners):
            (x0, y0), (x1, y1) = corners
            mask = (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)
        for (x, y), label in zip(coords, labels, strict=True):
            disc = (xs - x) ** 2 + (ys - y) ** 2 <= self.RADIUS**2
            if label == 1 and not len(corners):
                mask |= disc
            elif label == 0:
                mask &= ~disc
        part = mask & (ys < ys[mask].min() + 2) if mask.any() else mask
        # Unsorted on purpose: the base class ranks them (best by score and by area).
        return [(np.zeros(size, dtype=bool), 0.2), (mask, 0.9), (part, 0.1)]


@pytest.fixture
def fake_segmenter() -> FakeSegmenter:
    """A fresh :class:`FakeSegmenter`."""
    return FakeSegmenter()


class FakePropagator:
    """Deterministic stand-in for the video model: each frame shifts the prompt right.

    Frame ``k`` gets the most recent prompt mask of each object moved ``shift * (k - p)``
    pixels to the right, where ``p`` is that prompt's frame. Records every call.
    """

    def __init__(self, shift: int = 2):
        self.shift = shift
        self.calls: list[tuple[list[Path], dict]] = []

    def propagate(self, frames, prompts):
        self.calls.append((list(frames), prompts))
        for position in range(1, len(frames)):
            result = {}
            for object_id in prompts[0]:
                origin = max(p for p in prompts if p <= position and object_id in prompts[p])
                shift = self.shift * (position - origin)
                result[object_id] = np.roll(prompts[origin][object_id], shift, axis=1)
            yield {position: result}


@pytest.fixture
def fake_propagator() -> FakePropagator:
    """A fresh :class:`FakePropagator`."""
    return FakePropagator()
