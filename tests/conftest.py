"""Shared fixtures and test-suite configuration.

Tests are grouped by *type* in folders; each test is auto-marked with its folder name
(``unit``, ``integration``, ``gui``, ...). See ``docs/testing.md`` for the strategy.
"""

import os
from pathlib import Path

import pytest
from PIL import Image

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
