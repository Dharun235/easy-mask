import os
from pathlib import Path

import pytest
from PIL import Image

from easy_mask.core.shapes import Shape, ShapeType

# GUI tests must run without a display (CI, SSH).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


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
