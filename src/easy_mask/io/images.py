"""Finding and measuring image files."""

import re
from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})


def is_image(path: Path) -> bool:
    """Return True if ``path`` has a supported image extension."""
    return path.suffix.lower() in IMAGE_EXTENSIONS


def _natural_key(path: Path) -> list[int | str]:
    """Sort key so ``img2.png`` comes before ``img10.png``."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def list_images(folder: Path) -> list[Path]:
    """Return the images directly inside ``folder``, in natural sort order."""
    return sorted((p for p in folder.iterdir() if p.is_file() and is_image(p)), key=_natural_key)


def image_size(path: Path) -> tuple[int, int]:
    """Return ``(width, height)`` of an image without decoding all its pixels."""
    with Image.open(path) as image:
        return image.size
