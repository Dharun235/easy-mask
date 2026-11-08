"""Loaders for the interactive-segmentation benchmark datasets (GrabCut, Berkeley, DAVIS).

Expected layout under ``data/benchmarks`` is what the RITM / SimpleClick downloads unzip
to; see docs/benchmarks.md.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from easy_mask.io.images import list_images

ROOT = Path(__file__).parents[1] / "data" / "benchmarks"


@dataclass(frozen=True)
class Dataset:
    """Where one dataset's images and ground-truth masks live, relative to ``ROOT``."""

    name: str
    images: str
    masks: str
    mask_suffix: str
    expected_count: int

    def image_paths(self, root: Path = ROOT) -> list[Path]:
        """All images, in natural order."""
        return list_images(root / self.images)

    def mask_path(self, image_path: Path, root: Path = ROOT) -> Path:
        """Ground-truth mask file for ``image_path``."""
        return root / self.masks / (image_path.stem + self.mask_suffix)

    def is_available(self, root: Path = ROOT) -> bool:
        """True if the dataset has been downloaded."""
        return (root / self.images).is_dir()

    def samples(
        self, root: Path = ROOT, limit: int | None = None
    ) -> Iterator[tuple[Path, np.ndarray]]:
        """Yield ``(image_path, ground_truth_mask)`` pairs."""
        for image_path in self.image_paths(root)[:limit]:
            yield image_path, load_ground_truth(self.mask_path(image_path, root))


DATASETS = [
    Dataset("GrabCut", "GrabCut/GrabCut/data_GT", "GrabCut/GrabCut/boundary_GT", ".bmp", 50),
    Dataset("Berkeley", "Berkeley/Berkeley/images", "Berkeley/Berkeley/masks", ".png", 100),
    Dataset("DAVIS", "DAVIS/DAVIS345/img", "DAVIS/DAVIS345/gt", ".png", 345),
]


def load_ground_truth(mask_path: Path) -> np.ndarray:
    """Object pixels of a ground-truth mask, whatever the dataset's encoding.

    GrabCut: 0 / 128 (uncertain boundary band, excluded) / 255. Berkeley: RGB 0 / 255.
    DAVIS: palette image of object ids (0 = background); any id counts, as in RITM.
    """
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask > 128 if mask.max() == 255 else mask > 0


def available(root: Path = ROOT) -> list[Dataset]:
    """Datasets that are present on disk."""
    return [d for d in DATASETS if d.is_available(root)]
