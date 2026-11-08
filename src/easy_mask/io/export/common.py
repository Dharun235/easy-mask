"""Shared input handling for all exporters."""

import shutil
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from easy_mask.core import annotation as ann_io
from easy_mask.core.annotation import Annotation
from easy_mask.io.images import list_images


@dataclass
class AnnotatedImage:
    """An image file paired with its loaded annotation."""

    image_path: Path
    annotation: Annotation


def collect(image_dir: Path, annotation_dir: Path | None = None) -> list[AnnotatedImage]:
    """Pair every image in ``image_dir`` with its annotation JSON, skipping unannotated ones.

    Args:
        image_dir: Folder of images.
        annotation_dir: Folder of JSON files; defaults to ``image_dir``.
    """
    pairs = []
    for image_path in list_images(image_dir):
        json_path = ann_io.annotation_path_for(image_path, annotation_dir)
        if json_path.exists():
            pairs.append(AnnotatedImage(image_path, ann_io.load(json_path)))
    return pairs


def resolve_labels(items: list[AnnotatedImage], labels: list[str] | None) -> list[str]:
    """Return the class list: ``labels`` if given, else every label seen, sorted."""
    if labels:
        return list(labels)
    return sorted({shape.label for item in items for shape in item.annotation.shapes})


ProgressFn = Callable[[int, int], None]  # (images done, images total)


class Exporter(Protocol):
    """What every exporter in ``EXPORTERS`` accepts."""

    def __call__(
        self,
        items: list[AnnotatedImage],
        out_dir: Path,
        labels: list[str] | None = None,
        *,
        progress: ProgressFn | None = None,
    ) -> None:
        """Write ``items`` as a dataset into ``out_dir``."""


def tracked(items: list[AnnotatedImage], progress: ProgressFn | None) -> Iterator[AnnotatedImage]:
    """Iterate ``items``, reporting ``progress(done, total)`` after each one."""
    for done, item in enumerate(items, start=1):
        yield item
        if progress is not None:
            progress(done, len(items))


def copy_images(items: list[AnnotatedImage], images_dir: Path) -> None:
    """Copy every image into ``images_dir``."""
    images_dir.mkdir(parents=True, exist_ok=True)
    for item in items:
        shutil.copy2(item.image_path, images_dir / item.image_path.name)
