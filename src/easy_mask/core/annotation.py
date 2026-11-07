"""Per-image annotation file: load, save, and locate.

Each image gets one JSON file. The format is labelme-compatible (minus embedded image
data), so annotations move freely between the two tools.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from easy_mask import __version__
from easy_mask.core.shapes import Shape


@dataclass
class Annotation:
    """Everything annotated on one image.

    Attributes:
        image_path: Path to the image, relative to the JSON file's folder.
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        shapes: Annotated objects, in drawing order (later shapes paint on top).
        flags: Image-level boolean attributes, e.g. ``{"blurry": False}``.
    """

    image_path: str
    image_width: int
    image_height: int
    shapes: list[Shape] = field(default_factory=list)
    flags: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a labelme-compatible dict."""
        return {
            "version": __version__,
            "flags": dict(self.flags),
            "shapes": [s.to_dict() for s in self.shapes],
            "imagePath": self.image_path,
            "imageData": None,
            "imageHeight": self.image_height,
            "imageWidth": self.image_width,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Annotation":
        """Build from a labelme-compatible dict. Embedded ``imageData`` is ignored."""
        return cls(
            image_path=data["imagePath"],
            image_width=int(data["imageWidth"]),
            image_height=int(data["imageHeight"]),
            shapes=[Shape.from_dict(s) for s in data.get("shapes", [])],
            flags=dict(data.get("flags") or {}),
        )


def annotation_path_for(image_path: Path, output_dir: Path | None = None) -> Path:
    """Return where the JSON for ``image_path`` lives.

    Args:
        image_path: The image file.
        output_dir: Folder for annotation files; defaults to the image's own folder.
    """
    folder = output_dir if output_dir is not None else image_path.parent
    return folder / f"{image_path.stem}.json"


def relative_image_path(image_path: Path, json_path: Path) -> str:
    """Return ``image_path`` relative to the folder of ``json_path`` (POSIX separators)."""
    rel = os.path.relpath(image_path.resolve(), json_path.parent.resolve())
    return Path(rel).as_posix()


def save(annotation: Annotation, json_path: Path) -> None:
    """Write ``annotation`` to ``json_path``, creating parent folders as needed.

    Writes to a temporary file first and renames it, so a crash mid-write never
    leaves a half-written annotation behind.
    """
    json_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = json_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(annotation.to_dict(), indent=2), encoding="utf-8")
    tmp_path.replace(json_path)


def load(json_path: Path) -> Annotation:
    """Read an annotation JSON file."""
    return Annotation.from_dict(json.loads(json_path.read_text(encoding="utf-8")))
