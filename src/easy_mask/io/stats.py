"""Folder statistics: how far along is a dataset, and are its labels valid?

Used by ``easy-mask stats`` and the GUI's statistics dialog. An image counts as
*annotated* when its JSON file exists (the same rule as the file list's done-marks),
even if it has no shapes: "nothing to label here" is an answer too.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from easy_mask.core import annotation as ann_io
from easy_mask.core import geometry
from easy_mask.core.annotation import REVIEWED_FLAG
from easy_mask.core.shapes import Shape, ShapeType
from easy_mask.io.images import list_images


@dataclass
class LabelStats:
    """Totals for one label.

    Attributes:
        objects: Object instances; shapes sharing a ``group_id`` count once.
        shapes: Individual shapes.
        images: Images containing the label at least once.
        area: Sum of shape areas in pixels (mask pixels, polygon area for vector shapes,
            0 for lines and points).
    """

    objects: int = 0
    shapes: int = 0
    images: int = 0
    area: float = 0.0


@dataclass
class FolderStats:
    """Totals for one folder of images.

    Attributes:
        images: Image files in the folder.
        annotated: Images with an annotation file.
        reviewed: Images whose ``reviewed`` flag is set (see review mode).
        objects: Object instances over all labels.
        shapes: Shapes over all labels.
        labels: Per-label totals, sorted by label name.
        unknown_labels: Only when a label list was given: each label not in it, mapped to
            the images that use it.
    """

    images: int = 0
    annotated: int = 0
    reviewed: int = 0
    objects: int = 0
    shapes: int = 0
    labels: dict[str, LabelStats] = field(default_factory=dict)
    unknown_labels: dict[str, list[str]] = field(default_factory=dict)

    @property
    def unannotated(self) -> int:
        """Images without an annotation file."""
        return self.images - self.annotated

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict (``--json`` output)."""
        return {**asdict(self), "unannotated": self.unannotated}


def shape_area(shape: Shape) -> float:
    """Area of one shape in pixels: exact for masks, geometric for vector shapes."""
    if shape.shape_type == ShapeType.MASK:
        return float(shape.mask.sum()) if shape.mask is not None else 0.0
    if shape.shape_type.is_area:
        return geometry.polygon_area(shape.outline())
    return 0.0


def _add_image(stats: FolderStats, shapes: list[Shape], allowed: list[str] | None) -> set[str]:
    """Add one image's shapes to ``stats``; return the labels it uses."""
    groups: set[tuple[str, int]] = set()
    for shape in shapes:
        label = stats.labels.setdefault(shape.label, LabelStats())
        label.shapes += 1
        label.area += shape_area(shape)
        if shape.group_id is None or (shape.label, shape.group_id) not in groups:
            label.objects += 1
            stats.objects += 1
        if shape.group_id is not None:
            groups.add((shape.label, shape.group_id))
    stats.shapes += len(shapes)
    used = {s.label for s in shapes}
    for name in used:
        stats.labels[name].images += 1
    return used if allowed is None else used - set(allowed)


def compute(
    image_dir: Path, annotation_dir: Path | None = None, allowed_labels: list[str] | None = None
) -> FolderStats:
    """Count images, annotations, objects and per-label totals for a folder.

    Args:
        image_dir: Folder of images.
        annotation_dir: Folder of JSON files; defaults to ``image_dir``.
        allowed_labels: The project's label list. If given, labels outside it are
            reported in :attr:`FolderStats.unknown_labels`.
    """
    stats = FolderStats()
    for image_path in list_images(image_dir):
        stats.images += 1
        json_path = ann_io.annotation_path_for(image_path, annotation_dir)
        if not json_path.exists():
            continue
        annotation = ann_io.load(json_path)
        stats.annotated += 1
        if annotation.flags.get(REVIEWED_FLAG):
            stats.reviewed += 1
        unknown = _add_image(stats, annotation.shapes, allowed_labels)
        if allowed_labels is not None:
            for name in sorted(unknown):
                stats.unknown_labels.setdefault(name, []).append(image_path.name)
    stats.labels = dict(sorted(stats.labels.items()))
    return stats


def format_table(stats: FolderStats, validated: bool = False) -> str:
    """Plain-text report: totals, a per-label table, and the label validation result.

    Args:
        stats: Result of :func:`compute`.
        validated: True if a label list was checked, so the validation line is shown.
    """
    lines = [
        f"Images   {stats.images}  (annotated {stats.annotated}, "
        f"unannotated {stats.unannotated}, reviewed {stats.reviewed})",
        f"Objects  {stats.objects}  (shapes {stats.shapes})",
        "",
    ]
    width = max([len("label"), *(len(name) for name in stats.labels)])
    lines.append(
        f"{'label':<{width}}  {'objects':>7}  {'shapes':>6}  {'images':>6}  {'area px':>10}"
    )
    for name, s in stats.labels.items():
        lines.append(
            f"{name:<{width}}  {s.objects:>7}  {s.shapes:>6}  {s.images:>6}  {s.area:>10.0f}"
        )
    if validated:
        lines.append("")
        if not stats.unknown_labels:
            lines.append("All labels are in the label list.")
        else:
            lines.append("Labels not in the label list:")
            for name, images in stats.unknown_labels.items():
                lines.append(f"  {name}: {', '.join(images)}")
    return "\n".join(lines)
