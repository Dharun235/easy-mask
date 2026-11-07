"""Label helpers: parsing label lists and giving each label a stable colour."""

import colorsys
import zlib
from pathlib import Path

RGB = tuple[int, int, int]


GOLDEN_RATIO_CONJUGATE = 0.618033988749895


def palette_color(index: int) -> RGB:
    """Return the ``index``-th colour of an endless, well-separated palette.

    Stepping the hue by the golden ratio keeps every new colour far from all previous
    ones, so the first ~10 labels are clearly distinguishable.
    """
    hue = (index * GOLDEN_RATIO_CONJUGATE) % 1.0
    saturation = 0.85 if index % 2 == 0 else 0.6
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, 0.95)
    return round(r * 255), round(g * 255), round(b * 255)


def label_color(label: str, labels: list[str] | tuple[str, ...] = ()) -> RGB:
    """Return the display colour of ``label``.

    Labels in ``labels`` (the project's label list, in order) get palette colours by
    position, which keeps them distinct and matches the exported mask palette. Unknown
    labels fall back to a colour derived from a hash of the name, so they stay stable.
    """
    if label in labels:
        return palette_color(labels.index(label))
    return palette_color(zlib.crc32(label.encode("utf-8")) % 1000)


def parse_name_list(value: str | None) -> list[str]:
    """Parse a ``--labels`` / ``--flags`` style argument.

    Args:
        value: Either a path to a text file with one name per line, or a comma-separated
            list such as ``"cat,dog"``. ``None`` gives an empty list.

    Returns:
        Names in their given order, stripped, blanks and duplicates removed.
    """
    if not value:
        return []
    path = Path(value)
    raw = path.read_text(encoding="utf-8").splitlines() if path.is_file() else value.split(",")
    names: list[str] = []
    for name in (n.strip() for n in raw):
        if name and name not in names:
            names.append(name)
    return names
