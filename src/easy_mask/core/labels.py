"""Label helpers: parsing label lists and giving each label a stable colour."""

import colorsys
import zlib
from pathlib import Path

RGB = tuple[int, int, int]


def label_color(label: str) -> RGB:
    """Return a bright, stable RGB colour for ``label``.

    Hue, saturation and brightness come from different bits of a CRC32 hash of the label,
    so the same label always gets the same colour across images and sessions without
    storing a palette, and labels with similar hues still differ in shade.
    """
    h = zlib.crc32(label.encode("utf-8"))
    hue = (h & 0xFFFF) / 0x10000
    saturation = 0.55 + 0.4 * ((h >> 16) & 0xFF) / 255
    value = 0.75 + 0.25 * ((h >> 24) & 0xFF) / 255
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return round(r * 255), round(g * 255), round(b * 255)


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
