"""Per-folder project file: remembers a folder's labels, flags and annotation folder.

Opening a folder again restores them, so ``easy-mask images/`` needs no flags the second
time. The file is ``.easy-mask.json`` inside the image folder; it is small, human-readable
and safe to delete.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROJECT_FILE = ".easy-mask.json"


@dataclass
class Project:
    """What easy-mask remembers about one image folder.

    Attributes:
        labels: Known labels, in the order they get number keys and colours.
        flags: Image-level flag names.
        active_label: Label given to new shapes without asking, if any.
        output_dir: Annotation folder, relative to the image folder when possible.
        export_format: Format of the last export (a key of ``io.export.EXPORTERS``).
        export_dir: Folder of the last export, relative to the image folder when possible.
        export_on_close: Repeat the last export when easy-mask closes.
        object_parts: What the Objects export writes (masks / cutouts / coordinates).
    """

    labels: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    active_label: str | None = None
    output_dir: str | None = None
    export_format: str | None = None
    export_dir: str | None = None
    export_on_close: bool = False
    object_parts: list[str] | None = None


def load(folder: Path) -> Project | None:
    """Read the project file in ``folder``; None if there is none or it is unreadable."""
    path = folder / PROJECT_FILE
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    known = {name for name in Project.__dataclass_fields__}
    return Project(**{key: value for key, value in data.items() if key in known})


def save(project: Project, folder: Path) -> None:
    """Write the project file into ``folder`` (atomically)."""
    path = folder / PROJECT_FILE
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(project), indent=2), encoding="utf-8")
    tmp.replace(path)


def relative_dir(target: Path | None, folder: Path) -> str | None:
    """``target`` as a path relative to ``folder`` if it lies inside it, else absolute."""
    if target is None:
        return None
    try:
        return target.resolve().relative_to(folder.resolve()).as_posix()
    except ValueError:
        return str(target.resolve())


def resolve_dir(stored: str | None, folder: Path) -> Path | None:
    """Inverse of :func:`relative_dir`."""
    if stored is None:
        return None
    path = Path(stored)
    return path if path.is_absolute() else folder / path
