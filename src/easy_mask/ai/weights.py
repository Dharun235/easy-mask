"""Download model files once to a per-user cache, verified by sha256.

Weights never go into the repository or the install: the first time a model is used its
files are downloaded to :func:`cache_dir` (``~/.cache/easy-mask/models`` by default;
override with ``EASY_MASK_MODELS``). A download goes to ``<name>.part`` first and is only
renamed once its size and checksum match, so an interrupted download never leaves a
broken file behind.
"""

import hashlib
import http.client
import os
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

CHUNK = 1 << 20  # 1 MiB
TIMEOUT_S = 30

# Called as progress(file name, bytes downloaded so far, total bytes).
ProgressFn = Callable[[str, int, int], None]


class WeightsError(RuntimeError):
    """A model file could not be downloaded or failed verification."""


@dataclass(frozen=True)
class WeightFile:
    """One downloadable model file and how to check it."""

    url: str
    sha256: str
    size: int

    @property
    def name(self) -> str:
        """File name in the cache (the last part of the URL)."""
        return self.url.rsplit("/", 1)[-1]


def cache_dir() -> Path:
    """Folder that holds downloaded model files."""
    override = os.environ.get("EASY_MASK_MODELS")
    if override:
        return Path(override)
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "easy-mask" / "models"


def cached_path(file: WeightFile, folder: Path | None = None) -> Path | None:
    """Path of ``file`` if it is already downloaded (right size), else None."""
    path = (folder or cache_dir()) / file.name
    return path if path.is_file() and path.stat().st_size == file.size else None


def ensure(
    file: WeightFile, folder: Path | None = None, progress: ProgressFn | None = None
) -> Path:
    """Return the local path of ``file``, downloading and verifying it first if needed.

    Raises:
        WeightsError: Offline, server error, truncated download, or checksum mismatch.
            Nothing is left in the cache in those cases.
    """
    folder = folder or cache_dir()
    existing = cached_path(file, folder)
    if existing is not None:
        return existing
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / file.name
    part = path.with_name(path.name + ".part")
    try:
        digest = _download(file, part, progress)
    except (OSError, http.client.HTTPException) as exc:
        part.unlink(missing_ok=True)
        raise WeightsError(
            f"Could not download {file.name} from {file.url}: {exc}.\n"
            f"Check your internet connection, or download it yourself to {path}."
        ) from exc
    received = part.stat().st_size
    if received != file.size or digest != file.sha256:
        part.unlink()
        raise WeightsError(
            f"Downloaded {file.name} is corrupt ({received} of {file.size} bytes, "
            f"sha256 {digest[:12]}... expected {file.sha256[:12]}...). Please try again."
        )
    part.replace(path)
    return path


def _download(file: WeightFile, part: Path, progress: ProgressFn | None) -> str:
    """Stream ``file.url`` into ``part``; return the sha256 of what was received."""
    sha = hashlib.sha256()
    done = 0
    with urllib.request.urlopen(file.url, timeout=TIMEOUT_S) as response, part.open("wb") as out:
        while chunk := response.read(CHUNK):
            out.write(chunk)
            sha.update(chunk)
            done += len(chunk)
            if progress is not None:
                progress(file.name, done, file.size)
    return sha.hexdigest()
