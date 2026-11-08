"""Open video files by extracting their frames to a folder of images.

Everything else in easy-mask works on folders of images, so a video becomes one: frames
are written as images (JPEG by default; PNG or WebP on request) to ``<video stem>_frames/``
next to the video, or any folder the user picks, and annotated like any other folder (one
JSON per frame). Frame files are named by their index in the video, so
``frame_000120.jpg`` is frame 120 whatever ``step`` was used.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2

from easy_mask.io.images import list_images

VIDEO_EXTENSIONS = frozenset({".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"})
JPEG_QUALITY = 95
IMAGE_FORMATS = {"png": "PNG (lossless)", "jpg": "JPEG", "webp": "WebP"}


def is_video(path: Path) -> bool:
    """Return True if ``path`` has a supported video extension."""
    return path.suffix.lower() in VIDEO_EXTENSIONS


def frames_dir_for(video: Path) -> Path:
    """The folder a video's frames are extracted to."""
    return video.with_name(f"{video.stem}_frames")


@dataclass(frozen=True)
class VideoInfo:
    """Basic facts about a video file, for planning an import."""

    frames: int
    fps: float
    width: int
    height: int

    @property
    def seconds(self) -> float:
        """Duration in seconds (0 if the frame rate is unknown)."""
        return self.frames / self.fps if self.fps > 0 else 0.0


def probe(video: Path) -> VideoInfo:
    """Read frame count, frame rate and size without decoding the frames.

    Raises:
        ValueError: If the video cannot be opened.
    """
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video}")
    try:
        return VideoInfo(
            frames=max(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 0),
            fps=float(capture.get(cv2.CAP_PROP_FPS) or 0.0),
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        capture.release()


def planned_count(total_frames: int, step: int, max_frames: int | None) -> int:
    """How many images an import will write (0 if the video's frame count is unknown)."""
    count = -(-total_frames // step) if total_frames > 0 else 0  # ceil division
    return min(count, max_frames) if max_frames else count


def _write_params(fmt: str, quality: int) -> list[int]:
    if fmt == "jpg":
        return [cv2.IMWRITE_JPEG_QUALITY, quality]
    if fmt == "webp":
        return [cv2.IMWRITE_WEBP_QUALITY, quality]
    return [cv2.IMWRITE_PNG_COMPRESSION, 3]  # png: lossless; 3 = good speed / size


def extract_frames(
    video: Path,
    out_dir: Path | None = None,
    step: int = 1,
    max_frames: int | None = None,
    fmt: str = "jpg",
    quality: int = JPEG_QUALITY,
    progress: Callable[[int, int], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> Path:
    """Write every ``step``-th frame of ``video`` as an image, up to ``max_frames`` frames.

    If the folder already holds images (an earlier extraction, maybe with annotations),
    nothing is extracted again, so reopening a video is instant and keeps its work.

    Args:
        video: The video file.
        out_dir: Destination folder; defaults to :func:`frames_dir_for`.
        step: Keep one frame in ``step`` (1 = every frame).
        max_frames: Stop after this many frames (None = whole video).
        fmt: Image format, one of :data:`IMAGE_FORMATS` (``png`` is lossless).
        quality: 1-100 for ``jpg`` / ``webp``; ignored for ``png``.
        progress: Called with ``(images written, images planned)`` after each image;
            planned is 0 when the video does not report its length.
        should_stop: Polled after each image; returning True stops early (the images
            written so far are kept).

    Returns:
        The folder of frames.

    Raises:
        ValueError: If the video cannot be read, has no frames, or arguments are invalid.
    """
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")
    if fmt not in IMAGE_FORMATS:
        raise ValueError(f"format must be one of {sorted(IMAGE_FORMATS)}, got {fmt!r}")
    out_dir = out_dir if out_dir is not None else frames_dir_for(video)
    if out_dir.is_dir() and list_images(out_dir):
        return out_dir

    planned = planned_count(probe(video).frames, step, max_frames)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video}")
    out_dir.mkdir(parents=True, exist_ok=True)
    params = _write_params(fmt, quality)
    written, index = 0, 0
    try:
        while max_frames is None or written < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if index % step == 0:
                cv2.imwrite(str(out_dir / f"frame_{index:06d}.{fmt}"), frame, params)
                written += 1
                if progress is not None:
                    progress(written, planned)
                if should_stop is not None and should_stop():
                    break
            index += 1
    finally:
        capture.release()
    if written == 0:
        out_dir.rmdir()
        raise ValueError(f"No frames could be read from {video}")
    return out_dir
