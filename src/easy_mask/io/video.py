"""Open video files by extracting their frames to a folder of images.

Everything else in easy-mask works on folders of images, so a video becomes one: frames
are written as JPEGs to ``<video stem>_frames/`` next to the video and annotated like any
other folder (one JSON per frame). Frame files are named by their index in the video, so
``frame_000120.jpg`` is frame 120 whatever ``step`` was used.
"""

from pathlib import Path

import cv2

from easy_mask.io.images import list_images

VIDEO_EXTENSIONS = frozenset({".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"})
JPEG_QUALITY = 95


def is_video(path: Path) -> bool:
    """Return True if ``path`` has a supported video extension."""
    return path.suffix.lower() in VIDEO_EXTENSIONS


def frames_dir_for(video: Path) -> Path:
    """The folder a video's frames are extracted to."""
    return video.with_name(f"{video.stem}_frames")


def extract_frames(
    video: Path, out_dir: Path | None = None, step: int = 1, max_frames: int | None = None
) -> Path:
    """Write every ``step``-th frame of ``video`` as a JPEG, up to ``max_frames`` frames.

    If the folder already holds images (an earlier extraction, maybe with annotations),
    nothing is extracted again, so reopening a video is instant and keeps its work.

    Args:
        video: The video file.
        out_dir: Destination folder; defaults to :func:`frames_dir_for`.
        step: Keep one frame in ``step`` (1 = every frame).
        max_frames: Stop after this many frames (None = whole video).

    Returns:
        The folder of frames.

    Raises:
        ValueError: If the video cannot be read or has no frames.
    """
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")
    out_dir = out_dir if out_dir is not None else frames_dir_for(video)
    if out_dir.is_dir() and list_images(out_dir):
        return out_dir

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video}")
    out_dir.mkdir(parents=True, exist_ok=True)
    written, index = 0, 0
    try:
        while max_frames is None or written < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if index % step == 0:
                path = out_dir / f"frame_{index:06d}.jpg"
                cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                written += 1
            index += 1
    finally:
        capture.release()
    if written == 0:
        out_dir.rmdir()
        raise ValueError(f"No frames could be read from {video}")
    return out_dir
