from pathlib import Path

import cv2
import numpy as np
import pytest

from easy_mask.io.images import list_images
from easy_mask.io.video import extract_frames, frames_dir_for, is_video


def write_video(path: Path, frames: int = 10, size=(64, 48)) -> Path:
    """A tiny MJPG .avi whose frame k has brightness 20 * k (readable on every platform)."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, size)
    for k in range(frames):
        writer.write(np.full((size[1], size[0], 3), 20 * k, np.uint8))
    writer.release()
    return path


@pytest.fixture
def video(tmp_path) -> Path:
    return write_video(tmp_path / "clip.avi")


@pytest.mark.parametrize(
    ("name", "expected"), [("a.mp4", True), ("b.MOV", True), ("c.avi", True), ("d.png", False)]
)
def test_is_video(name, expected):
    assert is_video(Path(name)) is expected


def test_extract_every_frame_next_to_the_video(video):
    folder = extract_frames(video)
    assert folder == frames_dir_for(video) == video.parent / "clip_frames"
    names = [p.name for p in list_images(folder)]
    assert names[:2] == ["frame_000000.jpg", "frame_000001.jpg"] and len(names) == 10


def test_frame_content_matches_source(video):
    folder = extract_frames(video)
    brightness = cv2.imread(str(folder / "frame_000005.jpg")).mean()
    assert brightness == pytest.approx(100, abs=4)


def test_step_and_max_frames_keep_original_frame_numbers(video, tmp_path):
    folder = extract_frames(video, tmp_path / "out", step=3, max_frames=3)
    assert [p.name for p in list_images(folder)] == [
        "frame_000000.jpg",
        "frame_000003.jpg",
        "frame_000006.jpg",
    ]


def test_existing_frames_are_reused_not_reextracted(video):
    folder = extract_frames(video, step=5)
    (folder / "frame_000000.json").write_text("{}")
    assert extract_frames(video) == folder
    assert len(list_images(folder)) == 2


def test_bad_step_rejected(video):
    with pytest.raises(ValueError, match="step"):
        extract_frames(video, step=0)


def test_unreadable_video_raises_and_leaves_no_folder(tmp_path):
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")
    with pytest.raises(ValueError):
        extract_frames(broken)
    assert not frames_dir_for(broken).exists()


@pytest.mark.parametrize("fmt", ["png", "jpg", "webp"])
def test_every_image_format(video, tmp_path, fmt):
    out = extract_frames(video, tmp_path / fmt, step=5, fmt=fmt)
    assert [p.name for p in list_images(out)] == [f"frame_000000.{fmt}", f"frame_000005.{fmt}"]


def test_png_is_lossless(video, tmp_path):
    out = extract_frames(video, tmp_path / "png", max_frames=1, fmt="png")
    source = cv2.VideoCapture(str(video))
    _, frame = source.read()
    source.release()
    np.testing.assert_array_equal(cv2.imread(str(out / "frame_000000.png")), frame)


def test_progress_and_stop(video, tmp_path):
    seen = []
    out = extract_frames(
        video,
        tmp_path / "out",
        progress=lambda done, total: seen.append((done, total)),
        should_stop=lambda: len(seen) >= 3,
    )
    assert seen == [(1, 10), (2, 10), (3, 10)]
    assert len(list_images(out)) == 3  # stopping keeps what was written


def test_bad_format_rejected(video):
    with pytest.raises(ValueError, match="format"):
        extract_frames(video, fmt="gif")


def test_probe_and_planned_count(video):
    from easy_mask.io.video import planned_count, probe

    info = probe(video)
    assert (info.frames, info.width, info.height) == (10, 64, 48)
    assert info.fps == pytest.approx(10) and info.seconds == pytest.approx(1.0)
    assert planned_count(10, 3, None) == 4 and planned_count(10, 1, 5) == 5
    assert planned_count(0, 1, None) == 0
    with pytest.raises(ValueError):
        probe(video.with_name("missing.mp4"))
