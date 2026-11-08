"""How well do masks carry across video frames? DAVIS 2017 J&F, with and without a model.

Two protocols on DAVIS 2017 val (480p):

1. **Propagation:** give the model the ground-truth masks of the first frame, propagate
   through the sequence, score every later frame (the semi-supervised VOS protocol).
2. **Keyframe interpolation (no model):** for each 10-frame window, take the ground truth
   of its first and last frame and interpolate the 8 frames between
   (:func:`easy_mask.core.interpolate.interpolate_mask`); score those 8.

Scores follow the DAVIS definitions: J is region IoU, F the boundary F-measure with a
tolerance of 0.8% of the image diagonal, averaged per object over frames, then over
objects. Boundaries are traced with a 3x3 erosion rather than DAVIS's ``seg2bmap``, which
moves F by well under a point. Like the official evaluation, the first frame (the prompt)
and the last frame are excluded from propagation scores.

Usage::

    uv run python -m benchmarks.video [--sequences 5] [--max-frames N]
        [--model efficienttam_ti_512] [--device cpu] [--skip-model]
"""

import argparse
import platform
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from easy_mask.core import masks
from easy_mask.core.interpolate import interpolate_mask

ROOT = Path(__file__).parents[1] / "data" / "benchmarks" / "DAVIS2017" / "DAVIS"
WINDOW = 10  # frames per interpolation window, both ends included
BOUNDARY_TOLERANCE = 0.008  # fraction of the image diagonal (DAVIS)


@dataclass
class Sequence:
    """One DAVIS sequence: frame files and per-frame object-id masks."""

    name: str
    frames: list[Path]
    labels: list[np.ndarray]

    @property
    def object_ids(self) -> list[int]:
        """Objects present in the first frame (the ones to track)."""
        return [int(i) for i in np.unique(self.labels[0]) if i != 0]


def available(root: Path = ROOT) -> bool:
    """True if DAVIS 2017 has been downloaded and unzipped to ``root``."""
    return (root / "ImageSets" / "2017" / "val.txt").exists()


def load_sequences(count: int, max_frames: int | None = None, root: Path = ROOT) -> list[Sequence]:
    """The first ``count`` DAVIS 2017 val sequences, optionally cut to ``max_frames``."""
    names = (root / "ImageSets" / "2017" / "val.txt").read_text().split()[:count]
    sequences = []
    for name in names:
        frames = sorted((root / "JPEGImages" / "480p" / name).glob("*.jpg"))[:max_frames]
        labels = [
            np.array(Image.open(root / "Annotations" / "480p" / name / f"{f.stem}.png"))
            for f in frames
        ]
        sequences.append(Sequence(name, frames, labels))
    return sequences


# ---------------------------------------------------------------------- metrics


def boundary(full: np.ndarray) -> np.ndarray:
    """One-pixel-wide inner boundary of a mask."""
    as_u8 = full.astype(np.uint8)
    return (as_u8 - cv2.erode(as_u8, np.ones((3, 3), np.uint8))).astype(bool)


def f_measure(pred: np.ndarray, gt: np.ndarray) -> float:
    """DAVIS boundary F-measure: boundary precision/recall within a small tolerance."""
    radius = max(1, round(BOUNDARY_TOLERANCE * np.hypot(*gt.shape)))
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    pred_b, gt_b = boundary(pred), boundary(gt)
    if not pred_b.any() and not gt_b.any():
        return 1.0
    if not pred_b.any() or not gt_b.any():
        return 0.0
    gt_near = cv2.dilate(gt_b.astype(np.uint8), disc).astype(bool)
    pred_near = cv2.dilate(pred_b.astype(np.uint8), disc).astype(bool)
    precision = (pred_b & gt_near).sum() / pred_b.sum()
    recall = (gt_b & pred_near).sum() / gt_b.sum()
    return 0.0 if precision + recall == 0 else float(2 * precision * recall / (precision + recall))


def score(pairs: dict[int, list[tuple[np.ndarray, np.ndarray]]]) -> tuple[float, float]:
    """Mean J and F: per object over its ``(pred, gt)`` frames, then over objects."""
    j = [np.mean([masks.iou(p, g) for p, g in frames]) for frames in pairs.values()]
    f = [np.mean([f_measure(p, g) for p, g in frames]) for frames in pairs.values()]
    return float(np.mean(j)), float(np.mean(f))


# ---------------------------------------------------------------------- protocols


def run_interpolation(seq: Sequence) -> tuple[float, float]:
    """J, F of keyframe interpolation over consecutive 10-frame windows."""
    pairs: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {i: [] for i in seq.object_ids}
    for start in range(0, len(seq.frames) - WINDOW + 1, WINDOW - 1):
        end = start + WINDOW - 1
        for object_id in seq.object_ids:
            first, last = seq.labels[start] == object_id, seq.labels[end] == object_id
            for k in range(start + 1, end):
                t = (k - start) / (end - start)
                pairs[object_id].append(
                    (interpolate_mask(first, last, t), seq.labels[k] == object_id)
                )
    return score(pairs)


def run_propagation(seq: Sequence, propagator: object) -> tuple[float, float, float]:
    """J, F and seconds per frame of propagating the first frame's ground truth."""
    prompts = {0: {i: seq.labels[0] == i for i in seq.object_ids}}
    pairs: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {i: [] for i in seq.object_ids}
    start = time.perf_counter()
    for result in propagator.propagate(seq.frames, prompts):  # type: ignore[attr-defined]
        for position, by_id in result.items():
            if position == len(seq.frames) - 1:
                continue  # DAVIS skips the last frame
            for object_id, pred in by_id.items():
                pairs[object_id].append((pred, seq.labels[position] == object_id))
    seconds = (time.perf_counter() - start) / (len(seq.frames) - 1)
    j, f = score(pairs)
    return j, f, seconds


def machine() -> str:
    """One line describing where the benchmark ran."""
    cpu = platform.processor() or platform.machine()
    return f"{platform.platform()}, {cpu}, Python {platform.python_version()}"


def table(rows: list[list[str]], header: list[str]) -> None:
    """Print a markdown table."""
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for row in rows:
        print("| " + " | ".join(row) + " |")


def evaluate(
    sequences: list[Sequence], protocol: Callable[[Sequence], tuple[float, ...]]
) -> list[tuple[float, ...]]:
    """Run ``protocol`` on every sequence, printing progress."""
    results = []
    for seq in sequences:
        results.append(protocol(seq))
        print(f"<!-- {seq.name}: {results[-1]} -->", flush=True)
    return results


def main(argv: list[str] | None = None) -> int:
    """Run both protocols and print markdown tables."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sequences", type=int, default=5, help="first N val sequences")
    parser.add_argument("--max-frames", type=int, help="cut each sequence to N frames")
    parser.add_argument("--model", default="efficienttam_ti_512")
    parser.add_argument("--device", help="cpu, mps or cuda (default: fastest available)")
    parser.add_argument("--skip-model", action="store_true", help="interpolation only")
    args = parser.parse_args(argv)
    if not available():
        print(f"DAVIS 2017 not found in {ROOT}; see docs/benchmarks.md")
        return 1
    sequences = load_sequences(args.sequences, args.max_frames)
    print(f"Machine: {machine()}\n")

    rows = []
    for seq, (j, f) in zip(sequences, evaluate(sequences, run_interpolation), strict=True):
        rows.append(
            [seq.name, str(len(seq.object_ids)), f"{j:.3f}", f"{f:.3f}", f"{(j + f) / 2:.3f}"]
        )
    print("\nKeyframe interpolation (10-frame windows, no model)\n")
    table(_with_mean(rows), ["sequence", "objects", "J", "F", "J&F"])

    if args.skip_model:
        return 0
    from easy_mask.propagation.backend import Sam2Propagator

    propagator = Sam2Propagator(args.model, args.device)
    start = time.perf_counter()
    propagator._load()
    print(f"\n<!-- model load: {time.perf_counter() - start:.1f}s on {propagator.device} -->")
    rows = []
    results = evaluate(sequences, lambda seq: run_propagation(seq, propagator))
    for seq, (j, f, sec) in zip(sequences, results, strict=True):
        rows.append(
            [
                seq.name,
                str(len(seq.frames)),
                f"{j:.3f}",
                f"{f:.3f}",
                f"{(j + f) / 2:.3f}",
                f"{sec:.2f}",
            ]
        )
    print(f"\nPropagation from frame 0: {args.model} on {propagator.device}\n")
    table(_with_mean(rows), ["sequence", "frames", "J", "F", "J&F", "s / frame"])
    return 0


def _with_mean(rows: list[list[str]]) -> list[list[str]]:
    """Append a row with the column means of the numeric columns (after the first two)."""
    means = [
        f"{np.mean([float(r[c]) for r in rows]):.{3 if c < 5 else 2}f}"
        for c in range(2, len(rows[0]))
    ]
    return [*rows, ["**mean**", "", *means]]


if __name__ == "__main__":
    raise SystemExit(main())
