"""How many clicks does AI Segment need? The standard simulated-click protocol (RITM).

For every image: the first click goes to the point of the object farthest from its
boundary; every next click goes to the point farthest inside the largest error (positive
if foreground was missed, negative if background was included), up to 20 clicks. The
segmenter sees exactly what the app sends: the most recent clicks, best-scored mask.

Reports per dataset: NoC@85 / NoC@90 (mean clicks to reach that IoU, 20 if never),
NoF@90 (objects never reaching 90%), mIoU after 1 / 3 / 5 clicks, and CPU timing.

Usage::

    uv run python -m benchmarks.clicks --model efficientsam-vitt [--limit N]
"""

import argparse
import platform
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

from benchmarks.datasets import available, load_ignore_region
from easy_mask.ai import efficient_sam
from easy_mask.ai.segmenter import NEGATIVE, POSITIVE, Segmenter
from easy_mask.core.masks import iou

MAX_CLICKS = 20
THRESHOLDS = (0.85, 0.90)
MIOU_AT = (1, 3, 5)


def next_click(
    gt: np.ndarray, pred: np.ndarray, ignore: np.ndarray | None = None
) -> tuple[tuple[float, float], bool] | None:
    """Where a simulated user clicks next, and whether it is a positive click.

    Pixels in ``ignore`` never attract a click. Returns None if the prediction is perfect.
    """
    valid = ~ignore if ignore is not None else np.ones_like(gt)
    distances = []
    for error in (gt & ~pred & valid, pred & ~gt & valid):  # missed fg, false fg
        padded = np.pad(error, 1).astype(np.uint8)  # the image border counts as boundary
        distances.append(cv2.distanceTransform(padded, cv2.DIST_L2, 0)[1:-1, 1:-1])
    missed, extra = distances
    positive = bool(missed.max() >= extra.max())
    dist = missed if positive else extra
    if dist.max() == 0:
        return None
    y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return (float(x), float(y)), positive


@dataclass
class Result:
    """One object's IoU after each click and how long the model took."""

    ious: list[float] = field(default_factory=list)
    encode_s: float = 0.0
    click_s: list[float] = field(default_factory=list)

    def clicks_to(self, threshold: float) -> int:
        """Number of clicks until IoU >= threshold (MAX_CLICKS if never)."""
        for k, value in enumerate(self.ious, start=1):
            if value >= threshold:
                return k
        return MAX_CLICKS

    def iou_at(self, k: int) -> float:
        """IoU after ``k`` clicks (the last one if it stopped earlier)."""
        return self.ious[min(k, len(self.ious)) - 1]


def masked_iou(pred: np.ndarray, gt: np.ndarray, ignore: np.ndarray | None) -> float:
    """IoU over the pixels not in ``ignore``."""
    if ignore is None:
        return iou(pred, gt)
    return iou(pred[~ignore], gt[~ignore])


def simulate(
    segmenter: Segmenter, rgb: np.ndarray, gt: np.ndarray, ignore: np.ndarray | None = None
) -> Result:
    """Click on one object until IoU >= 90% (but at least 5 clicks) or 20 clicks."""
    result = Result()
    start = time.perf_counter()
    segmenter.set_image(rgb)
    result.encode_s = time.perf_counter() - start
    points: list[tuple[float, float]] = []
    labels: list[int] = []
    pred = np.zeros_like(gt)
    while len(points) < MAX_CLICKS:
        click = next_click(gt, pred, ignore)
        if click is None:  # perfect: further clicks change nothing (iou_at uses the last)
            break
        points.append(click[0])
        labels.append(POSITIVE if click[1] else NEGATIVE)
        start = time.perf_counter()
        predictions = segmenter.predict(points, labels)
        result.click_s.append(time.perf_counter() - start)
        pred = predictions[0][0] if predictions else np.zeros_like(gt)
        result.ious.append(masked_iou(pred, gt, ignore))
        if result.ious[-1] >= max(THRESHOLDS) and len(result.ious) >= max(MIOU_AT):
            break
    return result


def summarize(results: list[Result]) -> dict[str, float]:
    """Dataset-level metrics from per-object results."""
    row: dict[str, float] = {"n": len(results)}
    for t in THRESHOLDS:
        row[f"NoC@{round(t * 100)}"] = float(np.mean([r.clicks_to(t) for r in results]))
    row["NoF@90"] = sum(max(r.ious) < 0.90 for r in results)
    for k in MIOU_AT:
        row[f"mIoU@{k}"] = float(np.mean([r.iou_at(k) for r in results]))
    row["encode ms"] = 1000 * float(np.mean([r.encode_s for r in results]))
    row["click ms"] = 1000 * float(np.mean([s for r in results for s in r.click_s]))
    return row


COLUMNS = ["n", "NoC@85", "NoC@90", "NoF@90", "mIoU@1", "mIoU@3", "mIoU@5", "encode ms", "click ms"]


def format_row(name: str, row: dict[str, float]) -> str:
    """One markdown table row."""
    cells = [name]
    for col in COLUMNS:
        value = row[col]
        cells.append(
            f"{value:.0f}" if col in ("n", "NoF@90") or col.endswith("ms") else f"{value:.2f}"
        )
    return "| " + " | ".join(cells) + " |"


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark and print a markdown table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model", default=efficient_sam.DEFAULT_MODEL, choices=efficient_sam.MODELS
    )
    parser.add_argument("--limit", type=int, help="max images per dataset")
    args = parser.parse_args(argv)

    datasets = available()
    if not datasets:
        print("No datasets in data/benchmarks; see docs/benchmarks.md")
        return 1
    segmenter = efficient_sam.load(args.model)
    segmenter.set_image(np.zeros((480, 640, 3), np.uint8))  # warm-up, not timed
    print(f"{args.model} on {platform.processor() or platform.machine()}, {platform.platform()}")
    print("| dataset | " + " | ".join(COLUMNS) + " |")
    print("|---" * (len(COLUMNS) + 1) + "|")
    for dataset in datasets:
        results = []
        for image_path, gt in dataset.samples(limit=args.limit):
            rgb = np.array(Image.open(image_path).convert("RGB"))
            ignore = load_ignore_region(dataset.mask_path(image_path))
            results.append(simulate(segmenter, rgb, gt, ignore))
        print(format_row(dataset.name, summarize(results)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
