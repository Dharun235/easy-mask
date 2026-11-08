"""How well do the classic (model-free) pixel tools segment real objects?

For every ground-truth mask, each tool gets the prompt a user would give it:

* **GrabCut** from the ground-truth bounding box.
* **Otsu** threshold (gray) inside that box. Otsu splits the box into two classes; the
  one covering less of the box border is taken as the object (no ground truth used).
* **Magic wand** (tolerance 32) from the interior point farthest from the object edge.

Reports mean IoU with the ground truth and milliseconds per image.

Usage::

    uv run python -m benchmarks.classic_tools [--limit N]
"""

import argparse
import time
from collections.abc import Callable

import cv2
import numpy as np
from PIL import Image

from benchmarks.datasets import Dataset, available
from easy_mask.core import masks, pixel_ops

WAND_TOLERANCE = 32
Tool = Callable[[np.ndarray, np.ndarray], np.ndarray]


def bounding_box(gt: np.ndarray) -> pixel_ops.Box:
    """Tight ``(x0, y0, x1, y1)`` box (exclusive ends) around the ground truth."""
    ys, xs = np.nonzero(gt)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def interior_point(gt: np.ndarray) -> tuple[int, int]:
    """The object pixel farthest from the background (the standard first click)."""
    distance = cv2.distanceTransform(np.pad(gt, 1).astype(np.uint8), cv2.DIST_L2, 5)[1:-1, 1:-1]
    y, x = np.unravel_index(int(np.argmax(distance)), distance.shape)
    return int(x), int(y)


def grabcut_tool(rgb: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return pixel_ops.grabcut(rgb, bounding_box(gt))


def otsu_tool(rgb: np.ndarray, gt: np.ndarray) -> np.ndarray:
    x0, y0, x1, y1 = bounding_box(gt)
    region = np.zeros(gt.shape, dtype=bool)
    region[y0:y1, x0:x1] = True
    bright = pixel_ops.threshold(rgb, "otsu", region=region)
    border = region.copy()
    border[y0 + 1 : y1 - 1, x0 + 1 : x1 - 1] = False
    # The object rarely fills the box edge, so the class owning more of it is background.
    return bright if bright[border].mean() < 0.5 else region & ~bright


def wand_tool(rgb: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return pixel_ops.magic_wand(rgb, interior_point(gt), WAND_TOLERANCE)


TOOLS: dict[str, Tool] = {
    "GrabCut (box)": grabcut_tool,
    "Otsu (box)": otsu_tool,
    "Magic wand (click)": wand_tool,
}


def evaluate(dataset: Dataset, limit: int | None = None) -> dict[str, tuple[float, float]]:
    """Mean IoU and mean milliseconds per image for each tool on one dataset."""
    ious: dict[str, list[float]] = {name: [] for name in TOOLS}
    times: dict[str, list[float]] = {name: [] for name in TOOLS}
    for image_path, gt in dataset.samples(limit=limit):
        rgb = np.array(Image.open(image_path).convert("RGB"))
        for name, tool in TOOLS.items():
            start = time.perf_counter()
            result = tool(rgb, gt)
            times[name].append(1000 * (time.perf_counter() - start))
            ious[name].append(masks.iou(result, gt))
    return {name: (float(np.mean(ious[name])), float(np.mean(times[name]))) for name in TOOLS}


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark and print a markdown table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, help="max images per dataset")
    args = parser.parse_args(argv)

    datasets = available()
    if not datasets:
        print("No datasets in data/benchmarks; see docs/benchmarks.md")
        return 1
    header = ["dataset", "n", *(f"{name} IoU / ms" for name in TOOLS)]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for dataset in datasets:
        results = evaluate(dataset, args.limit)
        n = min(args.limit or dataset.expected_count, len(dataset.image_paths()))
        cells = [dataset.name, str(n)]
        cells += [f"{iou:.3f} / {ms:.0f}" for iou, ms in results.values()]
        print("| " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
