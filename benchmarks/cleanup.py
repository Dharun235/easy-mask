"""Does auto clean-up help AI Segment masks?

For each ground-truth object: one click at the interior point farthest from the boundary
(the standard first click), take AI Segment's mask, and compare IoU with the ground truth
before and after :func:`easy_mask.core.masks.clean_up` (the GUI's auto clean-up).

Usage::

    uv run python -m benchmarks.cleanup [--limit N] [--model efficientsam-vitt]
"""

import argparse

import cv2
import numpy as np
from PIL import Image

from benchmarks.datasets import available
from easy_mask.ai import efficient_sam
from easy_mask.core import masks


def first_click(gt: np.ndarray) -> tuple[float, float]:
    """The interior point farthest from the object boundary."""
    distance = cv2.distanceTransform(np.pad(gt, 1).astype(np.uint8), cv2.DIST_L2, 5)[1:-1, 1:-1]
    y, x = np.unravel_index(int(np.argmax(distance)), distance.shape)
    return float(x), float(y)


def main(argv: list[str] | None = None) -> int:
    """Print mean IoU before / after clean-up and how often it changed the mask."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model", default=efficient_sam.DEFAULT_MODEL)
    args = parser.parse_args(argv)
    segmenter = efficient_sam.load(args.model, None)
    print("| dataset | n | IoU raw | IoU cleaned | masks changed | better / worse |")
    print("|---|---|---|---|---|---|")
    for dataset in available():
        raw_ious, clean_ious, changed, better, worse = [], [], 0, 0, 0
        for image_path, gt in dataset.samples(limit=args.limit):
            with Image.open(image_path) as image:
                segmenter.set_image(np.asarray(image.convert("RGB")))
            (mask, _), *_ = segmenter.predict([first_click(gt)], [1])
            cleaned = masks.clean_up(mask, masks.cleanup_area(*mask.shape))
            raw, clean = masks.iou(mask, gt), masks.iou(cleaned, gt)
            raw_ious.append(raw)
            clean_ious.append(clean)
            changed += not np.array_equal(mask, cleaned)
            better += clean > raw + 1e-9
            worse += clean < raw - 1e-9
        n = len(raw_ious)
        print(
            f"| {dataset.name} | {n} | {np.mean(raw_ious):.4f} | {np.mean(clean_ious):.4f} "
            f"| {changed}/{n} | {better} / {worse} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
