"""How faithfully does easy-mask store and convert real object masks?

For every ground-truth mask in the benchmark datasets:

1. **Storage:** mask -> mask shape -> JSON -> mask shape -> full mask must be identical.
2. **Polygon conversion:** mask -> polygons at several tolerances -> rasterized back.
   Reports IoU with the original and the number of vertices (the size/accuracy trade-off
   behind ``Mask > Convert to Polygons``).

Usage::

    uv run python -m benchmarks.mask_fidelity [--limit N]
"""

import argparse
import json
import time

import numpy as np

from benchmarks.datasets import available
from easy_mask.core import masks, operations
from easy_mask.core.rasterize import shape_to_mask
from easy_mask.core.shapes import Shape

TOLERANCES = (0.5, 1.0, 2.0, 4.0)


def evaluate(gt: np.ndarray) -> dict[str, float]:
    """Storage and conversion metrics for one ground-truth mask."""
    height, width = gt.shape
    shape = masks.to_shape("object", gt)
    assert shape is not None
    stored = json.dumps(shape.to_dict())
    restored = Shape.from_dict(json.loads(stored))
    result = {
        "lossless": float(np.array_equal(masks.to_full(restored, height, width), gt)),
        "json_kb": len(stored) / 1024,
    }
    for tol in TOLERANCES:
        polygons = operations.to_polygons([shape], [0], height, width, tolerance=tol)
        back = np.zeros_like(gt)
        for polygon in polygons:
            back |= shape_to_mask(polygon, height, width)
        result[f"iou@{tol}"] = masks.iou(back, gt)
        result[f"vertices@{tol}"] = float(sum(len(p.points) for p in polygons))
    return result


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark and print a markdown table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, help="max images per dataset")
    args = parser.parse_args(argv)

    datasets = available()
    if not datasets:
        print("No datasets in data/benchmarks; see docs/benchmarks.md")
        return 1
    header = ["dataset", "n", "lossless", "JSON KB"] + [
        f"IoU / vertices @ tol {t}" for t in TOLERANCES
    ]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for dataset in datasets:
        start = time.perf_counter()
        rows = [evaluate(gt) for _, gt in dataset.samples(limit=args.limit)]
        mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        cells = [dataset.name, str(len(rows)), f"{mean['lossless']:.0%}", f"{mean['json_kb']:.1f}"]
        cells += [f"{mean[f'iou@{t}']:.4f} / {mean[f'vertices@{t}']:.0f}" for t in TOLERANCES]
        print("| " + " | ".join(cells) + " |")
        print(f"<!-- {dataset.name}: {time.perf_counter() - start:.1f}s -->")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
