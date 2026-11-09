"""Sanity check the classic pixel tools on a few real benchmark images.

Loose thresholds: this catches a broken tool (e.g. inverted output, wrong colour order),
not small quality changes. Full numbers: ``uv run python -m benchmarks.classic_tools``.
"""

import numpy as np
import pytest
from benchmarks.classic_tools import bounding_box, evaluate, interior_point
from benchmarks.datasets import DATASETS, ROOT, Dataset

pytestmark = [pytest.mark.benchmark, pytest.mark.slow]

MIN_IOU = {"GrabCut (box)": 0.6, "Otsu (box)": 0.15, "Magic wand (click)": 0.05}


@pytest.fixture(params=DATASETS, ids=lambda d: d.name)
def dataset(request) -> Dataset:
    ds = request.param
    if not (ROOT / ds.images).is_dir():
        pytest.skip(f"{ds.name} not downloaded to {ROOT} (see docs/benchmarks.md)")
    return ds


def test_classic_tools_beat_loose_thresholds(dataset):
    results = evaluate(dataset, limit=5)
    for name, (iou, _ms) in results.items():
        assert iou > MIN_IOU[name], f"{name} on {dataset.name}: IoU {iou:.3f}"


def test_prompt_helpers():
    gt = np.zeros((10, 12), bool)
    gt[2:7, 3:11] = True
    assert bounding_box(gt) == (3, 2, 11, 7)
    x, y = interior_point(gt)
    assert gt[y, x] and 4 <= x <= 9 and y == 4
