"""The real EfficientSAM model on real images.

Runs only when onnxruntime is importable and the ViT-T weights are
already cached (use AI Segment once, or run ``benchmarks.clicks``); never downloads.
Run with ``uv run pytest -m benchmark``.
"""

import importlib.util

import numpy as np
import pytest
from benchmarks import clicks
from benchmarks.datasets import DATASETS, load_ignore_region
from PIL import Image

from easy_mask.ai import efficient_sam, weights
from easy_mask.ai.segmenter import POSITIVE

pytestmark = [pytest.mark.benchmark, pytest.mark.slow]

GRABCUT = DATASETS[0]


@pytest.fixture(scope="module")
def model():
    if importlib.util.find_spec("onnxruntime") is None:
        pytest.skip("onnxruntime not importable (run uv sync)")
    encoder, decoder = efficient_sam.MODELS[efficient_sam.DEFAULT_MODEL]
    if weights.cached_path(encoder) is None or weights.cached_path(decoder) is None:
        pytest.skip(f"EfficientSAM weights not cached in {weights.cache_dir()}")
    return efficient_sam.load()


@pytest.fixture(scope="module")
def grabcut():
    if not GRABCUT.is_available():
        pytest.skip("GrabCut not downloaded (see docs/benchmarks.md)")
    return list(GRABCUT.samples(limit=3))


def test_one_click_in_the_middle_finds_the_object(model, grabcut):
    for image_path, gt in grabcut:
        rgb = np.array(Image.open(image_path).convert("RGB"))
        model.set_image(rgb, str(image_path))
        (point, _) = clicks.next_click(gt, np.zeros_like(gt))
        mask, score = model.predict([point], [POSITIVE])[0]
        assert mask.shape == gt.shape and 0 <= score <= 1
        assert clicks.masked_iou(mask, gt, load_ignore_region(GRABCUT.mask_path(image_path))) > 0.6


def test_click_benchmark_sanity(model, grabcut):
    results = []
    for image_path, gt in grabcut:
        rgb = np.array(Image.open(image_path).convert("RGB"))
        ignore = load_ignore_region(GRABCUT.mask_path(image_path))
        results.append(clicks.simulate(model, rgb, gt, ignore))
    row = clicks.summarize(results)
    assert row["mIoU@5"] > 0.85 and row["NoC@90"] < 10
