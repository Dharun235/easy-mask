"""EfficientSAM wiring, with onnxruntime replaced by a fake (the real model is tested in
tests/integration/test_efficient_sam_model.py when its weights are cached)."""

import sys
import types

import numpy as np
import pytest

from easy_mask.ai import efficient_sam, weights
from easy_mask.ai.efficient_sam import EfficientSam, MissingDependencyError
from easy_mask.ai.segmenter import NEGATIVE, POSITIVE


class FakeSession:
    """Records feeds; the 'encoder' echoes the input, the 'decoder' returns 3 masks:
    everything (score 0.3), the pixel under the last point (0.8), the left two columns (0.5).
    """

    def __init__(self, path, providers):
        self.path, self.providers, self.feeds = path, providers, []

    def run(self, outputs, feed):
        self.feeds.append(feed)
        if "batched_images" in feed:
            return [feed["batched_images"]]
        height, width = feed["orig_im_size"]
        logits = np.full((1, 1, 3, height, width), -1.0, dtype=np.float32)
        x, y = feed["batched_point_coords"][0, 0, -1].astype(int)
        logits[0, 0, 0] = 1.0
        logits[0, 0, 1, y, x] = 1.0
        logits[0, 0, 2, :, :2] = 1.0
        return [logits, np.array([[[0.3, 0.8, 0.5]]], dtype=np.float32)]


@pytest.fixture
def fake_ort(monkeypatch):
    module = types.SimpleNamespace(InferenceSession=FakeSession)
    monkeypatch.setattr(efficient_sam, "import_onnxruntime", lambda: module)
    return module


def test_encoder_gets_float_chw_in_0_1_and_decoder_gets_prompts(fake_ort, tmp_path):
    model = EfficientSam(tmp_path / "enc.onnx", tmp_path / "dec.onnx")
    image = np.full((4, 5, 3), 255, dtype=np.uint8)
    model.set_image(image)
    batch = model._encoder.feeds[0]["batched_images"]
    assert batch.shape == (1, 3, 4, 5) and batch.dtype == np.float32 and batch.max() == 1.0
    assert model._encoder.providers == ["CPUExecutionProvider"]

    (best, score), *_ = model.predict([(1.0, 2.0), (3.0, 1.0)], [POSITIVE, POSITIVE])

    feed = model._decoder.feeds[0]
    assert feed["batched_point_coords"].shape == (1, 1, 2, 2)
    assert feed["batched_point_labels"].tolist() == [[[1.0, 1.0]]]
    assert feed["orig_im_size"].tolist() == [4, 5] and feed["orig_im_size"].dtype == np.int64
    assert score == pytest.approx(0.8)
    assert best.dtype == bool and best.sum() == 1 and best[1, 3]


def test_negative_points_cut_out_their_smallest_region(fake_ort, tmp_path):
    model = EfficientSam(tmp_path / "enc.onnx", tmp_path / "dec.onnx")
    model.set_image(np.zeros((4, 5, 3), dtype=np.uint8))

    best, _ = model.predict([(3.0, 3.0), (1.0, 2.0)], [POSITIVE, NEGATIVE])[0]

    assert best.sum() == 19 and not best[2, 1]  # whole image (one click) minus one pixel
    positives_feed = model._decoder.feeds[0]
    assert positives_feed["batched_point_labels"].tolist() == [[[1.0]]]  # negative not sent

    model.predict([(3.0, 3.0), (1.0, 2.0), (4.0, 0.0)], [POSITIVE, NEGATIVE, POSITIVE])
    assert len(model._decoder.feeds) == 3  # the cut for (1, 2) was reused
    model.set_image(np.zeros((4, 5, 3), dtype=np.uint8))
    model.predict([(3.0, 3.0), (1.0, 2.0)], [POSITIVE, NEGATIVE])
    assert len(model._decoder.feeds) == 5  # a new image forgets cached cuts


def test_only_negative_points_give_nothing(fake_ort, tmp_path):
    model = EfficientSam(tmp_path / "enc.onnx", tmp_path / "dec.onnx")
    model.set_image(np.zeros((4, 5, 3), dtype=np.uint8))
    assert model.predict([(1.0, 1.0)], [NEGATIVE]) == []


def test_load_downloads_both_files(fake_ort, monkeypatch, tmp_path):
    fetched = []

    def ensure(file, folder=None, progress=None):
        fetched.append(file.name)
        return tmp_path / file.name

    monkeypatch.setattr(weights, "ensure", ensure)
    model = efficient_sam.load("efficientsam-vits")
    assert fetched == ["efficient_sam_vits_encoder.onnx", "efficient_sam_vits_decoder.onnx"]
    assert model._decoder.path.endswith("efficient_sam_vits_decoder.onnx")


def test_missing_onnxruntime_explains_how_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "onnxruntime", None)  # makes `import onnxruntime` fail
    with pytest.raises(MissingDependencyError, match="uv sync --extra ai"):
        efficient_sam.load()


def test_model_table_is_consistent():
    assert efficient_sam.DEFAULT_MODEL in efficient_sam.MODELS
    for encoder, decoder in efficient_sam.MODELS.values():
        for file in (encoder, decoder):
            assert len(file.sha256) == 64 and file.size > 1_000_000
            assert file.url.startswith("https://") and file.name.endswith(".onnx")
