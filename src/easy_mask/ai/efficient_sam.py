"""EfficientSAM (Xiong et al., 2024; Apache-2.0) running on CPU through onnxruntime.

The ONNX export by the labelme project takes the image at any size (it resizes to 1024
and normalizes inside the graph) and returns three candidate masks at full image size
with a predicted IoU each; :class:`~easy_mask.ai.segmenter.Segmenter` ranks them.

onnxruntime is an optional dependency: ``uv sync --extra ai``.
"""

import importlib
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from easy_mask.ai import weights
from easy_mask.ai.segmenter import POSITIVE, Box, Prediction, Segmenter
from easy_mask.ai.weights import ProgressFn, WeightFile
from easy_mask.core.geometry import Point

_RELEASE = "https://github.com/labelmeai/efficient-sam/releases/download/onnx-models-20231225/"

# name -> (encoder, decoder)
MODELS: dict[str, tuple[WeightFile, WeightFile]] = {
    "efficientsam-vitt": (
        WeightFile(
            _RELEASE + "efficient_sam_vitt_encoder.onnx",
            "7a73ee65aa2c37237c89b4b18e73082f757ffb173899609c5d97a2bbd4ebb02d",
            24_799_761,
        ),
        WeightFile(
            _RELEASE + "efficient_sam_vitt_decoder.onnx",
            "e1afe46232c3bfa3470a6a81c7d3181836a94ea89528aff4e0f2d2c611989efd",
            16_565_728,
        ),
    ),
    "efficientsam-vits": (
        WeightFile(
            _RELEASE + "efficient_sam_vits_encoder.onnx",
            "4cacbb23c6903b1acf87f1d77ed806b840800c5fcd4ac8f650cbffed474b8896",
            89_558_337,
        ),
        WeightFile(
            _RELEASE + "efficient_sam_vits_decoder.onnx",
            "4727baf23dacfb51d4c16795b2ac382c403505556d0284e84c6ff3d4e8e36f22",
            16_565_728,
        ),
    ),
}
DEFAULT_MODEL = "efficientsam-vitt"

INSTALL_HINT = (
    "AI Segment needs onnxruntime, which is an optional extra.\n"
    "Install it with:  uv sync --extra ai   (or: pip install 'easy-mask[ai]')"
)


class MissingDependencyError(ImportError):
    """onnxruntime is not installed."""


def import_onnxruntime() -> ModuleType:
    """Import onnxruntime, or explain how to install it.

    Raises:
        MissingDependencyError: With install instructions if it is missing.
    """
    try:
        return importlib.import_module("onnxruntime")
    except ImportError as exc:
        raise MissingDependencyError(INSTALL_HINT) from exc


class EfficientSam(Segmenter):
    """EfficientSAM encoder + decoder ONNX sessions on the CPU execution provider.

    Background clicks: EfficientSAM's prompt encoder gives label 0 no embedding of its
    own, and in practice a "negative" point acts almost like a positive one. So
    :meth:`predict` segments with the positive prompts only and then *cuts out* what a
    lone click on each negative point would select (its smallest candidate, i.e. the
    local part). On 10 images each of GrabCut / Berkeley / DAVIS this took NoC@90 from
    3.7 / 8.8 / 18.1 clicks to 1.4 / 3.8 / 4.9 (see docs/benchmarks.md).
    """

    def __init__(self, encoder_path: Path, decoder_path: Path, cache_size: int = 8):
        super().__init__(cache_size)
        ort = import_onnxruntime()
        providers = ["CPUExecutionProvider"]
        self._encoder = ort.InferenceSession(str(encoder_path), providers=providers)
        self._decoder = ort.InferenceSession(str(decoder_path), providers=providers)
        self._cuts: dict[Point, np.ndarray] = {}  # negative point -> region it removes

    def set_image(self, rgb: np.ndarray, key: str | None = None) -> None:
        """See :meth:`Segmenter.set_image`; also forgets cut regions of the last image."""
        super().set_image(rgb, key)
        self._cuts = {}

    def predict(
        self, points: Sequence[Point], labels: Sequence[int], box: Box | None = None
    ) -> list[Prediction]:
        """See :meth:`Segmenter.predict`; negative points cut regions out (class docstring).

        The 6-point limit applies to the positive points and box; every negative point
        is used (each costs one extra decode the first time, then is cached).
        """
        positives = [p for p, label in zip(points, labels, strict=True) if label == POSITIVE]
        negatives = [p for p, label in zip(points, labels, strict=True) if label != POSITIVE]
        predictions = super().predict(positives, [POSITIVE] * len(positives), box)
        if not predictions or not negatives:
            return predictions
        cut = np.zeros_like(predictions[0][0])
        for point in negatives:
            if point not in self._cuts:
                candidates = super().predict([point], [POSITIVE])
                self._cuts[point] = min(candidates, key=lambda c: int(c[0].sum()))[0]
            cut |= self._cuts[point]
        return [(mask & ~cut, score) for mask, score in predictions]

    def _encode(self, rgb: np.ndarray) -> Any:
        batch = (rgb.transpose(2, 0, 1)[None] / 255.0).astype(np.float32)
        return self._encoder.run(["image_embeddings"], {"batched_images": batch})[0]

    def _decode(
        self, embedding: Any, size: tuple[int, int], coords: np.ndarray, labels: np.ndarray
    ) -> list[Prediction]:
        masks, scores = self._decoder.run(
            ["output_masks", "iou_predictions"],
            {
                "image_embeddings": embedding,
                "batched_point_coords": coords[None, None],
                "batched_point_labels": labels[None, None],
                "orig_im_size": np.array(size, dtype=np.int64),
            },
        )
        return [(masks[0, 0, k] > 0, float(scores[0, 0, k])) for k in range(masks.shape[2])]


def load(name: str = DEFAULT_MODEL, progress: ProgressFn | None = None) -> EfficientSam:
    """Build an EfficientSAM segmenter, downloading its weights on first use.

    Args:
        name: A key of :data:`MODELS` (``efficientsam-vitt`` or ``efficientsam-vits``).
        progress: Download progress callback, see :data:`~easy_mask.ai.weights.ProgressFn`.

    Raises:
        KeyError: Unknown model name.
        MissingDependencyError: onnxruntime is not installed (checked before downloading).
        WeightsError: The weights could not be downloaded.
    """
    encoder, decoder = MODELS[name]
    import_onnxruntime()
    return EfficientSam(
        weights.ensure(encoder, progress=progress), weights.ensure(decoder, progress=progress)
    )
