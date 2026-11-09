"""Click-to-segment models (no Qt here, so the logic runs and tests headless).

* :mod:`~easy_mask.ai.segmenter` - the :class:`Segmenter` interface: encode an image once,
  then turn click / box prompts into masks.
* :mod:`~easy_mask.ai.efficient_sam` - EfficientSAM via onnxruntime (installed by default).
* :mod:`~easy_mask.ai.weights` - download model files to a per-user cache, sha256-checked.
"""
