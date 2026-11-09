"""Mask propagation with a SAM 2-family video predictor (needs the ``video`` extra).

Two checkpoints share SAM 2's video API: EfficientTAM-Ti 512 (the default: ~20x faster on
CPU, see docs/benchmarks.md) and SAM 2.1 Hiera-Tiny. Weights download on first use to
``~/.cache/easy-mask/checkpoints``. PyTorch and the model packages are imported lazily so
the rest of easy-mask never loads them.

The model is built from its YAML config with OmegaConf + Hydra's ``instantiate`` rather
than the packages' own builders: EfficientTAM's wheel ships without its configs (one is
bundled here) and both packages claim Hydra's global config search path.
"""

import importlib.util
import shutil
import tempfile
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from easy_mask.propagation.tracker import FrameMasks

CHECKPOINT_DIR = Path.home() / ".cache" / "easy-mask" / "checkpoints"
CONFIG_DIR = Path(__file__).parent / "configs"


@dataclass(frozen=True)
class ModelSpec:
    """Where a checkpoint comes from and how to build its predictor."""

    url: str
    config: str  # "package:relative/path.yaml" or a file name in CONFIG_DIR
    predictor: str  # dotted class path of the video predictor

    @property
    def filename(self) -> str:
        """Checkpoint file name in :data:`CHECKPOINT_DIR`."""
        return self.url.rsplit("/", 1)[-1]


MODELS = {
    "efficienttam_ti_512": ModelSpec(
        url="https://huggingface.co/yunyangx/efficient-track-anything/resolve/main/"
        "efficienttam_ti_512x512.pt",
        config="efficienttam_ti_512x512.yaml",
        predictor="efficient_track_anything.efficienttam_video_predictor."
        "EfficientTAMVideoPredictor",
    ),
    "sam2.1_hiera_tiny": ModelSpec(
        url="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
        config="sam2:configs/sam2.1/sam2.1_hiera_t.yaml",
        predictor="sam2.sam2_video_predictor.SAM2VideoPredictor",
    ),
}
DEFAULT_MODEL = "efficienttam_ti_512"

# Same post-processing switches SAM 2's own builder applies for video.
PREDICTOR_OPTIONS: dict[str, Any] = {
    "sam_mask_decoder_extra_args": {
        "dynamic_multimask_via_stability": True,
        "dynamic_multimask_stability_delta": 0.05,
        "dynamic_multimask_stability_thresh": 0.98,
    },
    "binarize_mask_from_pts_for_mem_enc": True,
    "fill_hole_area": 8,
}


def pick_device() -> str:
    """The fastest available torch device: CUDA, then Apple MPS, then CPU."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def checkpoint_path(spec: ModelSpec) -> Path:
    """Local checkpoint file, downloading it first if missing."""
    path = CHECKPOINT_DIR / spec.filename
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".part")
        urllib.request.urlretrieve(spec.url, partial)
        partial.replace(path)
    return path


def config_path(spec: ModelSpec) -> Path:
    """Resolve a spec's config: a file inside an installed package, or a bundled one."""
    if ":" not in spec.config:
        return CONFIG_DIR / spec.config
    package, relative = spec.config.split(":", 1)
    module = __import__(package)
    return Path(str(module.__file__)).parent / relative


def build_predictor(name: str, device: str) -> Any:
    """Instantiate a video predictor with its weights loaded, in eval mode."""
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    spec = MODELS[name]
    model_config = OmegaConf.load(config_path(spec)).model
    model_config._target_ = spec.predictor
    model_config.update(PREDICTOR_OPTIONS)
    if "compile_image_encoder" in model_config:
        model_config.compile_image_encoder = False  # torch.compile needs a recent GPU
    predictor = instantiate(model_config, _recursive_=True)
    state = torch.load(checkpoint_path(spec), map_location="cpu", weights_only=True)
    predictor.load_state_dict(state["model"])
    return predictor.to(device).eval()


def _numbered_jpegs(frames: list[Path], folder: Path) -> None:
    """Lay ``frames`` out as ``00000.jpg, 00001.jpg, ...``, the input SAM 2 expects."""
    for position, frame in enumerate(frames):
        target = folder / f"{position:05d}.jpg"
        if frame.suffix.lower() in (".jpg", ".jpeg"):
            target.symlink_to(frame.resolve())
        else:
            with Image.open(frame) as image:
                image.convert("RGB").save(target, quality=95)


class Sam2Propagator:
    """A :class:`~easy_mask.propagation.tracker.Propagator` backed by SAM 2's video API.

    Args:
        model: A key of :data:`MODELS`.
        device: Torch device; picked with :func:`pick_device` when None.

    Raises:
        ImportError: If PyTorch or the model's package is missing (no ``video`` extra).
    """

    def __init__(self, model: str = DEFAULT_MODEL, device: str | None = None):
        if model not in MODELS:
            raise ValueError(f"Unknown model {model!r}; choose from {sorted(MODELS)}")
        for package in ("torch", MODELS[model].predictor.split(".")[0]):
            if importlib.util.find_spec(package) is None:
                raise ImportError(f"{package} is not installed")
        self.model = model
        self.device = device
        self._predictor: Any = None

    def _load(self) -> Any:
        if self._predictor is None:
            self.device = self.device or pick_device()
            self._predictor = build_predictor(self.model, self.device)
        return self._predictor

    def propagate(self, frames: list[Path], prompts: FrameMasks) -> Iterator[FrameMasks]:
        """Track prompted objects through ``frames``; see ``Propagator.propagate``."""
        import torch

        predictor = self._load()
        folder = Path(tempfile.mkdtemp(prefix="easy-mask-frames-"))
        try:
            _numbered_jpegs(frames, folder)
            with torch.inference_mode():
                state = predictor.init_state(str(folder), offload_video_to_cpu=True)
                for position, by_id in sorted(prompts.items()):
                    for object_id, full in by_id.items():
                        predictor.add_new_mask(state, position, object_id, full)
                for position, object_ids, logits in predictor.propagate_in_video(state):
                    if position == 0:
                        continue
                    found = (logits[:, 0] > 0).cpu().numpy()
                    yield {
                        position: {
                            int(i): np.asarray(m) for i, m in zip(object_ids, found, strict=True)
                        }
                    }
        finally:
            shutil.rmtree(folder, ignore_errors=True)
