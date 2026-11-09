"""The SAM 2-family backend's plumbing; the model itself runs in tests/integration."""

import numpy as np
import pytest
from PIL import Image

from easy_mask.propagation import backend
from easy_mask.propagation.backend import (
    MODELS,
    Sam2Propagator,
    _numbered_jpegs,
    checkpoint_path,
    config_path,
)


def test_every_model_has_a_checkpoint_name_and_known_config():
    assert MODELS["efficienttam_ti_512"].filename == "efficienttam_ti_512x512.pt"
    assert MODELS["sam2.1_hiera_tiny"].filename == "sam2.1_hiera_tiny.pt"
    assert config_path(MODELS["efficienttam_ti_512"]).is_file()  # bundled with easy-mask


def test_unknown_model_rejected():
    with pytest.raises(ValueError, match="efficienttam_ti_512"):
        Sam2Propagator("sam9000")


def test_missing_packages_raise_import_error(monkeypatch):
    monkeypatch.setattr(backend.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ImportError, match="torch"):
        Sam2Propagator()


def test_existing_checkpoint_is_not_downloaded(tmp_path, monkeypatch):
    monkeypatch.setattr(backend, "CHECKPOINT_DIR", tmp_path)
    (tmp_path / "efficienttam_ti_512x512.pt").write_bytes(b"weights")
    monkeypatch.setattr(backend.urllib.request, "urlretrieve", pytest.fail)
    assert checkpoint_path(MODELS["efficienttam_ti_512"]) == tmp_path / "efficienttam_ti_512x512.pt"


def test_missing_checkpoint_is_downloaded_atomically(tmp_path, monkeypatch):
    def fake_download(url, target):
        assert target.suffix == ".part"
        target.write_bytes(b"weights")

    monkeypatch.setattr(backend, "CHECKPOINT_DIR", tmp_path / "ckpt")
    monkeypatch.setattr(backend.urllib.request, "urlretrieve", fake_download)
    path = checkpoint_path(MODELS["sam2.1_hiera_tiny"])
    assert path.read_bytes() == b"weights" and not list(path.parent.glob("*.part"))


def test_frames_become_numbered_jpegs(tmp_path):
    jpg, png = tmp_path / "a.jpg", tmp_path / "b.png"
    Image.new("RGB", (8, 6), (255, 0, 0)).save(jpg)
    Image.new("RGBA", (8, 6), (0, 255, 0, 255)).save(png)
    out = tmp_path / "out"
    out.mkdir()
    _numbered_jpegs([jpg, png], out)
    assert (out / "00000.jpg").is_symlink()
    converted = np.array(Image.open(out / "00001.jpg"))
    assert converted.shape == (6, 8, 3) and converted[..., 1].mean() > 200
