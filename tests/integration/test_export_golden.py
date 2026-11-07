"""Golden-file tests: exports must match checked-in expected output.

They catch *any* change to the output format, intended or not. Numbers are compared with
a tiny tolerance because circle vertices come from cos/sin, whose last bits can differ
between platforms; everything else must match exactly. When a change is
intended, regenerate and review the diff:

    UPDATE_GOLDEN=1 uv run pytest tests/integration/test_export_golden.py
"""

import json
import os
import shutil

import numpy as np
import pytest
from PIL import Image

from easy_mask.core import annotation as ann_io
from easy_mask.io import export

UPDATE = os.environ.get("UPDATE_GOLDEN") == "1"


def approx_tree(value):
    """``pytest.approx`` applied to every float inside nested dicts/lists."""
    if isinstance(value, dict):
        return {k: approx_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [approx_tree(v) for v in value]
    if isinstance(value, float):
        return pytest.approx(value, rel=1e-9, abs=1e-9)
    return value


@pytest.fixture
def scene(tmp_path, data_dir):
    """The labelme sample, paired with a blank image of the right size."""
    folder = tmp_path / "images"
    folder.mkdir()
    Image.new("RGB", (160, 120)).save(folder / "scene.jpg")
    shutil.copy(data_dir / "labelme_v5_sample.json", folder / "scene.json")
    return export.collect(folder)


def read_golden(actual: str, golden_path) -> str:
    """Return the golden text, (re)writing it first when UPDATE_GOLDEN=1."""
    if UPDATE:
        golden_path.write_text(actual, encoding="utf-8")
    assert golden_path.exists(), f"missing golden file; run with UPDATE_GOLDEN=1: {golden_path}"
    return golden_path.read_text(encoding="utf-8")


def test_coco_matches_golden(scene, tmp_path, data_dir):
    export.export_coco(scene, tmp_path / "out")
    actual = (tmp_path / "out" / "annotations.json").read_text(encoding="utf-8")
    golden = read_golden(actual, data_dir / "golden" / "coco.json")
    assert json.loads(actual) == approx_tree(json.loads(golden))


def test_yolo_matches_golden(scene, tmp_path, data_dir):
    export.export_yolo(scene, tmp_path / "out")
    actual = (tmp_path / "out" / "labels" / "scene.txt").read_text(encoding="utf-8")
    golden = read_golden(actual, data_dir / "golden" / "yolo_labels_scene.txt")
    for got, want in zip(actual.splitlines(), golden.splitlines(), strict=True):
        got_cls, *got_xy = got.split()
        want_cls, *want_xy = want.split()
        assert got_cls == want_cls
        assert [float(v) for v in got_xy] == pytest.approx([float(v) for v in want_xy], abs=2e-6)
    yaml = (tmp_path / "out" / "data.yaml").read_text(encoding="utf-8")
    assert yaml == read_golden(yaml, data_dir / "golden" / "yolo_data.yaml")


def test_masks_match_golden(scene, tmp_path, data_dir):
    export.export_masks(scene, tmp_path / "out")
    for kind in ("semantic", "instance"):
        actual = np.array(Image.open(tmp_path / "out" / kind / "scene.png"))
        golden = data_dir / "golden" / f"{kind}.npy"
        if UPDATE:
            np.save(golden, actual)
        np.testing.assert_array_equal(actual, np.load(golden))


def test_golden_files_are_sane(data_dir):
    """Guard against blessing a broken golden file."""
    coco = json.loads((data_dir / "golden" / "coco.json").read_text())
    assert [c["name"] for c in coco["categories"]] == ["ball", "lane", "nose", "person", "pole"]
    assert len(coco["annotations"]) == 2  # grouped person (2 parts) + ball
    semantic = np.load(data_dir / "golden" / "semantic.npy")
    assert set(np.unique(semantic)) == {0, 1, 2, 3, 4, 5}
    assert ann_io.load(data_dir / "labelme_v5_sample.json").shapes
