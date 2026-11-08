import json

import numpy as np
import pycocotools.mask as coco_mask
import pytest
from pycocotools.coco import COCO

from easy_mask import __version__
from easy_mask.cli import build_gui_parser, main
from easy_mask.core import annotation as ann_io
from easy_mask.core import masks
from easy_mask.core.annotation import Annotation

# pycocotools 2.0.x still calls np.array(copy=False) internally under numpy 2.
pytestmark = pytest.mark.filterwarnings("ignore:__array__ implementation:DeprecationWarning")


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_export_command(image_dir, square, tmp_path, capsys):
    ann_io.save(Annotation("img1.png", 100, 80, [square]), image_dir / "img1.json")
    out = tmp_path / "out"
    assert main(["export", str(image_dir), "--format", "coco", "--out", str(out)]) == 0
    assert (out / "annotations.json").exists()
    assert "Exported 1 images" in capsys.readouterr().out


def test_export_command_without_annotations_fails(image_dir, tmp_path):
    assert main(["export", str(image_dir), "--format", "yolo", "--out", str(tmp_path / "o")]) == 1


def save_ring(image_dir, square):
    """img1: a cat square plus a 'donut' mask with a hole."""
    full = np.zeros((80, 100), bool)
    full[10:50, 20:60] = True
    full[20:40, 30:50] = False
    shapes = [square, masks.to_shape("donut", full)]
    ann_io.save(Annotation("img1.png", 100, 80, shapes), image_dir / "img1.json")
    return full


def test_export_coco_rle_loads_in_pycocotools(image_dir, square, tmp_path, capsys):
    full = save_ring(image_dir, square)
    out = tmp_path / "out"
    assert main(["export", str(image_dir), "--format", "coco", "--rle", "--out", str(out)]) == 0
    assert "as coco-rle" in capsys.readouterr().out

    coco = COCO(str(out / "annotations.json"))
    (donut_id,) = coco.getCatIds(catNms=["donut"])
    (ann,) = coco.loadAnns(coco.getAnnIds(catIds=[donut_id]))
    np.testing.assert_array_equal(coco.annToMask(ann).astype(bool), full)
    rle = {"size": ann["segmentation"]["size"], "counts": ann["segmentation"]["counts"].encode()}
    assert coco_mask.area(rle) == ann["area"] == int(full.sum())
    assert coco_mask.toBbox(rle).tolist() == ann["bbox"]


def test_rle_flag_rejected_for_other_formats(image_dir, square, tmp_path, capsys):
    save_ring(image_dir, square)
    args = ["export", str(image_dir), "--format", "yolo", "--rle", "--out", str(tmp_path / "o")]
    assert main(args) == 2
    assert "--rle only applies" in capsys.readouterr().err


def test_stats_table(image_dir, square, capsys):
    save_ring(image_dir, square)
    assert main(["stats", str(image_dir)]) == 0
    out = capsys.readouterr().out
    assert "Images   3  (annotated 1, unannotated 2, reviewed 0)" in out
    assert "donut" in out and "label list" not in out


def test_stats_json(image_dir, square, capsys):
    save_ring(image_dir, square)
    assert main(["stats", str(image_dir), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["labels"]["donut"]["area"] == 1200
    assert data["objects"] == 2


def test_stats_label_validation_exit_code(image_dir, square, capsys):
    save_ring(image_dir, square)
    assert main(["stats", str(image_dir), "--labels", "cat,donut"]) == 0
    assert "All labels are in the label list." in capsys.readouterr().out
    assert main(["stats", str(image_dir), "--labels", "cat"]) == 1
    assert "donut: img1.png" in capsys.readouterr().out


def test_strict_labels_needs_labels(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--strict-labels"])
    assert exc.value.code == 2
    assert "--strict-labels needs --labels" in capsys.readouterr().err


def test_ai_model_option():
    parser = build_gui_parser()
    assert parser.parse_args([]).ai_model == "efficientsam-vitt"
    assert parser.parse_args(["--ai-model", "efficientsam-vits"]).ai_model == "efficientsam-vits"
    with pytest.raises(SystemExit):
        parser.parse_args(["--ai-model", "nope"])


def test_export_davis_command(image_dir, square, tmp_path):
    ann_io.save(Annotation("img1.png", 100, 80, [square]), image_dir / "img1.json")
    out = tmp_path / "davis"
    assert main(["export", str(image_dir), "--format", "davis", "--out", str(out)]) == 0
    assert (out / "Annotations" / image_dir.name / "img1.png").exists()
