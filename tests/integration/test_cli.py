import pytest

from easy_mask import __version__
from easy_mask.cli import main
from easy_mask.core import annotation as ann_io
from easy_mask.core.annotation import Annotation


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


def test_export_davis_command(image_dir, square, tmp_path):
    ann_io.save(Annotation("img1.png", 100, 80, [square]), image_dir / "img1.json")
    out = tmp_path / "davis"
    assert main(["export", str(image_dir), "--format", "davis", "--out", str(out)]) == 0
    assert (out / "Annotations" / image_dir.name / "img1.png").exists()
