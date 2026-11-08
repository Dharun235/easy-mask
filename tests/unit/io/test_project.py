from easy_mask.io import project as project_io
from easy_mask.io.project import PROJECT_FILE, Project


def test_save_load_roundtrip(tmp_path):
    original = Project(
        labels=["cat", "dog"], flags=["blurry"], active_label="dog", output_dir="labels"
    )
    project_io.save(original, tmp_path)
    assert (tmp_path / PROJECT_FILE).exists()
    assert project_io.load(tmp_path) == original


def test_missing_or_broken_file_gives_none(tmp_path):
    assert project_io.load(tmp_path) is None
    (tmp_path / PROJECT_FILE).write_text("{ broken")
    assert project_io.load(tmp_path) is None
    (tmp_path / PROJECT_FILE).write_text("[1, 2]")
    assert project_io.load(tmp_path) is None


def test_unknown_keys_are_ignored(tmp_path):
    (tmp_path / PROJECT_FILE).write_text('{"labels": ["a"], "future_option": 1}')
    assert project_io.load(tmp_path) == Project(labels=["a"])


def test_relative_and_resolve_dir(tmp_path):
    inside, outside = tmp_path / "labels", tmp_path.parent / "elsewhere"
    assert project_io.relative_dir(inside, tmp_path) == "labels"
    assert project_io.relative_dir(outside, tmp_path) == str(outside.resolve())
    assert project_io.relative_dir(None, tmp_path) is None
    assert project_io.resolve_dir("labels", tmp_path) == tmp_path / "labels"
    assert project_io.resolve_dir(str(outside.resolve()), tmp_path) == outside.resolve()
    assert project_io.resolve_dir(None, tmp_path) is None
