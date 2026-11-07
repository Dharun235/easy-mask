from easy_mask.core.labels import label_color, parse_name_list


def test_label_color_is_stable_and_distinct():
    assert label_color("cat") == label_color("cat")
    assert label_color("cat") != label_color("dog")
    assert all(0 <= c <= 255 for c in label_color("cat"))


def test_parse_comma_list_dedupes_and_strips():
    assert parse_name_list(" cat, dog,,cat ") == ["cat", "dog"]
    assert parse_name_list(None) == []


def test_parse_file(tmp_path):
    path = tmp_path / "labels.txt"
    path.write_text("cat\n\ndog\n")
    assert parse_name_list(str(path)) == ["cat", "dog"]
