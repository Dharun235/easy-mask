import colorsys

from easy_mask.core.labels import label_color, palette_color, parse_name_list


def test_label_color_uses_list_position():
    labels = ["cat", "dog", "box"]
    assert label_color("dog", labels) == palette_color(1)
    assert label_color("cat") == label_color("cat")  # unknown labels: stable hash colour
    assert all(0 <= c <= 255 for c in label_color("cat"))


def test_first_palette_colours_have_distinct_hues():
    hues = [colorsys.rgb_to_hsv(*(c / 255 for c in palette_color(i)))[0] for i in range(8)]
    for i, a in enumerate(hues):
        for b in hues[i + 1 :]:
            assert min(abs(a - b), 1 - abs(a - b)) > 0.05


def test_parse_comma_list_dedupes_and_strips():
    assert parse_name_list(" cat, dog,,cat ") == ["cat", "dog"]
    assert parse_name_list(None) == []


def test_parse_file(tmp_path):
    path = tmp_path / "labels.txt"
    path.write_text("cat\n\ndog\n")
    assert parse_name_list(str(path)) == ["cat", "dog"]
