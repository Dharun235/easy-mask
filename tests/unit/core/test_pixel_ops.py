import numpy as np
import pytest

from easy_mask.core import masks, pixel_ops


@pytest.fixture
def scene() -> np.ndarray:
    """Dark gray 60x80 image with a bright red 20x30 block at x 20-49, y 15-34."""
    rgb = np.full((60, 80, 3), 40, np.uint8)
    rgb[15:35, 20:50] = (220, 30, 30)
    return rgb


@pytest.fixture
def block() -> np.ndarray:
    mask = np.zeros((60, 80), bool)
    mask[15:35, 20:50] = True
    return mask


# ---------------------------------------------------------------------- channel


@pytest.mark.parametrize("name", pixel_ops.CHANNELS)
def test_channel_is_uint8_plane(scene, name):
    plane = pixel_ops.channel(scene, name)
    assert plane.shape == (60, 80) and plane.dtype == np.uint8


def test_channel_picks_rgb_planes(scene):
    assert pixel_ops.channel(scene, "red")[20, 25] == 220
    assert pixel_ops.channel(scene, "green")[20, 25] == 30
    assert pixel_ops.channel(scene, "blue")[0, 0] == 40


def test_channel_saturation_and_hue(scene):
    assert pixel_ops.channel(scene, "saturation")[0, 0] == 0  # gray
    assert pixel_ops.channel(scene, "saturation")[20, 25] > 150  # strong red
    assert pixel_ops.channel(scene, "hue")[20, 25] in (0, 179)


# ---------------------------------------------------------------------- threshold


@pytest.mark.parametrize("plane", ["gray", "red", "value"])
def test_otsu_finds_the_bright_block(scene, block, plane):
    assert np.array_equal(pixel_ops.threshold(scene, "otsu", plane=plane), block)


def test_manual_threshold_and_invert(scene, block):
    assert np.array_equal(pixel_ops.threshold(scene, "manual", level=100, plane="red"), block)
    assert np.array_equal(
        pixel_ops.threshold(scene, "manual", level=100, plane="red", invert=True), ~block
    )
    assert not pixel_ops.threshold(scene, "manual", level=255).any()


def test_threshold_is_limited_to_region(scene, block):
    region = np.zeros_like(block)
    region[:, :35] = True
    result = pixel_ops.threshold(scene, "otsu", plane="red", region=region)
    assert np.array_equal(result, block & region)


def test_otsu_level_comes_from_region_only(scene):
    # Inside a region of only background, Otsu must still split that region's values.
    scene = scene.copy()
    scene[0:10, 0:10] = 10
    region = np.zeros(scene.shape[:2], bool)
    region[0:12, 0:12] = True
    result = pixel_ops.threshold(scene, "otsu", region=region)
    assert result.sum() == 144 - 100  # the 40-gray pixels, not the dark ones


def test_otsu_with_empty_region_is_empty(scene):
    empty = np.zeros(scene.shape[:2], bool)
    assert not pixel_ops.threshold(scene, "otsu", region=empty).any()


def test_adaptive_threshold_compares_with_local_mean(scene):
    result = pixel_ops.threshold(scene, "adaptive", block_size=4, offset=5)  # made odd: 5
    assert result[0, 0] and result[25, 35]  # flat areas pass (value > mean - offset)
    assert not result[25, 18]  # dark pixel next to the bright block fails


def test_otsu_level_splits_two_values():
    assert 10 <= pixel_ops.otsu_level(np.array([10] * 50 + [200] * 50, np.uint8)) < 200


# ---------------------------------------------------------------------- colour


def test_magic_wand_selects_connected_region(scene, block):
    assert np.array_equal(pixel_ops.magic_wand(scene, (30, 20), tolerance=10), block)
    assert np.array_equal(pixel_ops.magic_wand(scene, (0, 0), tolerance=10), ~block)


def test_magic_wand_is_not_fooled_by_disconnected_same_colour(scene, block):
    scene[50:55, 60:70] = (220, 30, 30)
    assert np.array_equal(pixel_ops.magic_wand(scene, (30, 20), tolerance=10), block)


def test_magic_wand_tolerance_grows_region(scene):
    scene[10:15, 20:50] = (200, 30, 30)  # slightly darker band above the block
    small = pixel_ops.magic_wand(scene, (30, 20), tolerance=5)
    large = pixel_ops.magic_wand(scene, (30, 20), tolerance=30)
    assert large.sum() > small.sum() and np.all(large[small])


def test_magic_wand_clamps_seed_to_image(scene):
    assert pixel_ops.magic_wand(scene, (500, -3), tolerance=0)[0, 79]


def test_color_at(scene):
    assert pixel_ops.color_at(scene, (25.7, 20.2)) == (220, 30, 30)


def test_color_range_finds_every_matching_pixel(scene, block):
    scene[50:55, 60:70] = (215, 35, 30)
    result = pixel_ops.color_range(scene, (220, 30, 30))
    assert result[block].all() and result[50:55, 60:70].all() and not result[0, 0]


def test_color_range_hue_wraps_around():
    rgb = np.zeros((1, 2, 3), np.uint8)
    rgb[0, 0] = (255, 0, 10)  # red, hue just below 180
    rgb[0, 1] = (255, 10, 0)  # red, hue just above 0
    assert pixel_ops.color_range(rgb, (255, 0, 0), hue_tol=5).all()


# ---------------------------------------------------------------------- graph-based


def test_grabcut_from_box(scene, block):
    result = pixel_ops.grabcut(scene, (10, 5, 60, 45))
    assert masks.iou(result, block) > 0.95


def test_grabcut_degenerate_box_is_empty(scene):
    assert not pixel_ops.grabcut(scene, (10, 10, 11, 40)).any()
    assert not pixel_ops.grabcut(scene, (100, 100, 200, 200)).any()


def test_grabcut_box_covering_whole_image(scene, block):
    result = pixel_ops.grabcut(scene, (0, 0, 80, 60))
    assert masks.iou(result, block) > 0.9 or masks.iou(result, ~block) > 0.9


def test_grabcut_scribbles_are_respected(scene):
    fg = np.zeros(scene.shape[:2], bool)
    fg[2:4, 2:4] = True  # a background patch forced to be object
    bg = np.zeros_like(fg)
    bg[20:22, 30:32] = True  # an object patch forced to be background
    result = pixel_ops.grabcut(scene, (0, 0, 70, 50), foreground=fg, background=bg)
    assert result[fg].all() and not result[bg].any()


# ---------------------------------------------------------------------- superpixels
