"""Invariants of the classic pixel tools and display adjustments over random images."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from easy_mask.core import pixel_ops
from easy_mask.core.display import DisplaySettings, adjust

images = st.tuples(st.integers(2, 12), st.integers(2, 12)).flatmap(
    lambda hw: arrays(np.uint8, (*hw, 3))
)
seeds = st.tuples(st.floats(0, 11), st.floats(0, 11))
tolerances = st.integers(0, 255)
FAST = settings(max_examples=60, deadline=None)


@FAST
@given(images, st.sampled_from(["manual", "otsu"]), st.integers(0, 255), st.data())
def test_threshold_invert_is_complement_inside_region(rgb, method, level, data):
    region = data.draw(arrays(bool, rgb.shape[:2]))
    plain = pixel_ops.threshold(rgb, method, level, region=region)
    inverted = pixel_ops.threshold(rgb, method, level, invert=True, region=region)
    assert not (plain & ~region).any() and not (inverted & ~region).any()
    assert np.array_equal(plain | inverted, region) and not (plain & inverted).any()


@FAST
@given(images, seeds, tolerances, tolerances)
def test_magic_wand_contains_seed_and_grows_with_tolerance(rgb, seed, tol_a, tol_b):
    low, high = sorted((tol_a, tol_b))
    small = pixel_ops.magic_wand(rgb, seed, low)
    large = pixel_ops.magic_wand(rgb, seed, high)
    x = min(int(seed[0]), rgb.shape[1] - 1)
    y = min(int(seed[1]), rgb.shape[0] - 1)
    assert small[y, x] and np.all(large[small])


@FAST
@given(images)
def test_magic_wand_with_full_tolerance_selects_everything(rgb):
    assert pixel_ops.magic_wand(rgb, (0, 0), 255).all()


@FAST
@given(images, seeds, st.integers(0, 90), st.integers(0, 255))
def test_color_range_contains_picked_pixel_and_grows(rgb, point, hue_tol, sv_tol):
    color = pixel_ops.color_at(rgb, point)
    tight = pixel_ops.color_range(rgb, color, hue_tol, sv_tol, sv_tol)
    loose = pixel_ops.color_range(rgb, color, 90, 255, 255)
    x = min(int(point[0]), rgb.shape[1] - 1)
    y = min(int(point[1]), rgb.shape[0] - 1)
    assert tight[y, x] and loose.all() and np.all(loose[tight])


@FAST
@given(images, st.integers(-255, 255), st.floats(0.1, 4), st.floats(0.1, 4))
def test_display_adjustment_keeps_shape_and_order(rgb, brightness, contrast, gamma):
    out = adjust(rgb, DisplaySettings(brightness=brightness, contrast=contrast, gamma=gamma))
    assert out.shape == rgb.shape and out.dtype == np.uint8
    # Monotone: a brighter input pixel never becomes darker than a dimmer one.
    flat_in, flat_out = rgb.ravel(), out.ravel()
    order = np.argsort(flat_in, kind="stable")
    assert np.all(np.diff(flat_out[order].astype(int)) >= 0)
