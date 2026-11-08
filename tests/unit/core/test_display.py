import numpy as np
import pytest

from easy_mask.core.display import DisplaySettings, adjust


@pytest.fixture
def ramp() -> np.ndarray:
    """1x256 gray ramp with a coloured second row."""
    rgb = np.zeros((2, 256, 3), np.uint8)
    rgb[0] = np.arange(256)[:, None]
    rgb[1] = (200, 50, 10)
    return rgb


def test_default_settings_change_nothing(ramp):
    settings = DisplaySettings()
    assert not settings.changes_image
    assert np.array_equal(adjust(ramp, settings), ramp)


def test_opacity_alone_does_not_change_image():
    assert not DisplaySettings(mask_opacity=0.3).changes_image


@pytest.mark.parametrize(
    "settings",
    [
        DisplaySettings(brightness=10),
        DisplaySettings(contrast=2.0),
        DisplaySettings(gamma=2.0),
        DisplaySettings(clahe=True),
        DisplaySettings(view_channel="red"),
    ],
)
def test_each_setting_changes_image(settings):
    assert settings.changes_image


def test_brightness_shifts_and_clips(ramp):
    out = adjust(ramp, DisplaySettings(brightness=50))
    assert out[0, 0, 0] == 50 and out[0, 255, 0] == 255


def test_contrast_stretches_around_mid_gray(ramp):
    out = adjust(ramp, DisplaySettings(contrast=2.0))
    assert out[0, 128, 0] == 128 and out[0, 100, 0] == 72 and out[0, 10, 0] == 0


def test_gamma_above_one_brightens_shadows(ramp):
    out = adjust(ramp, DisplaySettings(gamma=2.0))
    assert out[0, 64, 0] > 64 and out[0, 0, 0] == 0 and out[0, 255, 0] == 255


def test_channel_view_is_grayscale(ramp):
    out = adjust(ramp, DisplaySettings(view_channel="red"))
    assert tuple(out[1, 0]) == (200, 200, 200)


def test_clahe_keeps_shape_and_type(ramp):
    out = adjust(ramp, DisplaySettings(clahe=True))
    assert out.shape == ramp.shape and out.dtype == np.uint8


def test_adjust_does_not_modify_input(ramp):
    before = ramp.copy()
    adjust(ramp, DisplaySettings(brightness=40, clahe=True, view_channel="value"))
    assert np.array_equal(ramp, before)
