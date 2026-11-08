"""The simulated-click protocol of benchmarks/clicks.py, with the fake segmenter."""

import numpy as np
import pytest
from benchmarks import clicks
from benchmarks.clicks import MAX_CLICKS, Result, next_click, simulate, summarize
from benchmarks.datasets import load_ignore_region
from PIL import Image


def square(size=60, lo=10, hi=50):
    gt = np.zeros((size, size), dtype=bool)
    gt[lo:hi, lo:hi] = True
    return gt


def test_first_click_is_the_object_point_farthest_from_its_boundary():
    (x, y), positive = next_click(square(), np.zeros((60, 60), dtype=bool))
    assert positive and 28 <= x <= 31 and 28 <= y <= 31


def test_missed_foreground_gives_a_positive_click_inside_it():
    pred = square()
    pred[10:50, 30:50] = False  # right half missing
    (x, _), positive = next_click(square(), pred)
    assert positive and x >= 30


def test_false_foreground_gives_a_negative_click():
    pred = square() | square(lo=0, hi=60) & (np.arange(60) >= 50)[None, :]
    (x, _), positive = next_click(square(), pred)
    assert not positive and x >= 50


def test_ignored_pixels_attract_no_click():
    gt = square()
    ignore = np.zeros_like(gt)
    ignore[10:50, 30:50] = True
    pred = gt & ~ignore
    assert next_click(gt, pred, ignore) is None


def test_perfect_prediction_needs_no_click():
    assert next_click(square(), square()) is None


def test_simulate_stops_once_good_enough(fake_segmenter):
    gt = np.zeros((60, 60), dtype=bool)
    ys, xs = np.mgrid[:60, :60]
    gt[(xs - 30) ** 2 + (ys - 30) ** 2 <= fake_segmenter.RADIUS**2] = True  # what one click gives
    result = simulate(fake_segmenter, np.zeros((60, 60, 3), np.uint8), gt)
    assert result.ious[0] == 1.0 and len(result.ious) == 1  # perfect after one click
    assert result.clicks_to(0.9) == 1 and result.iou_at(5) == 1.0
    assert result.encode_s >= 0 and len(result.click_s) == 1


def test_simulate_gives_up_after_max_clicks(fake_segmenter):
    gt = np.zeros((60, 60), dtype=bool)
    gt[::2, ::2] = True  # a checkerboard of dots no disc can match
    result = simulate(fake_segmenter, np.zeros((60, 60, 3), np.uint8), gt)
    assert len(result.ious) == MAX_CLICKS and result.clicks_to(0.85) == MAX_CLICKS


def test_ignore_region_is_left_out_of_iou(fake_segmenter):
    gt = square()
    ignore = np.zeros_like(gt)
    ignore[:, 40:] = True
    assert clicks.masked_iou(gt & ~ignore, gt, ignore) == 1.0
    assert clicks.masked_iou(gt & ~ignore, gt, None) < 1.0


def test_summary_metrics():
    results = [
        Result(ious=[0.5, 0.86, 0.95, 0.95, 0.95], encode_s=0.2, click_s=[0.01] * 5),
        Result(ious=[0.8] * MAX_CLICKS, encode_s=0.4, click_s=[0.03] * MAX_CLICKS),
    ]
    row = summarize(results)
    assert row["n"] == 2
    assert row["NoC@85"] == pytest.approx((2 + MAX_CLICKS) / 2)
    assert row["NoC@90"] == pytest.approx((3 + MAX_CLICKS) / 2)
    assert row["NoF@90"] == 1
    assert row["mIoU@1"] == pytest.approx(0.65) and row["mIoU@5"] == pytest.approx(0.875)
    assert row["encode ms"] == pytest.approx(300)
    assert row["click ms"] == pytest.approx(1000 * (5 * 0.01 + 20 * 0.03) / 25)
    line = clicks.format_row("Toy", row)
    assert line.startswith("| Toy | 2 | 11.00 | 11.50 | 1 | 0.65 |")


def test_main_without_data_says_where_to_put_it(monkeypatch, capsys):
    monkeypatch.setattr(clicks, "available", lambda: [])
    assert clicks.main([]) == 1
    assert "data/benchmarks" in capsys.readouterr().out


def test_grabcut_boundary_band_is_the_ignore_region(tmp_path):
    raw = np.zeros((4, 4), dtype=np.uint8)
    raw[1:3, 1:3] = 255
    raw[0, :] = 128
    Image.fromarray(raw).save(tmp_path / "gc.bmp")
    Image.fromarray((raw == 255).astype(np.uint8)).save(tmp_path / "davis.png")
    assert load_ignore_region(tmp_path / "gc.bmp").sum() == 4
    assert not load_ignore_region(tmp_path / "davis.png").any()
