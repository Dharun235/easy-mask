import numpy as np
import pytest

from easy_mask.core import rle

# Strings produced by pycocotools.mask.encode for the same arrays.
KNOWN_VECTORS = [
    (np.zeros((3, 4), bool), "<"),
    (np.ones((3, 4), bool), "0<"),
    (np.eye(3, dtype=bool), "013000"),
    (np.array([[0, 1, 1], [1, 0, 0]], bool), "121O0"),
    (np.zeros((0, 30), bool), "0"),
]


@pytest.mark.parametrize(("mask", "counts"), KNOWN_VECTORS)
def test_matches_pycocotools_known_vectors(mask, counts):
    assert rle.encode(mask) == {"size": list(mask.shape), "counts": counts}
    np.testing.assert_array_equal(rle.decode({"size": list(mask.shape), "counts": counts}), mask)


@pytest.mark.parametrize(
    "mask",
    [
        np.zeros((5, 7), bool),
        np.ones((5, 7), bool),
        np.pad(np.ones((1, 1), bool), ((2, 2), (3, 3))),  # single centre pixel
        np.pad(np.ones((1, 1), bool), ((0, 4), (0, 6))),  # single corner pixel
        np.random.default_rng(1).random((13, 17)) < 0.5,  # odd sizes
        np.ones((1, 1), bool),
    ],
    ids=["empty", "full", "centre-pixel", "corner-pixel", "odd-random", "1x1"],
)
def test_round_trip_edge_cases(mask):
    np.testing.assert_array_equal(rle.decode(rle.encode(mask)), mask)


def test_runs_are_column_major():
    mask = np.array([[1, 1, 0], [0, 0, 1]], bool)
    # Column by column: 1,0 | 1,0 | 0,1. Reading row by row would give [0, 2, 3, 1].
    assert rle.run_lengths(mask) == [0, 1, 1, 1, 2, 1]


def test_runs_always_start_with_background():
    assert rle.run_lengths(np.array([[1, 1, 0]], bool)) == [0, 2, 1]
    assert rle.run_lengths(np.array([[0, 1, 1]], bool)) == [1, 2]


def test_large_runs_use_negative_deltas():
    runs = [0, 5000, 3, 1]  # 3 - 5000 is negative: exercises sign extension
    assert rle.string_to_counts(rle.counts_to_string(runs)) == runs


def test_decode_accepts_bytes_and_uncompressed_counts():
    mask = np.array([[0, 1], [1, 1]], bool)
    encoded = rle.encode(mask)
    as_bytes = {"size": [2, 2], "counts": encoded["counts"].encode()}
    np.testing.assert_array_equal(rle.decode(as_bytes), mask)
    np.testing.assert_array_equal(rle.decode({"size": [2, 2], "counts": [1, 3]}), mask)


def test_decode_rejects_runs_of_wrong_total():
    with pytest.raises(ValueError, match="2x2"):
        rle.decode({"size": [2, 2], "counts": [1, 2]})
