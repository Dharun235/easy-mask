"""COCO RLE must be lossless and byte-identical to pycocotools for any mask."""

import numpy as np
import pycocotools.mask as coco_mask
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from easy_mask.core import rle

# pycocotools 2.0.x still calls np.array(copy=False) internally under numpy 2.
pytestmark = pytest.mark.filterwarnings("ignore:__array__ implementation:DeprecationWarning")

masks = st.tuples(st.integers(0, 40), st.integers(0, 40)).flatmap(lambda shape: arrays(bool, shape))


@given(masks)
def test_decode_inverts_encode(mask):
    np.testing.assert_array_equal(rle.decode(rle.encode(mask)), mask)


@given(masks)
def test_counts_match_pycocotools(mask):
    reference = coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))
    ours = rle.encode(mask)
    assert ours["counts"] == reference["counts"].decode("ascii")
    assert ours["size"] == reference["size"]


@given(masks)
def test_pycocotools_decodes_our_rle(mask):
    ours = rle.encode(mask)
    decoded = coco_mask.decode({"size": ours["size"], "counts": ours["counts"].encode()})
    np.testing.assert_array_equal(decoded.astype(bool), mask)
