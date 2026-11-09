"""COCO run-length encoding (RLE) of binary masks, compatible with pycocotools.

A COCO RLE is ``{"size": [height, width], "counts": "..."}``. The mask is read in
column-major (Fortran) order, top to bottom then left to right, and stored as alternating
run lengths of 0s and 1s, always starting with a (possibly empty) run of 0s.

``counts`` is pycocotools' compressed string: each run (from the third on, as the
difference to the run two places earlier) is written as a signed number in 5-bit
chunks, each chunk offset by 48 to land on a printable ASCII character. This module
is a pure-numpy port of ``rleToString`` / ``rleFrString`` from the COCO API so exports
need no compiled dependency.
"""

from typing import Any

import numpy as np


def run_lengths(mask: np.ndarray) -> list[int]:
    """Alternating 0/1 run lengths of ``mask`` in column-major order, starting with 0s."""
    flat = np.asarray(mask, dtype=bool).ravel(order="F")
    if flat.size == 0:
        return [0]  # pycocotools writes a single empty run for a 0-pixel mask
    change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    bounds = np.concatenate(([0], change, [flat.size]))
    runs = np.diff(bounds).tolist()
    return [0, *runs] if flat[0] else runs


def counts_to_string(runs: list[int]) -> str:
    """Compress run lengths into pycocotools' ASCII ``counts`` string."""
    chars = []
    for i, run in enumerate(runs):
        x = run - runs[i - 2] if i > 2 else run
        more = True
        while more:
            chunk = x & 0x1F
            x >>= 5  # arithmetic shift: negative numbers end at -1
            more = x != -1 if chunk & 0x10 else x != 0
            if more:
                chunk |= 0x20
            chars.append(chr(chunk + 48))
    return "".join(chars)


def string_to_counts(text: str) -> list[int]:
    """Decompress a pycocotools ``counts`` string back into run lengths."""
    runs: list[int] = []
    pos = 0
    while pos < len(text):
        x, shift, more = 0, 0, True
        while more:
            chunk = ord(text[pos]) - 48
            x |= (chunk & 0x1F) << shift
            more = bool(chunk & 0x20)
            pos += 1
            shift += 5
            if not more and chunk & 0x10:
                x |= -1 << shift  # sign-extend
        if len(runs) > 2:
            x += runs[-2]
        runs.append(x)
    return runs


def encode(mask: np.ndarray) -> dict[str, Any]:
    """Encode a 2-D boolean mask as a JSON-ready COCO RLE dict."""
    height, width = mask.shape
    return {"size": [int(height), int(width)], "counts": counts_to_string(run_lengths(mask))}


def decode(rle: dict[str, Any]) -> np.ndarray:
    """Decode a COCO RLE dict into a boolean ``(height, width)`` mask.

    Accepts both the compressed form (``counts`` is a string or bytes) and the
    uncompressed form (``counts`` is a list of run lengths).

    Raises:
        ValueError: If the runs do not add up to ``height * width``.
    """
    height, width = rle["size"]
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    runs = string_to_counts(counts) if isinstance(counts, str) else list(counts)
    if sum(runs) != height * width or any(r < 0 for r in runs):
        raise ValueError(f"RLE runs do not cover a {height}x{width} mask")
    values = np.arange(len(runs)) % 2 == 1  # runs alternate 0, 1, 0, ...
    flat = np.repeat(values, runs)
    return flat.reshape((height, width), order="F")
