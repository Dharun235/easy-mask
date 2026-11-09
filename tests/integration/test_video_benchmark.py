"""Video benchmark sanity checks and the real propagation model on DAVIS 2017.

The metric tests always run. The rest need DAVIS 2017 in ``data/benchmarks/DAVIS2017``
and, for the model, the ``video`` extra plus downloaded weights; they skip otherwise.
"""

import importlib.util

import numpy as np
import pytest
from benchmarks import video

from easy_mask.propagation import backend


def disc(cx, r):
    yy, xx = np.mgrid[:100, :100]
    return (xx - cx) ** 2 + (yy - 50) ** 2 <= r * r


def test_f_measure_perfect_close_and_far():
    assert video.f_measure(disc(50, 20), disc(50, 20)) == 1.0
    assert video.f_measure(disc(50, 20), disc(50, 21)) == 1.0  # within tolerance
    assert video.f_measure(disc(20, 10), disc(80, 10)) == 0.0
    empty = np.zeros((100, 100), bool)
    assert video.f_measure(empty, empty) == 1.0 and video.f_measure(disc(50, 5), empty) == 0.0


def test_score_averages_per_object_then_over_objects():
    perfect = [(disc(50, 20), disc(50, 20))]
    wrong = [(disc(20, 10), disc(80, 10)), (disc(20, 10), disc(80, 10))]
    assert video.score({1: perfect, 2: wrong}) == (0.5, 0.5)


davis = pytest.mark.skipif(not video.available(), reason="DAVIS 2017 not downloaded")


@pytest.fixture(scope="module")
def blackswan():
    (seq,) = video.load_sequences(2, max_frames=10)[1:]
    assert seq.name == "blackswan"
    return seq


@davis
@pytest.mark.benchmark
def test_interpolation_baseline_on_easy_sequence(blackswan):
    j, f = video.run_interpolation(blackswan)
    assert j > 0.85 and f > 0.9


def model_ready(name: str) -> bool:
    spec = backend.MODELS[name]
    return (
        importlib.util.find_spec("torch") is not None
        and (backend.CHECKPOINT_DIR / spec.filename).exists()
    )


@davis
@pytest.mark.slow
@pytest.mark.benchmark
@pytest.mark.skipif(not model_ready(backend.DEFAULT_MODEL), reason="video extra/weights missing")
def test_real_model_tracks_blackswan(blackswan):
    propagator = backend.Sam2Propagator(device="cpu")
    j, f, seconds = video.run_propagation(blackswan, propagator)
    assert j > 0.85 and f > 0.85
    assert seconds < 30
