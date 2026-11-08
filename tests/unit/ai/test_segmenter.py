import threading

import numpy as np
import pytest

from easy_mask.ai.segmenter import (
    BOX_BOTTOM_RIGHT,
    BOX_TOP_LEFT,
    NEGATIVE,
    POSITIVE,
    Segmenter,
    encode_prompts,
)


def rgb(height=40, width=60):
    return np.zeros((height, width, 3), dtype=np.uint8)


# ---------------------------------------------------------------------- encode_prompts


def test_points_only_keep_order_and_labels():
    coords, labels = encode_prompts([(1, 2), (3, 4)], [POSITIVE, NEGATIVE], None, 6)
    assert coords.tolist() == [[1, 2], [3, 4]]
    assert labels.tolist() == [1, 0]
    assert coords.dtype == labels.dtype == np.float32


def test_more_than_six_points_keeps_the_most_recent_six():
    points = [(i, i) for i in range(9)]
    coords, labels = encode_prompts(points, [POSITIVE] * 9, None, 6)
    assert coords[:, 0].tolist() == [3, 4, 5, 6, 7, 8]
    assert len(labels) == 6


def test_box_comes_first_as_normalized_corners():
    coords, labels = encode_prompts([(5, 5)], [POSITIVE], (30, 40, 10, 20), 6)
    assert coords.tolist() == [[10, 20], [30, 40], [5, 5]]
    assert labels.tolist() == [BOX_TOP_LEFT, BOX_BOTTOM_RIGHT, POSITIVE]


def test_box_takes_two_of_the_six_slots():
    points = [(i, i) for i in range(6)]
    coords, _ = encode_prompts(points, [NEGATIVE] * 6, (0, 0, 9, 9), 6)
    assert len(coords) == 6
    assert coords[2:, 0].tolist() == [2, 3, 4, 5]


def test_box_alone_and_no_prompts():
    coords, _ = encode_prompts([], [], (0, 0, 1, 1), 6)
    assert coords.shape == (2, 2)
    coords, labels = encode_prompts([], [], None, 6)
    assert coords.shape == (0, 2) and labels.shape == (0,)


def test_points_and_labels_must_match():
    with pytest.raises(ValueError, match="2 points but 1 labels"):
        encode_prompts([(0, 0), (1, 1)], [POSITIVE], None, 6)


# ---------------------------------------------------------------------- Segmenter


def test_predict_returns_best_score_first(fake_segmenter):
    fake_segmenter.set_image(rgb())
    predictions = fake_segmenter.predict([(30, 20), (35, 20)], [POSITIVE, POSITIVE])
    assert [score for _, score in predictions] == [0.9, 0.2, 0.1]
    best = predictions[0][0]
    assert best.shape == (40, 60) and best[20, 30] and not best[0, 0]


class PartFirst(Segmenter):
    """Scores a small part above the whole object, like EfficientSAM on one click."""

    def _encode(self, rgb):
        return None

    def _decode(self, embedding, size, coords, labels):
        whole = np.ones(size, dtype=bool)
        part = np.zeros(size, dtype=bool)
        part[:2, :2] = True
        return [(part, 0.9), (whole, 0.5)]


def test_a_single_click_prefers_the_whole_object():
    model = PartFirst()
    model.set_image(rgb())
    assert model.predict([(1, 1)], [POSITIVE])[0][0].all()
    assert not model.predict([(1, 1), (2, 2)], [POSITIVE, POSITIVE])[0][0].all()
    assert not model.predict([], [], box=(0, 0, 5, 5))[0][0].all()


def test_negative_point_cuts_the_mask(fake_segmenter):
    fake_segmenter.set_image(rgb())
    best, _ = fake_segmenter.predict([(30, 20), (36, 20)], [POSITIVE, NEGATIVE])[0]
    assert best[20, 25] and not best[20, 36]


def test_predict_passes_at_most_six_points_to_the_decoder(fake_segmenter):
    fake_segmenter.set_image(rgb())
    fake_segmenter.predict([(i, 1) for i in range(10)], [POSITIVE] * 10, box=(0, 0, 50, 30))
    coords, labels = fake_segmenter.calls[-1]
    assert len(coords) == 6 and labels[:2].tolist() == [2, 3]
    assert coords[2:, 0].tolist() == [6, 7, 8, 9]


def test_no_prompt_gives_nothing(fake_segmenter):
    fake_segmenter.set_image(rgb())
    assert fake_segmenter.predict([], []) == []
    assert fake_segmenter.calls == []


def test_predict_before_set_image_is_an_error(fake_segmenter):
    with pytest.raises(RuntimeError, match="set_image"):
        fake_segmenter.predict([(1, 1)], [POSITIVE])


def test_embeddings_are_cached_per_key(fake_segmenter):
    fake_segmenter.set_image(rgb(40, 60), "a.png")
    fake_segmenter.set_image(rgb(20, 30), "b.png")
    fake_segmenter.set_image(rgb(40, 60), "a.png")
    assert fake_segmenter.encoded == [(40, 60, 3), (20, 30, 3)]
    assert fake_segmenter.is_cached("a.png") and not fake_segmenter.is_cached("c.png")
    best, _ = fake_segmenter.predict([(1, 1)], [POSITIVE])[0]
    assert best.shape == (40, 60)  # the cached image became current again


def test_no_key_means_no_caching(fake_segmenter):
    fake_segmenter.set_image(rgb())
    fake_segmenter.set_image(rgb())
    assert len(fake_segmenter.encoded) == 2


def test_cache_drops_least_recently_used(fake_segmenter):
    small = type(fake_segmenter)(cache_size=2)
    small.set_image(rgb(), "a")
    small.set_image(rgb(), "b")
    small.set_image(rgb(), "a")  # touch a: b is now the oldest
    small.set_image(rgb(), "c")
    assert small.is_cached("a") and small.is_cached("c") and not small.is_cached("b")


def test_set_image_can_run_on_another_thread(fake_segmenter):
    worker = threading.Thread(target=fake_segmenter.set_image, args=(rgb(), "a"))
    worker.start()
    worker.join()
    assert fake_segmenter.is_cached("a")
    assert fake_segmenter.predict([(5, 5)], [POSITIVE])


def test_precompute_caches_without_changing_current_image(fake_segmenter):
    import numpy as np

    small = np.zeros((10, 12, 3), np.uint8)
    big = np.zeros((20, 30, 3), np.uint8)
    fake_segmenter.set_image(small, "small")
    fake_segmenter.precompute(big, "big")
    assert fake_segmenter.is_cached("big")
    (mask, _), *_ = fake_segmenter.predict([(5, 5)], [1])
    assert mask.shape == (10, 12)  # still segmenting the current image
    fake_segmenter.precompute(big, "big")  # already cached: not encoded again
    assert fake_segmenter.encoded.count((20, 30, 3)) == 1
