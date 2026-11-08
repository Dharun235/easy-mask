from pathlib import Path

import numpy as np

from easy_mask.core import masks
from easy_mask.core.shapes import Shape, ShapeType
from easy_mask.core.video import PROPAGATED_FLAG
from easy_mask.propagation.tracker import track

FRAMES = [Path(f"f{i}.jpg") for i in range(4)]


def rect(label, x, group_id=None, **flags):
    return Shape(label, ShapeType.RECTANGLE, [(x, 2), (x + 4, 6)], group_id=group_id, flags=flags)


def run(propagator, frame_shapes):
    return list(track(propagator, FRAMES, frame_shapes, 20, 40))


def test_first_frame_objects_are_tracked_and_flagged(fake_propagator):
    results = run(fake_propagator, [[rect("car", 0, 1), rect("car", 20, 2)], [], [], []])
    assert [position for position, _ in results] == [1, 2, 3]
    _, shapes = results[1]
    assert [(s.label, s.group_id, s.shape_type) for s in shapes] == [
        ("car", 1, ShapeType.MASK),
        ("car", 2, ShapeType.MASK),
    ]
    assert all(s.flags == {PROPAGATED_FLAG: True} for s in shapes)
    assert shapes[0].points == [(4, 2), (8, 6)]  # the fake shifts 2 px per frame


def test_prompts_are_rasterized_and_include_later_manual_shapes(fake_propagator):
    later_manual = rect("car", 30)
    later_auto = rect("car", 10, propagated=True)
    other_object = rect("bike", 0)
    run(fake_propagator, [[rect("car", 0)], [later_auto], [later_manual, other_object], []])
    ((_, prompts),) = fake_propagator.calls
    assert sorted(prompts) == [0, 2]
    assert set(prompts[2]) == {1}  # bike was not on the first frame
    assert prompts[0][1].sum() == 25 and prompts[2][1][2:7, 30:35].all()


def test_manual_prompt_restarts_tracking_from_that_frame(fake_propagator):
    results = run(fake_propagator, [[rect("car", 0)], [], [rect("car", 30)], []])
    assert results[2][1][0].points == [(32, 2), (36, 6)]


def test_nothing_to_track_yields_nothing(fake_propagator):
    line = Shape("edge", ShapeType.LINE, [(0, 0), (5, 5)])
    assert run(fake_propagator, [[line], [], [], []]) == []
    assert fake_propagator.calls == []


def test_empty_model_output_gives_no_shape():
    class Blind:
        def propagate(self, frames, prompts):
            yield {1: {1: np.zeros((20, 40), bool)}}

    assert run(Blind(), [[rect("car", 0)], [], [], []]) == [(1, [])]


def test_mask_prompts_keep_their_pixels(fake_propagator):
    full = np.zeros((20, 40), bool)
    full[5, 5:9] = True
    run(fake_propagator, [[masks.to_shape("hair", full)], [], [], []])
    assert (fake_propagator.calls[0][1][0][1] == full).all()


def test_copy_propagator_repeats_masks_and_follows_corrections():
    from pathlib import Path

    from easy_mask.propagation.tracker import CopyPropagator

    first = np.zeros((4, 4), bool)
    first[0, 0] = True
    fixed = np.zeros((4, 4), bool)
    fixed[3, 3] = True
    frames = [Path(f"{i}.jpg") for i in range(4)]
    results = list(CopyPropagator().propagate(frames, {0: {1: first}, 2: {1: fixed}}))
    assert [list(r) for r in results] == [[1], [2], [3]]
    assert np.array_equal(results[0][1][1], first)
    assert np.array_equal(results[1][2][1], fixed)  # the hand-fixed mask wins from frame 2
    assert np.array_equal(results[2][3][1], fixed)
    results[0][1][1][0, 0] = False
    assert first[0, 0]  # yields copies, never the caller's arrays
