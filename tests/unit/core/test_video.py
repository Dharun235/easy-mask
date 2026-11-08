import numpy as np

from easy_mask.core import masks
from easy_mask.core.shapes import Shape, ShapeType
from easy_mask.core.video import (
    INTERPOLATED_FLAG,
    PROPAGATED_FLAG,
    confirm,
    is_auto,
    manual_objects,
    merge_auto,
    object_key,
    object_masks,
)


def rect(label, x, group_id=None, **flags):
    return Shape(label, ShapeType.RECTANGLE, [(x, 0), (x + 4, 4)], group_id=group_id, flags=flags)


def test_object_key_is_label_and_group():
    assert object_key(rect("car", 0, group_id=3)) == ("car", 3)


def test_is_auto_for_either_flag():
    assert is_auto(rect("a", 0, propagated=True))
    assert is_auto(rect("a", 0, interpolated=True))
    assert not is_auto(rect("a", 0, propagated=False, occluded=True))


def test_confirm_drops_auto_flags_without_touching_input():
    shape = rect("a", 0, propagated=True, occluded=True)
    (confirmed,) = confirm([shape])
    assert confirmed.flags == {"occluded": True}
    assert shape.flags[PROPAGATED_FLAG]


def test_object_masks_unions_parts_and_skips_lines():
    shapes = [
        rect("dog", 0, group_id=1),
        rect("dog", 10, group_id=1),
        Shape("edge", ShapeType.LINE, [(0, 9), (9, 9)]),
    ]
    result = object_masks(shapes, 10, 20)
    assert list(result) == [("dog", 1)]
    assert result[("dog", 1)].sum() == 2 * 25


def test_manual_objects():
    shapes = [rect("a", 0), rect("b", 0, propagated=True)]
    assert manual_objects(shapes) == {("a", None)}


def test_merge_auto_replaces_old_auto_shapes():
    old = rect("car", 0, propagated=True)
    new = rect("car", 5, propagated=True)
    keep = rect("tree", 0)
    assert merge_auto([old, keep], {("car", None)}, [new]) == [keep, new]


def test_merge_auto_never_overwrites_hand_drawn_objects():
    manual = rect("car", 0)
    assert merge_auto([manual], {("car", None)}, [rect("car", 5, propagated=True)]) == [manual]


def test_merge_auto_removes_object_that_vanished():
    old = rect("car", 0, interpolated=True)
    assert merge_auto([old], {("car", None)}, []) == []


def test_merge_auto_leaves_other_auto_objects():
    other = rect("bike", 0, propagated=True)
    full = np.zeros((10, 20), bool)
    full[2:5, 2:5] = True
    new = masks.to_shape("car", full, flags={INTERPOLATED_FLAG: True})
    assert merge_auto([other], {("car", None)}, [new]) == [other, new]


def test_cut_region_cuts_every_shape_under_it():
    from easy_mask.core.shapes import Shape, ShapeType
    from easy_mask.core.video import cut_region

    wing = Shape("wing", ShapeType.RECTANGLE, [(0, 0), (19, 9)], flags={"propagated": True})
    leg = Shape("leg", ShapeType.RECTANGLE, [(10, 0), (19, 9)])
    far = Shape("bird", ShapeType.RECTANGLE, [(25, 15), (29, 19)])
    cut = np.zeros((20, 30), bool)
    cut[:, 10:20] = True
    new_wing, far_again = cut_region([wing, leg, far], cut, 20, 30)
    assert far_again is far  # untouched shapes are kept as they are
    full = masks.to_full(new_wing, 20, 30)
    assert full[5, 5] and not full[5, 15]  # the leg was cut away completely
    assert new_wing.flags == {"propagated": True}  # metadata kept


def test_cut_region_limited_to_objects_and_noop():
    from easy_mask.core.shapes import Shape, ShapeType
    from easy_mask.core.video import cut_region

    wing = Shape("wing", ShapeType.RECTANGLE, [(0, 0), (19, 9)])
    leg = Shape("leg", ShapeType.RECTANGLE, [(10, 0), (19, 9)])
    cut = np.zeros((20, 30), bool)
    cut[:, 10:20] = True
    kept_wing, new_leg = cut_region([wing, leg], cut, 20, 30, keys={("wing", None)})
    assert new_leg is leg and not masks.to_full(kept_wing, 20, 30)[5, 15]
    shapes = [wing]
    miss = np.zeros((20, 30), bool)
    miss[15:, 25:] = True
    assert cut_region(shapes, miss, 20, 30) is shapes


def test_free_group_id_separates_objects_with_the_same_label():
    from easy_mask.core.shapes import Shape, ShapeType
    from easy_mask.core.video import free_group_id

    def dog(group_id=None):
        return Shape("dog", ShapeType.RECTANGLE, [(0, 0), (5, 5)], group_id=group_id)

    assert free_group_id([], "dog") is None
    assert free_group_id([dog()], "dog") == 1
    assert free_group_id([dog(), dog(1), dog(4)], "dog") == 5
    assert free_group_id([dog(2)], "dog") is None  # the plain key is still free
    assert free_group_id([dog()], "cat") is None
