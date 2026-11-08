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


def test_subtract_from_object_cuts_only_that_object():
    import numpy as np

    from easy_mask.core import masks
    from easy_mask.core.shapes import Shape, ShapeType
    from easy_mask.core.video import edit_object

    wing = Shape("wing", ShapeType.RECTANGLE, [(0, 0), (19, 9)], flags={"propagated": True})
    leg = Shape("leg", ShapeType.RECTANGLE, [(10, 0), (19, 9)])
    cut = np.zeros((20, 30), bool)
    cut[:, 10:20] = True
    result = edit_object([wing, leg], ("wing", None), cut, 20, 30, "subtract")
    new_wing, same_leg = result
    assert same_leg is leg  # other objects untouched
    full = masks.to_full(new_wing, 20, 30)
    assert full[5, 5] and not full[5, 15]
    assert new_wing.flags == {"propagated": True}  # metadata kept


def test_subtract_from_object_noop_and_removal():
    import numpy as np

    from easy_mask.core.shapes import Shape, ShapeType
    from easy_mask.core.video import edit_object

    wing = Shape("wing", ShapeType.RECTANGLE, [(0, 0), (9, 9)])
    shapes = [wing]
    miss = np.zeros((20, 30), bool)
    miss[15:, 25:] = True
    assert edit_object(shapes, ("wing", None), miss, 20, 30, "subtract") is shapes
    assert edit_object(shapes, ("absent", None), miss, 20, 30, "subtract") is shapes
    everything = np.ones((20, 30), bool)
    assert edit_object(shapes, ("wing", None), everything, 20, 30, "subtract") == []


def test_edit_object_add_merges_and_creates_from_template():
    import numpy as np

    from easy_mask.core import masks
    from easy_mask.core.shapes import Shape, ShapeType
    from easy_mask.core.video import edit_object

    wing = Shape("wing", ShapeType.RECTANGLE, [(0, 0), (9, 9)], group_id=3)
    region = np.zeros((20, 30), bool)
    region[0:10, 20:25] = True
    (merged,) = edit_object([wing], ("wing", 3), region, 20, 30, "add")
    full = masks.to_full(merged, 20, 30)
    assert full[5, 5] and full[5, 22] and merged.group_id == 3
    assert edit_object([], ("wing", 3), region, 20, 30, "add") == []  # no template: nothing
    (created,) = edit_object([], ("wing", 3), region, 20, 30, "add", template=wing)
    assert created.label == "wing" and masks.to_full(created, 20, 30)[5, 22]
