from easy_mask.core.history import History


def test_undo_redo_cycle_on_one_frame(square):
    history = History()
    history.record(0, [], [square])
    moved = square.copy()
    moved.translate(1, 1)
    history.record(0, [square], [moved])

    assert history.undo() == {0: [square]}
    assert history.undo() == {0: []}
    assert history.undo() is None
    assert history.redo() == {0: [square]}


def test_steps_on_different_frames_undo_in_order(square):
    history = History()
    history.record(0, None, [square])  # frame 0 had no annotation file
    history.record(3, [square], [])
    assert history.undo() == {3: [square]}
    assert history.undo() == {0: None}
    assert history.redo() == {0: [square]}


def test_group_is_one_step_and_keeps_the_first_before(square):
    history = History()
    history.begin_group()
    history.record(1, None, [square])
    moved = square.copy()
    moved.translate(5, 5)
    history.record(1, [square], [moved])
    history.record(2, [], [square])
    assert history.end_group() == 2
    assert history.undo() == {1: None, 2: []}
    assert not history.can_undo
    assert history.redo() == {1: [moved], 2: [square]}


def test_empty_group_adds_no_step():
    history = History()
    history.begin_group()
    assert history.end_group() == 0 and not history.can_undo


def test_record_discards_redo_and_clear_forgets(square):
    history = History()
    history.record(0, [], [square])
    history.undo()
    history.record(0, [], [])
    assert not history.can_redo
    history.clear()
    assert not history.can_undo


def test_snapshots_are_copies(square):
    history = History()
    after = [square]
    history.record(0, [], after)
    square.translate(100, 100)  # mutate after recording
    history.undo()
    (kept,) = history.redo()[0]
    assert kept.points != square.points


def test_limit_drops_oldest(square):
    history = History(limit=2)
    for frame in range(3):
        history.record(frame, [], [square])
    assert history.undo() == {2: []} and history.undo() == {1: []}
    assert history.undo() is None
