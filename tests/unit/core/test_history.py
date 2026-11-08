from easy_mask.core.history import History


def test_undo_redo_cycle(square):
    history = History()
    history.reset([])
    history.push([square])
    moved = square.copy()
    moved.translate(1, 1)
    history.push([moved])

    assert history.undo() == [square]
    assert history.undo() == []
    assert history.undo() is None
    assert history.redo() == [square]


def test_push_discards_redo(square):
    history = History()
    history.push([square])
    history.undo()
    history.push([])
    assert not history.can_redo


def test_snapshots_are_copies(square):
    history = History()
    shapes = [square]
    history.push(shapes)
    square.translate(100, 100)  # mutate after pushing
    history.push([])
    assert history.undo()[0].points[0] == (10, 10)


def test_limit_drops_oldest(square):
    history = History(limit=3)
    for _ in range(10):
        history.push([square])
    undos = 0
    while history.undo() is not None:
        undos += 1
    assert undos == 2


def test_version_changes_on_every_state_change():
    history = History()
    seen = {history.version}
    history.push([])
    seen.add(history.version)
    history.undo()
    seen.add(history.version)
    history.redo()
    seen.add(history.version)
    history.reset([])
    seen.add(history.version)
    assert len(seen) == 5
