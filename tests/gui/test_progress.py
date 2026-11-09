"""Progress feedback: the status-bar panel and the tasks that report to it."""

from easy_mask.core.shapes import Shape, ShapeType
from easy_mask.gui.export_worker import ExportWorker
from easy_mask.gui.main_window import Settings
from easy_mask.gui.progress import ProgressPanel
from easy_mask.io import export


def test_panel_hidden_until_a_task_runs(qtbot):
    panel = ProgressPanel()
    qtbot.addWidget(panel)
    assert panel.isHidden()
    panel.begin("job", "Working")
    assert not panel.isHidden() and panel.bar.maximum() == 0  # busy indicator
    assert panel.label.text() == "Working..."
    panel.end("job")
    assert panel.isHidden()


def test_panel_shows_percentage_and_newest_task(qtbot):
    panel = ProgressPanel()
    qtbot.addWidget(panel)
    panel.begin("a", "Exporting", total=4)
    panel.report("a", 1)
    assert panel.label.text() == "Exporting  25%" and panel.bar.value() == 1
    panel.begin("b", "Loading AI model")
    assert panel.label.text() == "Loading AI model...  (+1 more)"
    panel.end("b")
    assert panel.label.text() == "Exporting  25%"
    panel.report("missing", 3)  # unknown tasks are ignored
    assert panel.is_running("a") and not panel.is_running("missing")


def test_cancel_button_only_for_cancellable_tasks(qtbot):
    panel = ProgressPanel()
    qtbot.addWidget(panel)
    cancelled = []
    panel.begin("plain", "Waiting")
    assert panel.cancel_button.isHidden()
    panel.begin("stoppable", "Propagating", total=10, cancel=lambda: cancelled.append(1))
    assert not panel.cancel_button.isHidden()
    panel.cancel_button.click()
    assert cancelled == [1] and not panel.is_running("stoppable")


def test_export_reports_progress_and_finishes(
    export_as, make_window, image_dir, tmp_path, driver, wait_export, monkeypatch
):

    win = make_window(Settings())
    win.open_path(image_dir)
    win.set_active_label("cat")
    win.set_tool(ShapeType.RECTANGLE)
    driver(win).drag((10, 10), (30, 30))
    win.save()
    seen = []
    original = win.progress.report
    monkeypatch.setattr(
        win.progress,
        "report",
        lambda key, done, *a, **k: seen.append((key, done)) or original(key, done, *a, **k),
    )
    export_as(win, "coco", tmp_path / "out")
    assert win.progress.is_running("export")
    wait_export(win)
    assert ("export", 1) in seen and not win.progress.is_running("export")
    assert "Exported 1 images" in win.statusBar().currentMessage()


def test_export_worker_cancel_stops_before_the_end(qtbot, image_dir, tmp_path, square):
    from easy_mask.core import annotation as ann_io
    from easy_mask.core.annotation import Annotation

    for name in ("img1", "img2", "img10"):
        ann_io.save(Annotation(f"{name}.png", 100, 80, [square]), image_dir / f"{name}.json")
    worker = ExportWorker(export.collect(image_dir), "yolo", tmp_path / "out", ["cat"])
    worker.cancel()  # as if Cancel was clicked: checked after each image
    finished = []
    worker.exported.connect(finished.append)
    worker.start()
    worker.wait()
    qtbot.wait(10)
    assert finished == []  # cancelled after the first image
    labels = list((tmp_path / "out" / "labels").glob("*.txt"))
    assert len(labels) == 1


def test_export_failure_is_reported(qtbot, image_dir, tmp_path):
    worker = ExportWorker([], "no-such-format", tmp_path, [])
    messages = []
    worker.failed.connect(messages.append)
    worker.start()
    worker.wait()
    qtbot.wait(10)
    assert messages and messages[0].startswith("Export failed")


def test_ai_preparation_shows_in_progress_panel(make_window, image_dir, qtbot, fake_segmenter):
    win = make_window(Settings())

    def load(progress):
        progress("encoder.onnx", 25, 100)
        return fake_segmenter

    win.load_segmenter = load
    seen = []
    win.ai_tool.busy.connect(lambda text, done, total: seen.append((text, done, total)))
    win.open_path(image_dir)
    win.set_tool("ai")
    qtbot.waitUntil(lambda: win.ai_tool.ready)
    assert seen[0] == ("Loading AI model", 0, 0)
    assert ("Downloading AI model (encoder.onnx)", 25, 100) in seen
    assert not win.progress.is_running("ai")


def test_propagation_shows_frames_done(qtbot, window, fake_propagator):
    window.video.make_propagator = lambda: fake_propagator
    window.ask_number = lambda *args: 2
    window.canvas.set_shapes([Shape("car", ShapeType.RECTANGLE, [(10, 10), (19, 19)])])
    window.video.propagate(+1)
    assert window.progress.is_running("propagate")
    assert window.progress.bar.maximum() == 2
    qtbot.waitUntil(lambda: not window.video.running, timeout=5000)
    assert not window.progress.is_running("propagate")
