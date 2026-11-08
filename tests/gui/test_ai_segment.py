"""AI Segment tool driven through the real window, with a fake segmenter backend."""

import threading

import numpy as np
import pytest
from PySide6.QtCore import Qt

from easy_mask.ai.efficient_sam import INSTALL_HINT, MissingDependencyError
from easy_mask.core import masks
from easy_mask.core.shapes import ShapeType
from easy_mask.gui.tools.ai_segment import pixmap_to_rgb
from easy_mask.gui.tools.ai_worker import EncodeJob, LazySegmenter

R = 10  # FakeSegmenter disc radius


@pytest.fixture
def ai(window, fake_segmenter, qtbot):
    """The window in AI Segment mode with the fake backend, image already encoded."""
    window.load_segmenter = lambda progress: fake_segmenter
    window.set_tool("ai")
    tool = window.pixel_tools["ai"]
    qtbot.waitUntil(lambda: tool.ready)
    yield tool
    tool.wait()


def full(window, index=0):
    return masks.to_full(window.canvas.shapes[index], 80, 100)


def test_click_previews_and_enter_accepts_a_new_labelled_mask(window, ui, ai):
    ui.click(50, 40)
    assert ai.mask is not None and ai.mask[40, 50] and not ai.mask[40, 50 + R + 2]
    assert window.canvas.shapes == []  # only a preview so far

    ui.key(Qt.Key.Key_Return)

    (shape,) = window.canvas.shapes
    assert (shape.label, shape.shape_type) == ("cat", ShapeType.MASK)
    assert full(window)[40, 50] and window.dirty
    assert ai.mask is None and ai.points == []
    assert window.canvas.selection == []  # next object starts a new mask


def test_right_click_and_alt_click_are_background_points(window, ui, ai):
    ui.click(50, 40)
    ui.click(58, 40, Qt.MouseButton.RightButton)
    assert ai.labels == [1, 0] and not ai.mask[40, 58] and ai.mask[40, 45]
    ui.qtbot.mouseClick(
        window.canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.AltModifier, ui.screen(42, 40)
    )
    assert ai.labels == [1, 0, 0] and not ai.mask[40, 42]


def test_drag_draws_a_box_prompt(window, ui, ai, fake_segmenter):
    ui.drag((10, 10), (40, 30))
    assert ai.box == pytest.approx((10, 10, 40, 30), abs=1) and ai.points == []
    assert ai.mask[20, 25] and not ai.mask[50, 50]
    ui.drag((60, 40), (90, 70))  # a second box replaces the first
    assert ai.box == pytest.approx((60, 40, 90, 70), abs=1)
    assert fake_segmenter.calls[-1][1].tolist() == [2, 3]


def test_right_drag_is_a_negative_point_not_a_box(window, ui, ai):
    ui.drag((10, 10), (40, 30), Qt.MouseButton.RightButton)
    assert ai.box is None and ai.labels == [0]
    assert ai.points[0] == pytest.approx((10, 10), abs=1)


def test_backspace_removes_last_prompt_then_falls_through(window, ui, ai):
    ui.drag((10, 10), (40, 30))
    ui.click(80, 60)
    ui.key(Qt.Key.Key_Backspace)
    assert ai.points == [] and ai.box is not None
    ui.key(Qt.Key.Key_Backspace)
    assert ai.box is None and ai.mask is None


def test_escape_clears_prompts_then_deselects(window, ui, ai):
    ui.click(50, 40)
    ui.click(20, 20)
    ui.key(Qt.Key.Key_Escape)
    assert ai.points == [] and ai.mask is None
    window.canvas.select(None)
    ui.key(Qt.Key.Key_Escape)  # nothing to clear: the canvas handles it, no error


def test_double_click_accepts_without_the_extra_point(window, ui, ai, fake_segmenter):
    ui.click(30, 40)
    ui.click(70, 40)  # first click of the double-click
    ui.double_click(70, 40)
    assert len(window.canvas.shapes) == 1
    assert full(window)[40, 30] and not full(window)[40, 70]


def test_double_click_alone_segments_and_accepts(window, ui, ai):
    ui.click(50, 40)
    ui.double_click(50, 40)
    assert len(window.canvas.shapes) == 1 and full(window)[40, 50]


def test_shift_click_adds_to_the_selected_mask(window, ui, ai):
    ui.click(20, 20)
    ui.key(Qt.Key.Key_Return)
    window.canvas.select(0)
    ui.qtbot.mouseClick(
        window.canvas,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.ShiftModifier,
        ui.screen(70, 60),
    )
    ui.key(Qt.Key.Key_Enter)
    assert len(window.canvas.shapes) == 1
    assert full(window)[20, 20] and full(window)[60, 70]
    assert window.canvas.selected == 0


def test_plain_click_makes_a_new_object_even_with_a_selection(window, ui, ai):
    """Regression: a mask left selected (e.g. after a brush stroke) swallowed new objects."""
    ui.click(20, 20)
    ui.key(Qt.Key.Key_Return)
    window.canvas.select(0)
    ui.click(70, 60)
    ui.key(Qt.Key.Key_Enter)
    assert len(window.canvas.shapes) == 2


def test_enter_without_a_mask_does_nothing(window, ui, ai):
    ui.key(Qt.Key.Key_Return)
    assert window.canvas.shapes == [] and not window.dirty


def test_cancelled_label_adds_nothing(window, ui, ai):
    window.ask_label = lambda known, initial, parent: None
    ui.click(50, 40)
    ui.key(Qt.Key.Key_Return)
    assert window.canvas.shapes == []


def test_undo_removes_accepted_mask(window, ui, ai):
    ui.click(50, 40)
    ui.key(Qt.Key.Key_Return)
    window.undo()
    assert window.canvas.shapes == []


def test_switching_tool_drops_prompts(window, ui, ai):
    ui.click(50, 40)
    window.set_tool(None)
    assert ai.points == [] and ai.mask is None


def test_ui_stays_responsive_while_encoding(make_window, image_dir, driver, qtbot, fake_segmenter):
    fake_segmenter.gate = threading.Event()
    win = make_window()
    win.load_segmenter = lambda progress: fake_segmenter
    win.open_path(image_dir)
    messages = []
    tool = win.pixel_tools["ai"]
    tool.status.connect(messages.append)
    win.set_tool("ai")
    ui = driver(win)

    ui.click(50, 40)  # the event loop is free: clicks are taken while the model works
    assert not tool.ready and len(tool.points) == 1 and tool.mask is None

    fake_segmenter.gate.set()
    qtbot.waitUntil(lambda: tool.ready)
    assert tool.mask is not None and tool.mask[40, 50]  # the waiting click is applied
    assert any("encoding" in m for m in messages) and "click the object" in messages[-1]


def test_images_are_encoded_once_and_reused(window, ui, ai, fake_segmenter, qtbot):
    window.next_image()
    qtbot.waitUntil(lambda: ai.ready)
    ai.wait()
    encoded = len(fake_segmenter.encoded)
    window.prev_image()
    qtbot.waitUntil(lambda: ai.ready)
    ai.wait()
    assert len(fake_segmenter.encoded) == encoded  # going back reuses the cached embedding


def test_next_images_are_prefetched_once_model_is_loaded(window, ai, fake_segmenter, qtbot):
    ai.wait()
    for path in window.image_paths:  # img1 shown; img2 and img10 prepared in the background
        assert fake_segmenter.is_cached(str(path))
    encoded = len(fake_segmenter.encoded)
    window.next_image()
    assert ai.ready or qtbot.waitUntil(lambda: ai.ready, timeout=1000) is None
    ai.wait()
    assert len(fake_segmenter.encoded) == encoded  # instant: nothing new to encode


def test_no_prefetch_or_download_before_ai_is_used(make_window, image_dir, fake_segmenter):
    loads = []
    win = make_window()
    win.load_segmenter = lambda progress: loads.append(1) or fake_segmenter
    win.open_path(image_dir)
    win.next_image()
    win.pixel_tools["ai"].wait()
    assert loads == [] and fake_segmenter.encoded == []


def test_prefetch_survives_unreadable_files(window, ai, fake_segmenter, image_dir):
    broken = image_dir / "broken.png"
    broken.write_bytes(b"not an image")
    ai.prefetch([str(broken)])
    ai.wait()
    assert not fake_segmenter.is_cached(str(broken))


def test_download_progress_is_shown(make_window, image_dir, qtbot, fake_segmenter):
    win = make_window()

    def load(progress):
        progress("encoder.onnx", 50, 100)
        return fake_segmenter

    win.load_segmenter = load
    win.open_path(image_dir)
    messages = []
    win.pixel_tools["ai"].status.connect(messages.append)
    win.set_tool("ai")
    qtbot.waitUntil(lambda: win.pixel_tools["ai"].ready)
    assert "AI Segment: loading model..." in messages
    assert "AI Segment: downloading encoder.onnx (50%)" in messages


def test_missing_backend_shows_install_hint_and_leaves_the_tool(
    make_window, image_dir, qtbot, dialogs
):
    dialogs.allow_warnings = True
    win = make_window()

    def load(progress):
        raise MissingDependencyError(INSTALL_HINT)

    win.load_segmenter = load
    win.open_path(image_dir)
    win.set_tool("ai")
    qtbot.waitUntil(lambda: bool(dialogs.warnings))
    assert "uv sync" in dialogs.warnings[0]
    assert win.canvas.tool is None and win.tool_actions[None].isChecked()


def test_tool_without_image_waits(make_window, fake_segmenter):
    win = make_window()
    win.load_segmenter = lambda progress: fake_segmenter
    win.set_tool("ai")
    tool = win.pixel_tools["ai"]
    tool.wait()
    assert not tool.ready and fake_segmenter.encoded == []


def test_pixmap_to_rgb_matches_the_file(window):
    rgb = pixmap_to_rgb(window.canvas.pixmap)
    assert rgb.shape == (80, 100, 3) and rgb.dtype == np.uint8
    assert (rgb == 120).all()


def run_job(job):
    """Run an encode job on this thread (coverage does not trace Qt pool threads)."""
    seen = {"progress": [], "done": [], "failed": []}
    for name, record in seen.items():
        getattr(job.signals, name).connect(record.append)
    job.run()
    return seen


def test_encode_job_loads_once_reports_and_encodes(qtbot, fake_segmenter):
    loads = []

    def load(progress):
        loads.append(1)
        progress("enc.onnx", 1, 4)
        return fake_segmenter

    lazy = LazySegmenter(load)
    rgb = np.zeros((8, 8, 3), np.uint8)
    first = run_job(EncodeJob(lazy, rgb, "a", 1))
    again = run_job(EncodeJob(lazy, rgb, "a", 2))

    assert loads == [1] and lazy.segmenter is fake_segmenter
    assert first["progress"] == [
        "AI Segment: downloading enc.onnx (25%)",
        "AI Segment: encoding image...",
    ]
    assert first["done"] == [1] and again == {"progress": [], "done": [2], "failed": []}
    assert fake_segmenter.encoded == [(8, 8, 3)]


def test_encode_job_reports_failure(qtbot):
    def load(progress):
        raise MissingDependencyError(INSTALL_HINT)

    seen = run_job(EncodeJob(LazySegmenter(load), np.zeros((8, 8, 3), np.uint8), "a", 1))
    assert seen["done"] == [] and "uv sync" in seen["failed"][0]


def test_auto_cleanup_tidies_ai_result_and_can_be_turned_off(window, ui, ai, fake_segmenter):
    import numpy as np

    window.set_active_label("cat")
    speckled = np.zeros((80, 100), bool)
    speckled[20:60, 20:60] = True
    speckled[30, 30] = False  # pinhole
    speckled[5, 90] = True  # speck
    window._apply_smart(speckled.copy(), None)
    from easy_mask.core import masks as m

    stored = m.to_full(window.canvas.shapes[-1], 80, 100)
    assert stored[30, 30] and not stored[5, 90]

    window.auto_cleanup = False
    window._apply_smart(speckled.copy(), None)
    raw = m.to_full(window.canvas.shapes[-1], 80, 100)
    assert not raw[30, 30] and raw[5, 90]


def test_brush_is_never_cleaned_up(window, ui):
    window.set_active_label("cat")
    window.set_tool("brush")
    window.pixel_tools["brush"].radius = 1
    ui.drag((10, 10), (11, 10))  # a tiny stroke, smaller than the clean-up size
    assert len(window.canvas.shapes) == 1
