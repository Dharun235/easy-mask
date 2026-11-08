# Testing guide

How easy-mask is tested, which kind of test to write when, and how to run them.

## TL;DR

```bash
uv run pytest                        # everything that can run here (~10 s)
uv run pytest -m unit                # one layer: unit | integration | gui | property | architecture | smoke
uv run pytest -m "not slow and not gui"   # fastest feedback loop while coding
uv run pytest --cov                  # with coverage; fails under 90%
uv run pytest -m benchmark           # real benchmark data (needs data/benchmarks, see below)
uv run pytest tests/gui/test_canvas.py -k zoom -x   # one file, matching names, stop at first failure
```

## The test pyramid

Many fast, isolated tests at the bottom; few slow, broad ones at the top. Each layer
catches bugs the others miss.

```
            smoke            does the real app start in a fresh process?
          gui (e2e)          real mouse/key events on a headless window
        integration          several modules + real files: CLI, export pipeline, labelme files
   unit · property · architecture    one function at a time; invariants; project rules
```

| Type | Folder | What it checks | Speed | Write one when... |
| --- | --- | --- | --- | --- |
| **Unit** | `tests/unit/` | One function/class in isolation (`core/`, `io/`) | ms | you add or change any core/io function |
| **Property** | `tests/property/` | Invariants over thousands of generated inputs (Hypothesis) | ~1 s | a rule must hold for *all* inputs: round-trips, geometry identities, undo/redo |
| **Architecture** | `tests/architecture/` | Layering: `core`/`io` never import Qt or `gui`; export never loads Qt | ms | you add a new layer or project-wide rule |
| **Integration** | `tests/integration/` | Modules together with real files: CLI export, all formats agree, labelme compatibility, golden outputs | 10-100 ms | a feature crosses module boundaries or touches file formats |
| **GUI / end-to-end** | `tests/gui/` | The real window driven by `qtbot` clicks, drags and keys | 10-100 ms | you change user-visible behaviour |
| **Smoke** | `tests/smoke/` | `python -m easy_mask` in a subprocess: version, help, GUI stays up | 1-3 s | you change packaging, entry points or startup |
| **Benchmark data** | `tests/integration/test_benchmark_data.py` | Pipeline on GrabCut / Berkeley / DAVIS | seconds | you change image loading or exporters |

Every test is auto-marked with its folder name (see `tests/conftest.py`), so `-m <type>`
always works. `slow` and `benchmark` are added explicitly.

## Special techniques used here

- **Regression tests.** Every bug fix starts with a test that fails before the fix (e.g.
  `test_double_click_relabels_unselected_shape`,
  `test_gui_export_keeps_labels_added_in_session`).
- **Golden files.** `tests/integration/test_export_golden.py` compares exports with
  checked-in expected output in `tests/data/golden/`. If you *intend* to change an
  output format, regenerate and review the diff in your PR:
  `UPDATE_GOLDEN=1 uv run pytest tests/integration/test_export_golden.py`.
- **Cross-checking formats.** `test_pipeline.py` rasterizes COCO and YOLO polygons and
  compares them with the mask PNGs: a bug in one exporter shows up as disagreement.
- **Reference implementation.** `core/rle.py` is checked byte-for-byte against
  pycocotools (a dev-only dependency) on known vectors and Hypothesis-generated masks.
- **Model-based testing.** `test_history_matches_simple_model` runs random push/undo/redo
  sequences against an obviously-correct list model.
- **Headless GUI.** `QT_QPA_PLATFORM=offscreen` is set in `tests/conftest.py`. Modal dialogs
  would block forever, so `tests/gui/conftest.py` replaces message boxes with the scriptable
  `dialogs` fixture and the label dialog with `window.ask_label`. Use the `ui` driver to
  click in *image* coordinates: `ui.click(10, 20)`, `ui.drag((0, 0), (30, 30))`.
- **Fake model.** AI Segment logic is tested with `FakeSegmenter` (`fake_segmenter`
  fixture, `tests/conftest.py`): deterministic discs around clicks, no onnxruntime needed.
  Weight downloads are tested against an HTTP server on `127.0.0.1` (checksum mismatch,
  truncated download, offline). The real EfficientSAM model runs in
  `tests/integration/test_efficient_sam_model.py` only when onnxruntime is installed and
  the weights are already cached; it never downloads.
- **Fake model.** Video propagation is tested with the `fake_propagator` fixture
  (`tests/conftest.py`), which shifts each prompt mask 2 px per frame: deterministic, fast,
  no PyTorch. The real model runs only in `tests/integration/test_video_benchmark.py`,
  which skips unless the `video` extra, its weights and DAVIS 2017 are present.

## Writing good tests

- **One behaviour per test**, named for it: `test_zero_size_rectangle_is_ignored`.
- **Arrange, act, assert.** Set up, do one thing, check the result. Blank lines between.
- **Test behaviour, not implementation.** Assert on outputs and visible state, not on
  private attributes, unless there is no public way.
- **Deterministic.** No network, no sleeps (use `qtbot.waitSignal`), no dependence on test
  order, temp files only in `tmp_path`. Floats from trig: compare with `pytest.approx`.
- **Fast.** Keep unit tests in milliseconds; mark anything over ~1 s with `@pytest.mark.slow`.
- **Shared setup goes in fixtures** (`conftest.py`), not copy-paste. Use
  `@pytest.mark.parametrize` for the same check over many inputs.
- **GUI tests: mimic the OS.** A real double-click is press, release, then a double-click
  event: `ui.click(x, y)` then `ui.double_click(x, y)`.

## Coverage

`uv run pytest --cov` measures line and branch coverage of `src/easy_mask` and fails below
90% (configured in `pyproject.toml`). Coverage shows what is *not* tested; it does not
prove what is tested is checked well. Missing lines are listed so you can see the gaps.

## Benchmark data

The benchmark tests need GrabCut, Berkeley and DAVIS in `data/benchmarks/` (gitignored;
~60 MB), and DAVIS 2017 (`DAVIS2017/DAVIS`, 833 MB zip) for the video benchmark. They are skipped automatically when the data is missing, e.g. in CI. Dataset
sources and the metrics we will report are in [benchmarks.md](benchmarks.md).

## Continuous integration

`.github/workflows/ci.yml` runs on every push to `main` and every PR:

1. **lint:** `ruff check` and `ruff format --check`.
2. **test:** the full suite with coverage on Linux, macOS and Windows (headless Qt).

A PR is mergeable only when both are green.
