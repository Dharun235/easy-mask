# CLAUDE.md

Guidance for Claude Code (and any AI or human contributor) working in this repo.

## Project in one paragraph

**easy-mask** is a local-first, no-code desktop tool for drawing object masks on images
and videos. Users click objects for AI Segment (EfficientSAM), draw polygons and boxes,
paint and erase masks, propagate masks through video frames (EfficientTAM / SAM 2.1),
edit centroid / corner points, and export masks, cutouts, points, COCO, YOLO and DAVIS.
Everything runs on the user's machine: no cloud, no account, no telemetry. Annotations
are one labelme-compatible JSON file per image. The app is for non-programmers: keep the
UI to a few labelled buttons and right-click menus, and keep the README short.

## Commands

```bash
uv sync                       # create .venv and install deps (incl. dev group)
uv run easy-mask [PATH]       # launch the app on an image or folder
uv run pytest                 # run tests (GUI tests run headless via pytest-qt)
uv run pytest -m unit         # one layer; see docs/testing.md
uv run pytest --cov           # coverage; CI fails under 90%
uv run ruff check --fix .     # lint
uv run ruff format .          # format
uv run mypy                   # type check
uv run pre-commit install     # one-time: ruff on every commit
```

Always use `uv` (`uv add <pkg>`, `uv run ...`). Never `pip install` into the env and never
edit `uv.lock` by hand.

## Layout and boundaries

```
src/easy_mask/
  core/     pure Python + numpy: shapes, annotation file, undo history, rasterizing.
            NO Qt imports here — this is what makes the logic easy to test.
  io/       reading images/folders, exporting datasets (masks, COCO, YOLO).
            NO Qt imports here either.
  gui/      PySide6 widgets. Thin: translate user input into core calls and draw core state.
  cli.py    argument parsing and entry point.
tests/      by type: unit/ property/ architecture/ integration/ gui/ smoke/ (docs/testing.md).
docs/       architecture.md, command-line.md, roadmap.md, ideas.md, testing.md, benchmarks.md.
```

Dependency direction is one-way: `gui -> io -> core`. If `core` ever needs to import from
`gui`, the design is wrong; stop and rethink.

## Code guidelines

- **Readable first.** A new contributor should understand any module in one sitting.
  Prefer small functions with obvious names over clever code.
- **Docstrings on every public module, class and function** (Google style, enforced by
  ruff `D` rules). Say *what* and *why*, not a restatement of the signature.
- **Type hints everywhere** in function signatures. Use `from __future__ import annotations`
  only if needed; target Python 3.12 syntax (`list[str]`, `X | None`).
- **Coordinates:** image pixel space, `(x, y)` floats, origin top-left. Widget/screen
  coordinates never leave `gui/canvas.py`.
- **Data classes for data.** `Shape` and `Annotation` are dataclasses; keep behaviour that
  needs Qt out of them.
- **No speculative abstractions.** Add an interface when the second implementation lands,
  not before (e.g. `ai.segmenter.Segmenter` arrived together with EfficientSAM).
- **Errors:** fail loudly on programmer error; show a dialog for user error (bad file,
  missing image). Don't swallow exceptions silently.
- **Tests:** every bug fix gets a regression test that fails before the fix. Every new
  core/io function gets a unit test; user-visible behaviour gets a GUI test; file-format
  changes update the golden files (`UPDATE_GOLDEN=1`) and the diff is reviewed. Pick the
  layer with docs/testing.md. Keep coverage >= 90%.

## Formatting, linting and type checking

- `ruff` formats and lints (config in `pyproject.toml`); `mypy` type-checks `src/`.
- pre-commit runs ruff on every commit; CI runs ruff and mypy on every PR.
- Don't hand-format or silence rules; a needed `# noqa`/`# type: ignore` names the code.

## Git workflow

- `main` is always releasable: CI green, app launches.
- One feature per branch: `feat/<name>`, `fix/<name>`, `docs/<name>`, `chore/<name>`.
  Open a PR into `main`; squash-merge.
- Conventional commit messages (`feat: add circle tool`, `fix: clamp points to image`).
- Do not commit datasets, exported masks, or model weights (see `.gitignore`).

## Writing docs

Keep docs short and plain: essentials only, a line per idea, no walls of text.

## Housekeeping

- Commit only what the change needs: no scratch scripts, debug output or local data.
- Remove code, tests and docs your change made unused.
- Periodically (e.g. per milestone) do a `chore:` cleanup PR: dead code, unused
  dependencies, stale docs, merged branches. Ask before deleting anything shared.

## Problem-solving principles (after Andrej Karpathy's guidelines)

These bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think before coding
Don't assume. Don't hide confusion. Surface tradeoffs.
- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them; don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop, name what is confusing, and ask.

### 2. Simplicity first
Minimum code that solves the problem. Nothing speculative.
- No features beyond what was asked; no abstractions for single-use code.
- No "flexibility" or configurability that wasn't requested.
- No error handling for impossible scenarios.
- If you wrote 200 lines and it could be 50, rewrite it.
- Ask: "Would a senior engineer call this overcomplicated?" If yes, simplify.

### 3. Surgical changes
Touch only what you must. Clean up only your own mess.
- Don't "improve" adjacent code, comments or formatting; match existing style.
- Don't refactor what isn't broken. Mention unrelated dead code; don't delete it.
- Remove imports/variables/functions that *your* change made unused.
- Test: every changed line traces directly to the request.

### 4. Goal-driven execution
Define success criteria. Loop until verified.
- "Add validation" -> write tests for invalid inputs, then make them pass.
- "Fix the bug" -> write a test that reproduces it, then make it pass.
- "Refactor X" -> tests pass before and after.
- For multi-step work, state a short plan: `1. step -> verify: check`.
