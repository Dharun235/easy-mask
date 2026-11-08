# CLAUDE.md

Guidance for Claude Code (and any AI or human contributor) working in this repo.

## Project in one paragraph

**easy-mask** is a local-first desktop tool for annotating images with object masks for
computer vision. Users draw polygons, rectangles, circles, lines and points (and, in later
branches, click objects for SAM2 to segment). Everything runs on the user's machine: no
cloud, no account, no telemetry. Annotations are stored as one JSON file per image
(labelme-compatible) and export to mask PNGs, COCO and YOLO-seg.

## Commands

```bash
uv sync                       # create .venv and install deps (incl. dev group)
uv run easy-mask [PATH]       # launch the app on an image or folder
uv run pytest                 # run tests (GUI tests run headless via pytest-qt)
uv run pytest -m unit         # one layer; see docs/testing.md
uv run pytest --cov           # coverage; CI fails under 90%
uv run ruff check --fix .     # lint + autofix
uv run ruff format .          # format
uv run mypy                   # type check src/
uv run pre-commit install     # one-time: ruff + mypy + whitespace checks on every commit
uv run pre-commit run --all-files   # run every hook on the whole repo
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
docs/       architecture.md, roadmap.md, testing.md, benchmarks.md.
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
  not before (e.g. the segmenter backend interface arrives with the SAM2 branch).
- **Errors:** fail loudly on programmer error; show a dialog for user error (bad file,
  missing image). Don't swallow exceptions silently.
- **Tests:** every bug fix gets a regression test that fails before the fix. Every new
  core/io function gets a unit test; user-visible behaviour gets a GUI test; file-format
  changes update the golden files (`UPDATE_GOLDEN=1`) and the diff is reviewed. Pick the
  layer with docs/testing.md. Keep coverage >= 90%.

## Formatting, linting and type checking

All three are automatic; you should rarely run them by hand.

| Tool | What it enforces | Runs on |
| --- | --- | --- |
| `ruff format` | Black-compatible formatting, 100-char lines, double quotes | save (VS Code), every Claude edit, commit |
| `ruff check` | pycodestyle, pyflakes, isort, bugbear, pyupgrade, simplify, naming, Google docstrings | same + CI |
| `mypy` | Every function typed; strict-ish checks configured in `pyproject.toml` | commit + CI |
| pre-commit hooks | Trailing whitespace, final newline, LF endings, valid YAML/TOML, no merge markers, no `breakpoint()`, no files > 1 MB | commit |
| `.editorconfig` | UTF-8, LF, 4-space indent (2 for YAML/JSON/TOML/MD) | any editor |

Where it is wired up:
- **Claude Code:** `.claude/settings.json` runs `.claude/hooks/format_python.py` after every
  Edit/Write, so Python files Claude touches are fixed and formatted immediately.
- **Git:** `.pre-commit-config.yaml` (install once with `uv run pre-commit install`). Ruff and
  mypy run through `uv`, so hook versions always match `uv.lock`.
- **Editor:** `.vscode/settings.json` formats and organises imports on save; recommended
  extensions are in `.vscode/extensions.json`.
- **CI:** the `lint` job runs `ruff check`, `ruff format --check` and `mypy`; a PR cannot merge
  red.

Rules:
- Never hand-format: let the formatter decide, and don't fight it with odd line breaks.
- Don't silence a rule to get green. If a `# noqa: <CODE>` or `# type: ignore[<code>]` is
  truly needed, use the specific code and add a short reason on the same line.
- Changing the rule set (`[tool.ruff]`, `[tool.mypy]`) is its own `chore:` PR, with the
  whole repo fixed in the same PR.
- `tests/data/` is excluded from whitespace fixers: golden files stay exactly as generated.

## Git workflow

- `main` is always releasable: CI green, app launches.
- One feature per branch: `feat/<name>`, `fix/<name>`, `docs/<name>`, `chore/<name>`.
  Open a PR into `main`; squash-merge.
- Conventional commit messages (`feat: add circle tool`, `fix: clamp points to image`).
- Do not commit datasets, exported masks, or model weights (see `.gitignore`).

## Housekeeping and cleanup

Keep the repo small and obvious. A file nobody needs is a file every newcomer must read.

**Before every commit**
- `git status` shows only files you meant to change. No scratch scripts, debug prints,
  screenshots, notebooks, `*.orig`/`*.rej`, or local data.
- Throwaway scripts and experiments live outside the repo (or in a gitignored folder),
  never in `src/` or `tests/`.
- Remove imports, functions, fixtures and docs that *your* change made unused.
- A new file type that should never be committed gets a `.gitignore` entry in the same PR.

**Regular cleanup pass** (at the end of each milestone, or about monthly), as its own
`chore/cleanup-<date>` PR, never mixed into feature work:

1. **Untracked junk:** `git clean -ndX` lists ignored files (preview), `git clean -ndx`
   also lists untracked files. Delete only after reading the list; never `-f` blindly.
   Caches (`.pytest_cache`, `.ruff_cache`, `.hypothesis`, `__pycache__`, `.coverage`) are
   always safe to remove; `.venv` is rebuilt by `uv sync`; `data/` holds downloaded
   benchmarks, so keep it.
2. **Dead code:** `uv run ruff check . --select F401,F841` (unused imports/variables).
   Look for functions with no callers (`grep -rn "def name"` vs. usages) and delete them
   along with their tests. Commented-out code goes: git remembers it.
3. **Dependencies:** each entry in `pyproject.toml` must still be imported somewhere;
   remove the rest with `uv remove <pkg>`. Then `uv lock --upgrade` and run the full
   suite to pick up fixes.
4. **Tests:** delete tests for removed features; refresh golden files only with a
   reviewed diff (`UPDATE_GOLDEN=1`); keep coverage >= 90%.
5. **Docs:** README, CLAUDE.md, docs/*.md and the CHANGELOG still match the code (commands
   run, paths exist, roadmap checkboxes are true). Fix or delete stale sections.
6. **Branches:** delete merged or abandoned branches, locally (`git branch -d`) and on
   GitHub, once their PR is merged or closed for good. Ask the maintainer before deleting
   a branch with unmerged work.
7. **Size check:** `git ls-files | xargs du -h | sort -rh | head` finds big files. Nothing
   over ~1 MB without a reason (pre-commit enforces this for new files).

Deleting files or branches is irreversible for collaborators: list what you plan to
remove and get a yes before doing it.

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
