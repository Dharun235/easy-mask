# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[SemVer](https://semver.org/).

## [Unreleased]

### Fixed
- `shape_to_mask` returned a read-only array.
- Label colours: index-based golden-ratio palette instead of hash, so labels are clearly distinct; mask PNG palette matches.
- Double-clicking an unselected shape now opens its label dialog.
- GUI export kept only `--labels`; labels added during the session are now exported too.

### Added
- Workflow: COCO RLE export (`--rle`), `easy-mask stats` and a statistics dialog, review mode (`N` / `Shift+N`, `reviewed` flag), copy shapes from previous image, `--strict-labels`; RLE vs polygon benchmark.
- Mask shapes (labelme-compatible), brush/eraser, multi-select with union/subtract/intersect, mask clean-up and mask <-> polygon conversion; mask-fidelity benchmark.
- mypy type checking in CI; short formatting and housekeeping notes in CLAUDE.md.
- Layered test suite (unit, property-based, architecture, integration, golden-file, GUI, smoke, benchmark-data), coverage gate at 90%, docs/testing.md.
- Annotation GUI (PySide6): polygon, rectangle, circle, line, line strip and point tools.
- Edit mode: select, move shapes, drag vertices, delete, duplicate, copy/paste, relabel.
- Snapshot undo/redo; label dialog with autocomplete, group id and description.
- Panels for files (done-marks), labels (counts), shapes (visibility) and image flags.
- Zoom at cursor, pan, fit window; labelme-compatible JSON with auto-save and output folder.
- `easy-mask export` and File > Export: semantic/instance mask PNG, COCO, YOLO-seg.
- README badges, related projects, TODO, contributors; benchmarks doc; issue templates, Code of Conduct, CITATION.cff.
- Project scaffold: uv packaging, ruff, pytest, pre-commit, CI, CLAUDE.md, docs.
