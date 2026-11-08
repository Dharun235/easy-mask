# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[SemVer](https://semver.org/).

## [Unreleased]

### Removed
- Add / Subtract Through Frames (replaced by the Cut tool and Cut Through Frames).
- The Save toolbar button (auto-save is on; Ctrl+S and File > Save remain).
- GrabCut, Watershed and Superpixels tools (AI Segment's box prompt, the wand and the brush cover them; core GrabCut stays as a benchmark baseline).
- Workflow > Copy Shapes from Previous Image (same as Copy to Next Frames / Track).
- Line, line strip and point left the toolbar (still in the Draw menu with L / T / O).

### Changed
- onnxruntime is a regular dependency: AI Segment works without `--extra ai` (kept as an empty alias).
- Auto-save is on by default (`--no-autosave` restores the save prompt).
- AI Segment keeps only the regions that contain a positive click (click-only prompts), so
  stray blobs elsewhere in the image are dropped; neutral on the click benchmark.

### Fixed
- `shape_to_mask` returned a read-only array.
- Label colours: index-based golden-ratio palette instead of hash, so labels are clearly distinct; mask PNG palette matches.
- Double-clicking an unselected shape now opens its label dialog.
- GUI export kept only `--labels`; labels added during the session are now exported too.

### Added
- Cut tool (`K`): drag a lasso to cut that part out of every mask under it, no label needed (only the selected objects, if any). Video > Cut Through Frames (`Ctrl+Shift+X`) repeats the last cut on the next frames, the video model following the part (copy without the video add-on).
- Video > Clear Labels on Frames: delete every shape on this and the next frames.
- One undo per video run: right after AI Propagate, Copy, Track, Cut Through, Interpolate or Clear, `Ctrl+Z` restores every frame the run touched (Video > Undo Last Video Run, `Ctrl+Alt+Z`, works until the next run).
- New objects whose label is already on the frame get the next free group id, so same-label objects stay separate when propagating, cutting and exporting.
- Objects export (File > Export Folder As > Objects, or `--format objects --parts ...`): per object a binary mask PNG, a cutout of the original pixels (RGBA, cropped) and/or coordinates (`objects.json` / `objects.csv` with box, area and outline points); the choice is remembered per folder.
- AI Propagate vs Copy: separate actions and toolbar buttons. AI Propagate gives the frame's masks (from any area shape) to EfficientTAM / SAM 2.1; Copy repeats them unchanged. Without the video add-on, AI Propagate explains how to install it and offers copying instead.
- Progress feedback: a status-bar progress panel with Cancel for AI model download / image preparation, propagation, export and video import; exports and video imports run in the background.
- File > Import Video Frames: choose output folder, PNG / JPEG / WebP and quality, every n-th frame and a frame limit; opening a not-yet-imported video shows this dialog.
- Calm minimal theme (light / dark, follows the system), Lucide icons, one icon-only toolbar, quieter start page; side panels hidden until an image is open; View > Panels.
- Automation: track objects into the next frame on Next Image (Video > Track Objects into Next Frame, Ctrl+Shift+T); auto clean-up of AI / smart-tool masks (Mask > Auto Clean-Up); File > Export Again (Ctrl+Shift+E) and Export Again When Closing, with the last export remembered per folder; review mode visits automatic shapes first.
- Ease of use: active label with keys 1-9, start page with recent folders, icon toolbar with shortcut tooltips, Help > Keyboard Shortcuts (F1), remembered window layout / tool sizes / recent folders, per-folder `.easy-mask.json` (labels, flags, active label, annotation folder), AI Segment prefetch of upcoming images.
- Workflow: COCO RLE export (`--rle`), `easy-mask stats` and a statistics dialog, review mode (`N` / `Shift+N`, `reviewed` flag), copy shapes from previous image, `--strict-labels`; RLE vs polygon benchmark.
- Classic pixel tools: magic wand (`W`), GrabCut box (`G`), superpixel picker (`S`), watershed from markers (`M`), threshold / colour-range panel with live preview (`Ctrl+T`); Shift adds to and Alt subtracts from the selected mask. Classic-tools benchmark.
- View > Adjust Display: brightness, contrast, gamma, CLAHE, single-channel view and mask opacity (display only).
- AI Segment tool (`I`): click / right-click / drag a box, live preview, Enter or double-click keeps the mask. EfficientSAM ViT-T (or `--ai-model efficientsam-vits`) on CPU via onnxruntime (`uv sync --extra ai`); weights downloaded once to `~/.cache/easy-mask/models`, sha256-checked; image encoding runs in the background. `Segmenter` interface in `easy_mask.ai`; click benchmark `benchmarks.clicks` (NoC@85/90, NoF@90, mIoU@k).
- Video: open mp4/avi/mov by extracting frames (`--frame-step`, `--max-frames`); keyframe interpolation (outline correspondence, SDF for masks); mask propagation with EfficientTAM-Ti (default) or SAM 2.1 Hiera-Tiny on a background thread with cancel, optional `video` extra; propagates only the selected objects if any are selected; without the extra (or `--video-model copy`) it copies masks unchanged; DAVIS-style export (`--format davis`); DAVIS 2017 J&F benchmark.
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
