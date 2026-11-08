# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[SemVer](https://semver.org/).

## [Unreleased]

### Changed
- Labels panel: no number prefixes (keys 1-9 still work).
- Four menus (File, Edit, View, Help). Video commands are one Propagate button with a dropdown (backward, AI or copy, frames per run, interpolate, clear labels from here, undo, stop) plus the Track toggle; mask operations moved to Edit > Mask and the right-click menu; labels are picked in the Labels panel; tools live in the toolbar.
- AI Segment: a plain click always starts a new object; Shift adds to the selected one (a selected mask used to swallow new objects).
- The Export button asks the format the first time, then repeats it.

### Removed
- Separate Copy / Cut Through Frames buttons, the Video, Draw, Mask, Labels and Workflow menus, and multi-key shortcuts for them.
- Grow / Shrink / Smooth, the Auto Save and Auto Clean-Up toggles (both always on) and Change Annotation Folder from the menus (`--output` / `--no-autosave` remain).
- File > Import Video Frames (File > Open Video imports first).
- Add / Subtract Through Frames (replaced by the Cut tool, which Propagate carries through frames).
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
- Erase as a label: finish any shape with *Erase With It* (label dialog) or the ✂ Erase row (Labels panel) and it cuts the masks under it instead of being kept; Propagate carries the cut. Or propagate a shape first and right-click > *Erase With This*: it cuts the other masks on this frame and every frame it was propagated to, then goes (one Undo reverts).
- Cut tool (scissors): drag a loop to cut that part out of every mask under it, no label needed (only the selected objects, if any). Propagate and Track carry the cut part through the next frames like another object (copy without the video add-on).
- Clear Labels From Here (Propagate's dropdown): delete every shape on this and the next frames.
- One undo per video run: right after Propagate, Track, Interpolate or Clear Labels, Undo restores every frame the run touched (Undo Last Run in Propagate's dropdown works until the next run). The status bar says how many frames changed, and why when none could.
- Right-click menu on shapes: edit label, delete, duplicate, propagate, mask operations.
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
