# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[SemVer](https://semver.org/).

## [Unreleased]

### Changed
- Brush is now **Paint** and works by where a stroke starts: on a mask it extends that mask, on empty image it makes a new mask. No need to select first (a still-selected shape no longer swallows strokes painted elsewhere).
- Undo / Redo cover the whole folder, not just the frame on screen: every edit on any frame is a step, a Propagate / erase / clear / interpolate run is one step over all its frames, and undoing an edit made on another frame goes back to it. Undo Last Run is gone (plain Undo does it).
- Toolbar buttons have a word under each icon, and only the everyday tools are on it (Select, AI, Polygon, Box, Brush, Eraser); circle, line, point and magic wand keep their keys. Tools, navigation, Propagate and Export are disabled until an image is open.
- Propagate's dropdown in plain words: Use AI, Auto-Propagate on Next Image (was Track, no longer a separate button), How Many Frames; Stop moved to the progress bar's Cancel.
- Label dialog: group id and description folded under More options.
- Start page: a one-line Open → AI or draw → Propagate → Export guide instead of key hints.
- Labels panel: no number prefixes (keys 1-9 still work).
- Four menus (File, Edit, View, Help). Video commands are one Propagate button with a dropdown (backward, AI or copy, frames per run, interpolate, clear labels from here, undo, stop) plus the Track toggle; mask operations moved to Edit > Mask and the right-click menu; labels are picked in the Labels panel; tools live in the toolbar.
- AI Segment: a plain click always starts a new object; Shift adds to the selected one (a selected mask used to swallow new objects).
- The Export button asks the format the first time, then repeats it.

### Removed
- The separate Cut tool (the Eraser does it), Copy / Cut Through Frames buttons, the Video, Draw, Mask, Labels and Workflow menus, and multi-key shortcuts for them.
- Grow / Shrink / Smooth, the Auto Save and Auto Clean-Up toggles (both always on) and Change Annotation Folder from the menus (`--output` / `--no-autosave` remain).
- File > Import Video Frames (File > Open Video imports first).
- Add / Subtract Through Frames (replaced by the Eraser and Erase shapes, which Propagate carries through frames).
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
- Labels, flags, the active label and the export target of one folder (or video) no longer carry over into the next folder opened, nor get written into its project file.
- `shape_to_mask` returned a read-only array.
- Label colours: index-based golden-ratio palette instead of hash, so labels are clearly distinct; mask PNG palette matches.
- Double-clicking an unselected shape now opens its label dialog.
- GUI export kept only `--labels`; labels added during the session are now exported too.

### Added
- The status bar explains the active tool (how to place, finish, resize or cancel) while it is in use, and hovering any toolbar button or Propagate option explains what it does.
- Label management in the Labels panel: an *Add a label* box; right-click a label to use it, apply it to the selected shapes, rename it on every frame, or delete it from this frame or from all frames (each one undo step). Right-click a mask > Label to change its label. New masks announce their label in the status bar.
- Erase as a label: finish any shape with *Erase With It* (label dialog) and it cuts the masks under it instead of being kept; Propagate carries the cut. Or propagate a shape first and right-click > *Erase With This*: it cuts the other masks on this frame and every frame it was propagated to, then goes (one Undo reverts).
- Eraser removes what you paint from every mask under it (only the selected shapes, if any), and Propagate carries the erased part through the next frames like another object (copy without the video add-on).
- Clear Labels From Here (Propagate's dropdown): delete every shape on this and the next frames.
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
