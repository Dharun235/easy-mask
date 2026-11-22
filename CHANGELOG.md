# Changelog

All notable changes are listed here. Format:
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), versions: [SemVer](https://semver.org/).

## [Unreleased]

### Fixed
- Propagation no longer floods the terminal with progress bars and a model warning.
- Progress shows "Starting AI tracking" while the model loads, then "AI tracking 13 of 30",
  instead of sitting at 0%; the tool hint no longer overlaps it.

### Changed
- Calm Nordic look: cool neutrals and one blue accent.
- Status bar says "New masks: <label>"; Back / Next say when at the first or last image.
- Recent folders show the folder name first.

### Removed
- Export Again, Statistics and review mode from the app (`easy-mask stats` remains).
- Hidden circle, line, line strip and magic wand tools, the threshold panel and the
  label number keys.

## [0.1.0] - 2026-11-09

First release.

### Added
- **Draw:** polygon, box, circle, line, point and pixel masks; **Paint** and **Eraser**
  brushes; magic wand and threshold tools.
- **AI Segment:** click or drag a box around an object and EfficientSAM outlines it, on
  the CPU, offline after a one-time download.
- **Video:** open a video or frame folder; right-click a label > **Propagate** to track
  it through frames (EfficientTAM / SAM 2.1, or plain copy without the video extra);
  erased parts are carried too; keyframe interpolation; clear frames.
- **Labels panel:** a checklist of labels with their shapes; right-click a label to
  propagate, hide, rename or delete it from this frame, a frame range or all frames.
- **Points:** each mask's centroid and rotated-box corners as points you can drag.
- **Undo / Redo** across every frame; a whole propagate run is one step.
- **Export window:** semantic / instance mask PNGs, COCO (polygons or RLE), YOLO-seg,
  DAVIS, and per-object masks, cutouts, coordinates, centroid, corners, points and frames
  with overlays; any frame range; Export Again and export on close.
- Auto-save, labelme-compatible JSON, light / dark theme, status-bar hints for every
  tool, progress bar with Cancel for slow tasks, start page with recent folders.
- Command line for scripts: `easy-mask export` and `easy-mask stats`.
