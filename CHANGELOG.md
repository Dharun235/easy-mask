# Changelog

All notable changes are listed here. Format:
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), versions: [SemVer](https://semver.org/).

## [Unreleased]

### Changed
- Shorter README for non-programmers; command-line usage moved to `docs/command-line.md`.
- Depends on `PySide6-Essentials` instead of the full PySide6 (~700 MB smaller install).

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
