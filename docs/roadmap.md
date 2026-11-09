# Roadmap

Each milestone lands as its own branch and PR into `main`.

## v0.1 — core annotation (`feat/core-annotation`)
Manual tools on par with classic annotators.
- [x] Polygon, rectangle, circle, line, line strip, point
- [x] Select / move / vertex drag / delete / duplicate / copy-paste
- [x] Undo / redo
- [x] Label dialog, label colours, label list with counts, shape list with visibility
- [x] Open image / folder, file list, next / prev, zoom, pan, fit
- [x] Image-level flags
- [x] labelme-compatible JSON, auto-save, output folder
- [x] Export: semantic + instance mask PNG, COCO, YOLO-seg

## v0.2 — mask shapes (`feat/mask-shape`)
- [x] Native `mask` shape type (PNG crop in JSON, labelme-compatible), pixel-exact export
- [x] Brush and eraser (`B` / `X`, `[` `]` for size); brushing a polygon turns it into a mask
- [x] Multi-select (Shift/Ctrl+click, shape list) and union / subtract / intersect
- [x] Clean-up: fill holes, remove small specks
- [x] Polygon <-> mask conversion with a simplification tolerance
- [x] Mask-fidelity benchmark on GrabCut / Berkeley / DAVIS

## v0.3 — classic pixel tools (`feat/pixel-tools`)
- [x] Threshold (manual / Otsu / adaptive), colour range, magic wand
- [x] ~~GrabCut box, watershed, superpixels~~ (removed in v0.9: AI Segment's box prompt and the brush cover them)
- [x] View adjustments: brightness / contrast / gamma, CLAHE, channel view, mask opacity
- [x] Classic-tools benchmark on GrabCut / Berkeley / DAVIS

## v0.4 — AI click-to-segment on CPU (`feat/ai-segment`)
- [x] `Segmenter` interface with EfficientSAM (ONNX, CPU) as first backend
- [x] Positive / negative clicks and box prompts, live preview, result is a mask shape
- [x] Weights downloaded on first use into a user cache dir; encoding off the UI thread
- [x] Click benchmark: NoC@85/90 and mIoU@k on GrabCut / Berkeley / DAVIS

## v0.5 — video (`feat/video`)
- [x] Open a video file (frames extracted with step / max-frames) or frame folder
- [x] Propagate masks across frames with memory (EfficientTAM / SAM 2.1), correct and re-run
- [x] Keyframe interpolation without a model (outline correspondence, SDF for masks)
- [x] Copy-forward without a model (Propagate copies the masks when the video extra is missing)
- [x] Export DAVIS-style mask sequences
- [x] Video benchmark: J&F and seconds per frame on DAVIS 2017 val

## v0.6 — workflow (`feat/workflow`)
- [x] COCO RLE export (`--rle`, pure numpy, byte-identical to pycocotools)
- [x] Folder statistics (`easy-mask stats`, File > Statistics): per-label objects, shapes, area
- [x] Review mode: step through every object (`N` / `Shift+N`), zoom and select, `reviewed` flag
- [x] Copy shapes from the previous image (undoable)
- [x] Label validation: `--strict-labels`, unknown-label report in stats
- [x] RLE vs polygon export benchmark

## v0.7 — ease of use (`feat/ux`)
- [x] Active label with keys 1-9: no label dialog per shape
- [x] Auto-save on by default
- [x] Remember window layout, tool sizes, recent folders; per-folder project file
- [x] AI Segment prefetch: next images encoded in the background (862 ms -> 8 ms)
- [x] Icon toolbar grouped by kind, shortcut tooltips, F1 shortcut list, start page
- [x] Track while navigating: auto-propagate selected objects to the next frame
- [x] Auto clean-up of AI masks on accept (fill holes, remove specks)
- [x] AI out of the box (onnxruntime as a regular dependency)
- [x] Remember export format/folder; Export Again (Ctrl+Shift+E) and export on close
- [x] Review mode visits automatic "(auto)" shapes first

## v0.8 — simpler masks and video (`feat/polish`)
- [x] Toolbar trimmed to the essentials, every button labelled, status-bar hint per tool
- [x] Paint and Eraser; eraser cuts carried through frames by Propagate, Erase With This
- [x] Labels panel as a checklist tree (labels with their shapes); right-click a label to
      propagate, hide, rename or delete it (this frame, a frame range, all frames)
- [x] One undo history across all frames; a video run is one step
- [x] Key points: centroid and rotated-box corners as editable points, kept by Propagate
- [x] Export window: format, parts (masks, cutouts, coordinates, centroid, corners,
      points, overlays), frame range, folder

## Later / ideas
- Text-prompted detection (e.g. Grounding-DINO / SAM3-style) to pre-label
- Pascal VOC export, import from COCO
- Standalone app bundles (PyInstaller)
