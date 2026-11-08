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
- [x] Clean-up: fill holes, remove small specks, grow / shrink, smooth edges
- [x] Polygon <-> mask conversion with a simplification tolerance
- [x] Mask-fidelity benchmark on GrabCut / Berkeley / DAVIS

## v0.3 — classic pixel tools (`feat/pixel-tools`)
- [x] Threshold (manual / Otsu / adaptive), colour range, magic wand
- [x] GrabCut from a box, watershed from markers, superpixel picking
- [x] View adjustments: brightness / contrast / gamma, CLAHE, channel view, mask opacity
- [x] Classic-tools benchmark on GrabCut / Berkeley / DAVIS

## v0.4 — AI click-to-segment on CPU (`feat/ai-segment`)
- [ ] `Segmenter` interface with EfficientSAM (ONNX, CPU) as first backend
- [ ] Positive / negative clicks and box prompts, live preview, result is a mask shape
- [ ] Weights downloaded on first use into a user cache dir; encoding off the UI thread
- [ ] Click benchmark: NoC@85/90 and mIoU@k on GrabCut / Berkeley / DAVIS

## v0.5 — video (`feat/video`)
- [ ] Open a video file or frame folder
- [ ] Propagate masks across frames with memory (SAM 2 / EfficientTAM), correct and re-run
- [ ] Keyframe interpolation and copy-forward without a model
- [ ] Export DAVIS-style mask sequences

## v0.6 — workflow (`feat/workflow`)
- [ ] Review mode, per-label statistics, COCO RLE export

## Later / ideas
- Text-prompted detection (e.g. Grounding-DINO / SAM3-style) to pre-label
- Keep previous frame's annotations when moving to next image
- Dataset statistics panel, label validation against a fixed list
- Pascal VOC export, import from COCO
- Standalone app bundles (PyInstaller)
