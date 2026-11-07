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

## v0.2 — SAM2 click-to-segment (`feat/sam2`)
The headline feature. Runs locally on CPU, CUDA or Apple MPS.
- [ ] `Segmenter` backend interface (introduced with its first real implementation)
- [ ] SAM2 backend: embed image once, cache per image, prompt with points/boxes
- [ ] Positive / negative clicks, box prompt, live preview mask
- [ ] Accept as `mask` shape (bitmap) or polygonize to `polygon`
- [ ] Model manager: download weights on first use into a user cache dir
- [ ] Background thread so the UI never freezes during encoding

## v0.3 — mask editing
- [ ] Native `mask` shape type (RLE / PNG in JSON, labelme-compatible)
- [ ] Brush and eraser to refine masks
- [ ] Polygon <-> mask conversion, simplify polygon tolerance slider

## v0.4 — video and propagation
- [ ] Open video / frame folder
- [ ] SAM2 video predictor: propagate masks across frames, correct and re-propagate
- [ ] Object tracks with consistent ids

## Later / ideas
- Text-prompted detection (e.g. Grounding-DINO / SAM3-style) to pre-label
- Brightness / contrast adjustment for dark images
- Keep previous frame's annotations when moving to next image
- Dataset statistics panel, label validation against a fixed list
- Pascal VOC export, import from COCO
- Standalone app bundles (PyInstaller)
