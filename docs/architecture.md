# Architecture

easy-mask is split into three layers with a one-way dependency: `gui -> io -> core`.

| Layer | Knows about | Must not import |
| --- | --- | --- |
| `core/` | numpy, Pillow, OpenCV (headless), stdlib | Qt, `io`, `gui` |
| `io/` | `core`, numpy, Pillow | Qt, `gui` |
| `ai/` | `core`, numpy, onnxruntime (optional extra) | Qt, `io`, `gui` |
| `gui/` | everything | — |

Keeping Qt out of `core` and `io` means the annotation logic (geometry, file format,
rasterizing, export) is plain Python that runs and tests without a display.

## Module map

```
src/easy_mask/
  cli.py                 argparse: `easy-mask [PATH]` (GUI) and `easy-mask export` (headless)
  core/
    geometry.py          distances, point-in-polygon, area, bbox, circle -> polygon
    shapes.py            ShapeType enum + Shape dataclass (hit-testing, outline, (de)serialize)
    annotation.py        Annotation dataclass, labelme-compatible JSON load/save, file paths
    history.py           snapshot-based undo/redo stack
    labels.py            stable per-label colours, parsing --labels / --flags
    rasterize.py         Shape -> boolean mask; semantic and instance masks
    masks.py             mask shapes <-> full masks, brush strokes, clean-up, mask -> polygons
    operations.py        pure list edits: combine (union/subtract/intersect), map, convert
  ai/
    segmenter.py         Segmenter interface: cached image embedding + click/box prompts -> masks
    efficient_sam.py     EfficientSAM ONNX backend (onnxruntime, CPU), model table, load()
    weights.py           download model files to ~/.cache/easy-mask/models, sha256-checked
  io/
    images.py            list images in a folder (natural sort), read size
    export/
      common.py          collect (image, annotation) pairs, resolve class list, copy images
      masks.py           semantic + instance PNG masks
      coco.py            COCO instance segmentation JSON
      yolo.py            YOLO-seg label files + data.yaml
  gui/
    app.py               QApplication bootstrap
    main_window.py       menus, docks, open/save/navigate, edit commands, export menu
    canvas.py            view transform, drawing tools, multi-select/move/vertex drag, painting
    tools/base.py        CanvasTool: plug-in point for pixel tools (canvas forwards events)
    tools/brush.py       brush and eraser
    tools/ai_segment.py  AI Segment: prompts, mask preview, accept into a mask shape
    tools/ai_worker.py   QThreadPool job that loads the model and encodes images
    label_dialog.py      label / group id / description dialog with autocomplete
    widgets.py           list widgets for files, labels, shapes, flags
```

## Data flow for one edit

```
mouse/keys -> Canvas ----shape_drawn----> MainWindow._on_shape_drawn -> LabelDialog
                    \---shapes_edited---> MainWindow._commit
                                              |- History.push(shapes)
                                              |- dirty = True (title shows *)
                                              '- refresh Labels / Shapes panels
```

The canvas mutates shapes only while dragging; adding, deleting and relabelling happen
in `MainWindow`, which always ends with `_commit()`. Undo/redo replaces the whole shape
list with a snapshot from `History`.

## Coordinates

All stored points are image pixels, origin top-left, `(x, y)` floats. `Canvas` converts
widget positions with `to_image()`; `scale` and `offset` are its only view state.

## Pixel tools

Vector shapes are drawn by the canvas itself. Anything that works on pixels (brush,
eraser, and later magic wand or AI clicks) subclasses `CanvasTool` and is activated with
`Canvas.set_tool`; the canvas forwards mouse, key and paint events. A tool reports its
result through a callback, and `MainWindow` stores it and records undo history.

## AI Segment

```
activate (tool on / new image) --EncodeJob (worker thread)--> Segmenter.set_image(rgb, path)
                                                      done --> tool.ready, predict waiting clicks
click / right-click / drag box --> Segmenter.predict(points, labels, box) --> preview mask
Enter / double-click ----------> MainWindow._apply_stroke(full_mask, index_or_None)
```

* `Segmenter` (in `ai/`, no Qt) caches embeddings per image path (LRU, 8 images), so
  going back to an image is instant. `predict` takes at most 6 points (a box uses two),
  keeping the most recent clicks, and ranks the model's candidate masks: by predicted IoU,
  except for a lone click, where the largest (whole object) wins.
* EfficientSAM ignores negative labels in practice, so `EfficientSam.predict` segments
  with the positive prompts and cuts out what a click on each negative point selects
  (see docs/benchmarks.md for the effect).
* Encoding runs in a one-thread `QThreadPool`; each request gets a generation number and
  only the newest result is used, so fast image switching cannot show a stale mask.
  Prediction (~50-100 ms) runs on the UI thread.
* A backend for another model (e.g. SAM 2 for video) subclasses `Segmenter` and
  implements `_encode(rgb)` and `_decode(embedding, size, coords, labels)`.
