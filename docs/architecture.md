# Architecture

easy-mask is split into three layers with a one-way dependency: `gui -> io -> core`.

| Layer | Knows about | Must not import |
| --- | --- | --- |
| `core/` | numpy, Pillow, OpenCV (headless), stdlib | Qt, `io`, `gui` |
| `io/` | `core`, numpy, Pillow | Qt, `gui` |
| `ai/` | `core`, numpy, onnxruntime (optional extra) | Qt, `io`, `gui` |
| `propagation/` | `core`, numpy, Pillow; PyTorch lazily (`video` extra) | Qt, `gui` |
| `gui/` | everything | — |

Keeping Qt out of `core` and `io` means the annotation logic (geometry, file format,
rasterizing, export) is plain Python that runs and tests without a display.

## Module map

```
src/easy_mask/
  cli.py                 argparse: `easy-mask [PATH]` (GUI), `easy-mask export` and `easy-mask stats`
  core/
    geometry.py          distances, point-in-polygon, area, bbox, circle -> polygon
    shapes.py            ShapeType enum + Shape dataclass (hit-testing, outline, (de)serialize)
    annotation.py        Annotation dataclass, labelme-compatible JSON load/save, file paths
    keypoints.py         a mask's centroid + rotated-box corners as editable point shapes
    history.py           undo/redo across every frame of a folder (snapshots per step)
    labels.py            stable per-label colours, parsing --labels / --flags
    rasterize.py         Shape -> boolean mask; semantic and instance masks
    masks.py             mask shapes <-> full masks, brush strokes, clean-up, mask -> polygons
    operations.py        pure list edits: combine (union/subtract/intersect), map, convert
    rle.py               COCO RLE encode/decode (pycocotools-compatible, pure numpy)
    pixel_ops.py         classic segmentation: threshold, magic wand, colour range, GrabCut,
                         (image -> full-size bool mask); GrabCut kept as a benchmark baseline
    display.py           display-only adjustments (brightness, gamma, CLAHE, channel, opacity)
  ai/
    segmenter.py         Segmenter interface: cached image embedding + click/box prompts -> masks
    efficient_sam.py     EfficientSAM ONNX backend (onnxruntime, CPU), model table, load()
    weights.py           download model files to ~/.cache/easy-mask/models, sha256-checked
  io/
    images.py            list images in a folder (natural sort), read size
    project.py           .easy-mask.json per image folder (labels, flags, active label)
    stats.py             folder statistics and label validation (CLI `stats`, GUI dialog)
    video.py             object identity across frames, auto vs manual shapes, merging results
    interpolate.py       keyframe interpolation: outline correspondence, SDF mask blending
  io/
    images.py            list images in a folder (natural sort), read size
    video.py             extract a video's frames to <stem>_frames/ (step, max frames)
    export/
      common.py          collect (image, annotation) pairs, resolve class list, copy images
      masks.py           semantic + instance PNG masks
      coco.py            COCO instance segmentation JSON (polygons, or RLE for masks)
      yolo.py            YOLO-seg label files + data.yaml
      davis.py           DAVIS-style indexed PNG per frame, object ids stable across frames
  propagation/
    tracker.py           Propagator protocol, CopyPropagator (no model); track(): shapes -> prompts -> flagged masks
    backend.py           EfficientTAM / SAM 2.1 video predictor, weights download, device pick
    configs/             EfficientTAM config (its wheel ships none)
  gui/
    app.py               QApplication bootstrap
    main_window.py       menus, docks, open/save/navigate, edit commands, export menu
    canvas.py            view transform, drawing tools, multi-select/move/vertex drag, painting
    session.py           remembered state: window layout, tool sizes, recent folders, projects
    welcome.py           start page (open buttons, recent folders)
    shortcuts.py         Help > Keyboard Shortcuts, built from the menus
    icons.py, icons/     Lucide toolbar icons (ISC), tinted to the palette
    theme.py             calm light / dark theme: palette, stylesheet, Fusion style
    progress.py          status-bar progress panel (several tasks, Cancel)
    export_worker.py     dataset export on a background thread
    object_export_dialog.py  Objects export: choose binary masks / cutouts / coordinates
    video_import.py      video import dialog (format, folder, every n-th frame) + worker
    tools/base.py        CanvasTool: plug-in point for pixel tools (canvas forwards events)
    tools/brush.py       brush and eraser
    tools/region.py      magic wand (Shift add / Alt subtract)
    threshold_dialog.py  threshold / colour-range panel with live canvas preview
    display_dialog.py    View > Adjust Display panel
    tools/ai_segment.py  AI Segment: prompts, mask preview, accept into a mask shape
    tools/ai_worker.py   QThreadPool job that loads the model and encodes images
    label_dialog.py      label / group id / description dialog with autocomplete
    widgets.py           list widgets for files, labels, shapes, flags
    workflow.py          Workflow menu: review mode, statistics dialog
    video_menu.py        Video menu: open video, interpolate, propagate / stop
    propagate.py         PropagationWorker: QThread running the tracker, one signal per frame
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
in `MainWindow`, which always ends with `_commit()`. One `History` covers the whole folder:
`_commit()` records the shown frame's shapes before and after the edit, and a video run
records every frame it writes inside one group, so it is one step. Undo/redo returns each
touched frame's state: the shown frame is replaced on screen, others are written to disk
(None removes the file), and a one-frame step on another frame first goes to that frame.

## Coordinates

All stored points are image pixels, origin top-left, `(x, y)` floats. `Canvas` converts
widget positions with `to_image()`; `scale` and `offset` are its only view state.

## Pixel tools

Vector shapes are drawn by the canvas itself. Anything that works on pixels (brush,
eraser, magic wand, AI Segment) subclasses
`CanvasTool` and is activated with `Canvas.set_tool`; the canvas forwards mouse, key and
paint events. A tool reports its result through a callback, and `MainWindow` stores it and records undo history.
The algorithms live in `core/pixel_ops.py` and read the original pixels from
`Canvas.image_rgb()`; display adjustments (`core/display.py`) only change what the canvas
draws, so tools, saving and export never see them.

## AI Segment
## Video

A video is a folder of frames (a video file is extracted to one first), so it reuses the
file list, navigation and one-JSON-per-frame storage unchanged. An object's identity
across frames is `(label, group_id)`; `free_group_id` gives a new object with a label
already on the frame its own group id.

```
Propagate button -> VideoMenu.propagate: confirm this frame's shapes (drop auto flags),
                     read the range's annotations, start PropagationWorker
PropagationWorker (QThread) -> track(propagator, frames, shapes)
                                 |- prompts: this frame's area shapes (rasterized)
                                 |           + hand-drawn shapes of those objects later on
                                 '- Propagator.propagate() -> masks per frame
      frame_done(position, shapes) --queued signal--> VideoMenu._store (GUI thread)
                                 |- frame on screen: merge_auto -> MainWindow._replace_shapes
                                 '- other frames:   load JSON, merge_auto, save
```

`merge_auto` replaces an object's earlier automatic shapes (flag `propagated` or
`interpolated`) and never touches hand-drawn ones, which is what makes "fix a frame, run
again" safe. The model sits behind the small `Propagator` protocol, so GUI tests use a
deterministic fake and the real backend (PyTorch) is only imported when first used.

What is erased on a frame (by the Eraser or an Erase shape, via `MainWindow.cut`) is
remembered as `VideoMenu.last_cut`: the removed pixels, not the whole stroke or shape.
Propagate and auto-propagate send it along as one more object (a temporary `__region__`
label) and call `cut_region` on each frame with where it moved;
masks the cut is fixing do not prompt the model in that run, or they would pull the part
back. A shape finished with the reserved label `ERASE_LABEL` goes through the same cut
path instead of being stored. `VideoMenu.erase_with` turns an object into an eraser on
its frame and the contiguous frames holding automatic copies of it (one undo step).

Per-folder settings (labels, flags, active label, annotation and export folders) come only
from the command line and that folder's `.easy-mask.json`: `MainWindow.reset_folder_state`
clears the previous folder's before a new one is applied.

## Where SAM2 will plug in

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
