# Architecture

easy-mask is split into three layers with a one-way dependency: `gui -> io -> core`.

| Layer | Knows about | Must not import |
| --- | --- | --- |
| `core/` | numpy, Pillow, OpenCV (headless), stdlib | Qt, `io`, `gui` |
| `io/` | `core`, numpy, Pillow | Qt, `gui` |
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
    pixel_ops.py         classic segmentation: threshold, magic wand, colour range, GrabCut,
                         watershed, superpixels (image -> full-size bool mask)
    display.py           display-only adjustments (brightness, gamma, CLAHE, channel, opacity)
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
    tools/region.py      magic wand, GrabCut box, superpixel picker (Shift add / Alt subtract)
    tools/watershed.py   watershed from object / background marker scribbles
    threshold_dialog.py  threshold / colour-range panel with live canvas preview
    display_dialog.py    View > Adjust Display panel
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
eraser, magic wand, GrabCut, superpixels, watershed, and later AI clicks) subclasses
`CanvasTool` and is activated with `Canvas.set_tool`; the canvas forwards mouse, key and
paint events. A tool reports its result through a callback, and `MainWindow` stores it and records undo history.
The algorithms live in `core/pixel_ops.py` and read the original pixels from
`Canvas.image_rgb()`; display adjustments (`core/display.py`) only change what the canvas
draws, so tools, saving and export never see them.

## Where SAM2 will plug in

The `feat/sam2` branch adds a `segmenters/` package (behind an interface introduced with its
first real backend), a click-prompt tool in `Canvas`, and a background worker so image
encoding never blocks the UI. Masks it produces flow through the same `_commit()` path.
