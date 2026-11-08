<div align="center">

# easy-mask

**Local-first image annotation for computer-vision masks.**

[![CI](https://github.com/Dharun235/easy-mask/actions/workflows/ci.yml/badge.svg)](https://github.com/Dharun235/easy-mask/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

[Features](#features) · [Install](#install) · [Usage](#usage) · [Roadmap / TODO](#roadmap--todo) ·
[Related projects](#related-projects) · [Contributing](#contributing)

![easy-mask main window](docs/screenshot.png)

</div>

Draw polygons, rectangles, circles, lines and points around the objects *you* care about,
then export pixel masks and datasets ready for training. Everything runs on your machine:
no cloud upload, no account, no telemetry. **AI Segment** outlines an object from a click
or a box with [EfficientSAM](https://github.com/yformer/EfficientSAM), on the CPU.

> **Status: early alpha (v0.1).** The on-disk format is labelme-compatible, so you can move
> annotations between the two tools.

## Features

| Area | What you get |
| --- | --- |
| Shapes | polygon, rectangle, circle, line, line strip, point, pixel **mask** |
| AI Segment | click an object (right/Alt-click = background, drag = box), live preview, Enter to keep; EfficientSAM on CPU, offline after a one-time download |
| Mask editing | brush and eraser, union / subtract / intersect, fill holes, remove specks, grow / shrink, smooth, mask <-> polygon |
| Pixel tools (no AI) | magic wand, GrabCut from a box, superpixel picking, watershed from scribbles, threshold (manual / Otsu / adaptive) and colour range with live preview |
| Display | brightness / contrast / gamma, CLAHE, single-channel view, mask opacity (view only, never saved) |
| Editing | select (Shift+click for several), move shapes, drag vertices, relabel (double-click), delete, duplicate, copy/paste across images, undo/redo |
| Labels | **active label** with keys `1`-`9` (no dialog per shape), label dialog with autocomplete, distinct colours, group id, description |
| Automation | track objects into the next frame on **D**, auto clean-up of AI masks, Export Again / export on close, review automatic shapes first |
| Ease of use | start page with recent folders, icon toolbar with shortcut tooltips, `F1` shortcut list, auto-save on, remembers layout / tool sizes / per-folder labels, AI prefetch of upcoming images |
| Navigation | open image or folder, file list with done-marks, next/prev, zoom at cursor, pan, fit |
| Image flags | per-image checkboxes for classification or data cleaning (`--flags`) |
| Saving | one JSON per image next to the image or in `--output`, optional auto-save, crash-safe writes |
| Export | semantic mask PNG, instance mask PNG, COCO instance JSON (polygons or exact RLE), YOLO-seg TXT, DAVIS-style video masks (GUI or CLI) |
| Workflow | review mode (step and zoom through every object, `reviewed` flag), copy shapes from previous image, folder statistics, strict label list |
| Video | open mp4/avi/mov (frames extracted, `--frame-step`, `--max-frames`) or a frame folder; keyframe interpolation; mask propagation with memory (EfficientTAM / SAM 2.1, optional extra) |
| Privacy | 100% offline; images never leave your disk |

## Install

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+ (uv fetches Python for you).

```bash
git clone https://github.com/Dharun235/easy-mask.git
cd easy-mask
uv sync
```

The first time you use AI Segment it downloads the EfficientSAM ViT-T model (41 MB) to
`~/.cache/easy-mask/models` (override with `EASY_MASK_MODELS`), checks its sha256, and
then works offline. `--ai-model efficientsam-vits` picks the larger, slower model (106 MB).

### Optional: video mask propagation

AI propagation needs PyTorch and the EfficientTAM / SAM 2.1 model code (~1 GB with
PyTorch; more on Linux, where PyPI's PyTorch includes CUDA). It is an extra so the base
app stays light. Without it, **Propagate still works by copying the mask unchanged** to
each frame (choose this explicitly with `--video-model copy`):

```bash
SAM2_BUILD_CUDA=0 uv sync --extra video
```

Weights download on first use to `~/.cache/easy-mask/checkpoints` (EfficientTAM-Ti 72 MB;
SAM 2.1 Hiera-Tiny 156 MB if selected). The device is picked automatically: CUDA, then
Apple MPS, then CPU. On a laptop CPU expect roughly 0.3-2 s per frame with EfficientTAM
(more objects, more time) and 5-10x that with SAM 2.1; see
[docs/benchmarks.md](docs/benchmarks.md#video-propagation-v05).
Pick SAM 2.1 (better quality, slower) with `--video-model sam2.1_hiera_tiny`. Both models'
code and weights are Apache-2.0 ([EfficientTAM](https://github.com/yformer/EfficientTAM),
[SAM 2](https://github.com/facebookresearch/sam2)).

## Quick start: annotate fast

1. `uv run easy-mask` opens a start page: **Open Folder**, or pick a recent folder.
2. Press **1-9** to pick a label (or click it in the Labels panel). New shapes take it
   with no dialog; **0** goes back to asking each time.
3. Press **I** for AI Segment and click an object; **Enter** keeps it. Images ahead are
   prepared in the background, so the next one is instant.
4. Press **D** for the next image. Work is saved automatically.
5. **Video / similar frames:** turn on **Video > Track Objects into Next Frame**
   (`Ctrl+Shift+T`) and every **D** carries the selected (or all) objects forward: AI
   tracking with the video extra, a plain copy without it. **N** then reviews the
   automatic shapes first.
6. Export once (File > Export Folder As); afterwards **Ctrl+Shift+E** repeats it, or tick
   *Export Again When Closing* to keep the dataset current automatically.
7. **F1** lists every shortcut.

AI and smart-tool masks are tidied automatically (specks and pinholes removed; Mask > Auto
Clean-Up). AI Segment needs no extra install: the model downloads on first use.

easy-mask remembers each folder's labels, flags, active label and annotation folder (in
`.easy-mask.json` next to the images), plus window layout, brush sizes and recent folders.

## Usage

```bash
uv run easy-mask                              # start page; open from there or the File menu
uv run easy-mask photo.jpg                    # annotate one image
uv run easy-mask images/ --output labels/     # annotate a folder, JSON goes to labels/
uv run easy-mask images/ --labels cat,dog     # predefine labels (or --labels labels.txt)
uv run easy-mask images/ --labels cat,dog --strict-labels   # refuse any other label
uv run easy-mask images/ --flags blurry,occluded --no-autosave   # ask before leaving an image

# export a folder of annotations to a training dataset
uv run easy-mask export images/ --format coco  --out dataset/
uv run easy-mask export images/ --format masks --out dataset/   # semantic + instance PNGs
uv run easy-mask export images/ --format yolo  --out dataset/
uv run easy-mask export images/ --format coco --rle --out dataset/  # masks as exact COCO RLE

# progress and label check: per-label objects/area, annotated/reviewed counts
uv run easy-mask stats images/                     # plain table (or --json)
uv run easy-mask stats images/ --labels cat,dog    # exit 1 if other labels are used

# video: frames are extracted next to the video (clip_frames/), one JSON per frame
uv run easy-mask clip.mp4 --frame-step 2 --max-frames 300
uv run easy-mask export clip_frames/ --format davis --out davis/   # indexed PNG per frame
```

### Video workflow

1. Open a video (or a folder of frames). Draw or brush the objects on one frame. Give
   objects that share a label different **group ids**: `(label, group id)` is an object's
   identity across frames.
2. Optionally select the objects to track (Shift+click for several; nothing selected =
   every object). **Video > Propagate Forward / Backward** (`Ctrl+Shift+Right` / `Left`)
   tracks them through the next frames on a background thread (AI with the video extra,
   plain copy without it); **Video > Stop** cancels. Results are
   mask shapes flagged `propagated` and shown as `(auto)` in the shape list.
3. Fix a bad frame by hand and propagate again from there: automatic shapes are
   replaced, hand-drawn ones are never overwritten and also steer the model.
4. Without the model: annotate two keyframes and use **Video > Interpolate From Previous
   Keyframe** (`Ctrl+Shift+I`) on the later one. Polygons and boxes morph vertex by
   vertex, masks through signed distance fields; results are flagged `interpolated`.

### Keyboard shortcuts

| Key | Action | Key | Action |
| --- | --- | --- | --- |
| `1`-`9` | use label 1-9 for new shapes | `0` | ask for the label each time |
| `F1` / `?` | list all shortcuts | `Ctrl+Shift+E` | export again (last format and folder) |
| `Ctrl+Shift+T` | track objects into next frame (toggle) | | |
| `P` | polygon | `E` | edit (select) mode |
| `R` | rectangle | `Enter` / double-click | finish polygon / strip |
| `C` | circle | `Backspace` | remove last point while drawing, else delete shape |
| `L` | line | `Esc` | cancel drawing / deselect |
| `T` | line strip | `Ctrl+D` | duplicate selected shape |
| `O` | point | `Ctrl+Z` / `Ctrl+Shift+Z` | undo / redo |
| `D` / `A` | next / previous image | `Ctrl+S` | save |
| `Ctrl+wheel` | zoom at cursor | wheel / middle-drag | pan |
| `F` | fit image to window | `Ctrl+C` / `Ctrl+V` | copy / paste shape |
| `B` / `X` | brush / eraser | `[` / `]` | smaller / bigger brush (or wand tolerance, superpixel / marker size) |
| `W` | magic wand (Shift add, Alt subtract) | `G` | GrabCut: drag a box (Shift add, Alt subtract) |
| `S` | superpixels: click / drag to pick | `M` | watershed: drag object markers, Alt+drag background, `Enter` runs |
| `Ctrl+T` | threshold / colour-range panel | View > Adjust Display | brightness, CLAHE, channel, mask opacity |
| `N` / `Shift+N` | review next / previous object | `Ctrl+Shift+V` | copy shapes from previous image |
| `Ctrl+I` | folder statistics | | |
| `I` | AI Segment | click / right-click / drag | object point / background point / box |
| `Enter` / double-click (AI) | keep the AI mask | `Esc` / `Backspace` (AI) | clear prompts / undo last prompt |

Rectangles, circles and lines can be drawn by dragging or by two clicks. AI Segment adds the
mask to the selected shape if one is selected, otherwise it creates a new mask and asks
for its label.

### Annotation format

One JSON per image, compatible with [labelme](https://github.com/wkentaro/labelme):

```json
{
  "version": "0.1.0",
  "imagePath": "photo.jpg",
  "imageHeight": 480,
  "imageWidth": 640,
  "flags": {"blurry": false},
  "shapes": [
    {"label": "cat", "shape_type": "polygon", "points": [[10, 10], [50, 12], [30, 60]],
     "group_id": null, "description": "", "flags": {}}
  ]
}
```

Rectangles store two opposite corners; circles store the centre and one point on the rim.

## Roadmap / TODO

Full plan with milestones: [docs/roadmap.md](docs/roadmap.md).

- [x] **v0.1 Core annotation:** all manual shape tools, editing, undo/redo, labelme JSON, mask/COCO/YOLO export
- [x] **v0.2 Mask shapes:** pixel masks, brush/eraser, boolean combine, clean-up, mask <-> polygon
- [x] **v0.4 AI click-to-segment on CPU:** EfficientSAM, positive/negative clicks and box prompts, live preview
- [x] **v0.4 Click benchmark:** NoC@85/90 and mIoU on GrabCut, Berkeley, DAVIS ([docs/benchmarks.md](docs/benchmarks.md))
- [x] **v0.5 Video:** open videos, keyframe interpolation, mask propagation (EfficientTAM / SAM 2.1), DAVIS export
- [ ] Text-prompted pre-labelling (SAM3 / Grounding-DINO style)
- [ ] Brightness / contrast, keep previous annotations, COCO import, Pascal VOC export
- [ ] Standalone app bundles (no Python needed)

Have an idea? [Open a feature request](https://github.com/Dharun235/easy-mask/issues/new/choose).

## Benchmarks

Quality is measured on the standard interactive-segmentation suite (GrabCut, Berkeley,
DAVIS, SBD, Pascal VOC, COCO-MVal) with NoC@85/90 and mIoU@k, plus timed real-world
sessions on COCO val. Click-to-segment results (NoC@85/90, mIoU@k, CPU timing) for
EfficientSAM, plus datasets, metrics and protocol, are in [docs/benchmarks.md](docs/benchmarks.md).

## Related projects

easy-mask stands on the shoulders of these tools and models. If you need something it
doesn't do yet, one of them probably does.

| Project | What it is | How easy-mask relates |
| --- | --- | --- |
| [labelme](https://github.com/wkentaro/labelme) | The classic Python/Qt polygon annotator (GPL-3.0) | Same JSON format; feature reference for v0.1 |
| [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling) / [AnyLabeling](https://github.com/vietanhdev/anylabeling) | labelme-style desktop apps with many AI models (GPL-3.0) | Inspiration for AI-assisted labelling |
| [CVAT](https://github.com/cvat-ai/cvat) | Full web platform for teams, video, tracking (MIT) | Heavier, server-based; easy-mask stays a single local app |
| [Label Studio](https://github.com/HumanSignal/label-studio) | Multi-modal web labelling platform (Apache-2.0) | Broader scope; easy-mask focuses on image masks |
| [LabelImg](https://github.com/HumanSignal/labelImg) | Bounding-box annotator, now archived (MIT) | Boxes only; easy-mask covers boxes and masks |
| [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything) | Promptable segmentation model (Apache-2.0) | Model family behind click-to-segment |
| [EfficientSAM](https://github.com/yformer/EfficientSAM) | Lightweight SAM (Apache-2.0); ONNX export by the labelme project | AI Segment backend |
| [SAM 2](https://github.com/facebookresearch/sam2) | SAM for images and video (Apache-2.0) | Optional video propagation backend (SAM 2.1 Hiera-Tiny) |
| [EfficientTAM](https://github.com/yformer/EfficientTAM) | Efficient Track Anything: SAM 2's video API with a light ViT (Apache-2.0) | Default video propagation backend |
| [SAM 3](https://github.com/facebookresearch/sam3) | Text/concept-prompted segmentation | Candidate for text pre-labelling |
| [SimpleClick](https://github.com/uncbiag/SimpleClick) | Interactive segmentation research code (MIT) | Benchmark datasets and evaluation protocol |
| [FiftyOne](https://github.com/voxel51/fiftyone) | Dataset curation and visualisation (Apache-2.0) | Good companion for inspecting exported datasets |

easy-mask is an independent implementation: it shares labelme's file format but no code.

## Contributing

Contributions of all sizes are welcome: bug reports, docs, tests, features.

1. Read [CONTRIBUTING.md](CONTRIBUTING.md), [docs/architecture.md](docs/architecture.md) and
   [docs/testing.md](docs/testing.md).
2. Pick an issue (or open one), branch as `feat/<name>` or `fix/<name>`, open a PR.
3. AI coding assistants: follow [CLAUDE.md](CLAUDE.md).

```bash
uv sync                      # installs dev tools too
uv run pre-commit install    # ruff on every commit
uv run pytest --cov          # 140+ tests: unit, property, integration, GUI, smoke (headless)
uv run ruff check . && uv run mypy
```

Please follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Contributors

Thanks to everyone who has contributed!

<a href="https://github.com/Dharun235/easy-mask/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Dharun235/easy-mask" alt="Contributors" />
</a>

| Who | Role |
| --- | --- |
| [@Dharun235](https://github.com/Dharun235) | Creator and maintainer |

Contributed something? Add yourself to the table in your PR.

## Citation

If easy-mask helps your research, please cite it (see [CITATION.cff](CITATION.cff)), and
cite the models you used through it (e.g. SAM 2).

## License

[Apache-2.0](LICENSE) © easy-mask contributors.

Model weights are not part of easy-mask: AI Segment downloads EfficientSAM (Xiong et al.,
2024, Apache-2.0) from the [labelmeai/efficient-sam](https://github.com/labelmeai/efficient-sam)
ONNX release. Cite EfficientSAM if you use it in research.
