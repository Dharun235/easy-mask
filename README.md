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
no cloud upload, no account, no telemetry. Click-to-segment with
[SAM2](https://github.com/facebookresearch/sam2) is the next milestone.

> **Status: early alpha (v0.1).** The on-disk format is labelme-compatible, so you can move
> annotations between the two tools.

## Features

| Area | What you get |
| --- | --- |
| Shapes | polygon, rectangle, circle, line, line strip, point, pixel **mask** |
| Mask editing | brush and eraser, union / subtract / intersect, fill holes, remove specks, grow / shrink, smooth, mask <-> polygon |
| Pixel tools (no AI) | magic wand, GrabCut from a box, superpixel picking, watershed from scribbles, threshold (manual / Otsu / adaptive) and colour range with live preview |
| Display | brightness / contrast / gamma, CLAHE, single-channel view, mask opacity (view only, never saved) |
| Editing | select (Shift+click for several), move shapes, drag vertices, relabel (double-click), delete, duplicate, copy/paste across images, undo/redo |
| Labels | label dialog with autocomplete, distinct per-label colours, group id (one object in several parts), description |
| Navigation | open image or folder, file list with done-marks, next/prev, zoom at cursor, pan, fit |
| Image flags | per-image checkboxes for classification or data cleaning (`--flags`) |
| Saving | one JSON per image next to the image or in `--output`, optional auto-save, crash-safe writes |
| Export | semantic mask PNG, instance mask PNG, COCO instance JSON, YOLO-seg TXT (GUI or CLI) |
| Privacy | 100% offline; images never leave your disk |

## Install

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+ (uv fetches Python for you).

```bash
git clone https://github.com/Dharun235/easy-mask.git
cd easy-mask
uv sync
```

## Usage

```bash
uv run easy-mask                              # empty window, open from the File menu
uv run easy-mask photo.jpg                    # annotate one image
uv run easy-mask images/ --output labels/     # annotate a folder, JSON goes to labels/
uv run easy-mask images/ --labels cat,dog     # predefine labels (or --labels labels.txt)
uv run easy-mask images/ --flags blurry,occluded --autosave

# export a folder of annotations to a training dataset
uv run easy-mask export images/ --format coco  --out dataset/
uv run easy-mask export images/ --format masks --out dataset/   # semantic + instance PNGs
uv run easy-mask export images/ --format yolo  --out dataset/
```

### Keyboard shortcuts

| Key | Action | Key | Action |
| --- | --- | --- | --- |
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

Rectangles, circles and lines can be drawn by dragging or by two clicks.

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
- [ ] **v0.2 SAM2 click-to-segment:** positive/negative clicks and box prompts, runs locally on CUDA / Apple MPS / CPU
- [ ] **v0.2 Benchmarks:** NoC@85/90 and mIoU on GrabCut, Berkeley, DAVIS ([docs/benchmarks.md](docs/benchmarks.md))
- [ ] **v0.4 Video:** SAM2 propagation across frames, object tracks
- [ ] Text-prompted pre-labelling (SAM3 / Grounding-DINO style)
- [ ] Brightness / contrast, keep previous annotations, COCO import, Pascal VOC export
- [ ] Standalone app bundles (no Python needed)

Have an idea? [Open a feature request](https://github.com/Dharun235/easy-mask/issues/new/choose).

## Benchmarks

Quality is measured on the standard interactive-segmentation suite (GrabCut, Berkeley,
DAVIS, SBD, Pascal VOC, COCO-MVal) with NoC@85/90 and mIoU@k, plus timed real-world
sessions on COCO val. Results land with the SAM2 milestone; datasets, metrics and protocol
are described in [docs/benchmarks.md](docs/benchmarks.md).

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
| [SAM 2](https://github.com/facebookresearch/sam2) | SAM for images and video (Apache-2.0) | Planned v0.2 backend |
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
