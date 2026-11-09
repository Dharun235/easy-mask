<div align="center">

# easy-mask

**Draw object masks on images and videos. No code, no cloud.**

[![CI](https://github.com/Dharun235/easy-mask/actions/workflows/ci.yml/badge.svg)](https://github.com/Dharun235/easy-mask/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

![easy-mask main window](docs/screenshot.png)

</div>

Click an object and AI outlines it, or draw and paint masks yourself. Track masks through a
video, fix any frame, and export masks, points and datasets. Everything runs on your
computer: no upload, no account.

## Install (once)

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
git clone https://github.com/Dharun235/easy-mask.git
cd easy-mask
SAM2_BUILD_CUDA=0 uv sync --extra video
```

Leave out `--extra video` for a lighter install (~1 GB less) without AI video tracking.

## Start

```bash
uv run easy-mask
```

That's the only command. Everything else is buttons.

## How to use

1. **Open** a folder of images, or **Video** to open a video.
2. Type a label in the **Labels** panel and press Enter.
3. **AI**: click an object, double-click to keep it. Or draw it with **Polygon**, **Box**
   or **Paint**. **Eraser** removes parts.
4. **Propagate** (videos): right-click a label > **Propagate Forward**. AI tracks it through
   the next frames. Fix a bad frame and propagate again from there.
5. **Points**: the centre and 4 corners of each mask, which you can drag.
6. **Export**: pick what to save (masks, cutouts, points, COCO, YOLO, ...) and which frames.

Work saves automatically. **Undo** covers every frame. The bar at the bottom tells you how
to use the tool you picked; right-click anything for more.

AI models download once on first use (40-70 MB), then everything works offline.

## More

- [Command line](docs/command-line.md) for scripts, and the annotation file format
  (labelme-compatible JSON)
- [Benchmarks](docs/benchmarks.md) · [Architecture](docs/architecture.md) ·
  [Roadmap](docs/roadmap.md) · [Contributing](CONTRIBUTING.md)

## License

[Apache-2.0](LICENSE). Uses [EfficientSAM](https://github.com/yformer/EfficientSAM),
[EfficientTAM](https://github.com/yformer/EfficientTAM) and
[SAM 2](https://github.com/facebookresearch/sam2) (all Apache-2.0); cite them if you use
easy-mask in research ([CITATION.cff](CITATION.cff)).
