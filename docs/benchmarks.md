# Benchmarks and test data

How we will measure whether easy-mask (and especially its SAM2 click-to-segment tool) is
good, using the same datasets and metrics as the interactive-segmentation literature.
Toy images are fine for unit tests; quality claims must come from the data below.

## Metrics

| Metric | Meaning | Lower/higher is better |
| --- | --- | --- |
| **NoC@85 / NoC@90** | Mean number of clicks needed to reach 85% / 90% IoU with the ground-truth mask (capped at 20). The standard interactive-segmentation score. | lower |
| **NoF@90** | Number of objects that never reach 90% IoU within 20 clicks. | lower |
| **mIoU@k** | Mean IoU after exactly *k* clicks (k = 1, 3, 5). SAM reports mIoU@1 with a single point. | higher |
| **Boundary IoU** | IoU restricted to a thin band around the object edge; catches sloppy outlines that plain IoU forgives. | higher |
| **Seconds per object** | Wall-clock annotation time in the real app, including the human. The number users actually feel. | lower |

Clicks are simulated the usual way: the first click goes to the point farthest from the
object boundary; each next click goes to the centre of the largest error region
(positive if it is a missed foreground area, negative if it is false foreground).

## Interactive segmentation benchmarks

The standard suite used by RITM, SimpleClick, SAM and follow-up work:

| Dataset | Size | Why it is used |
| --- | --- | --- |
| [GrabCut](https://www.microsoft.com/en-us/research/project/image-understanding/) (Rother et al., 2004) | 50 images, 1 object each | Classic, tiny; good first sanity check |
| [Berkeley](https://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/) (McGuinness & O'Connor, 2010) | 96 images, 100 objects | Harder boundaries than GrabCut |
| [DAVIS](https://davischallenge.org/) (Perazzi et al., 2016), interactive subset | 345 frames | High-quality masks, thin structures |
| [SBD](http://home.bharathh.info/pubs/codes/SBD/download.html) (Hariharan et al., 2011) | 2,820 val images | Large, many instances per image |
| [Pascal VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) val | 1,449 images | Common baseline |
| COCO-MVal (from [COCO](https://cocodataset.org/)) | 800 objects, 10 per class | Class-balanced COCO sample |

Ready-made downloads and an evaluation protocol: the
[SimpleClick repository](https://github.com/uncbiag/SimpleClick) (MIT) links every
dataset above in the format RITM introduced
([RITM paper](https://arxiv.org/abs/2102.06583)).

## Large-scale and video data

| Dataset | Use for easy-mask |
| --- | --- |
| [COCO 2017](https://cocodataset.org/) val (5k images, polygon masks) | Real-world manual-annotation tests; round-trip our COCO export against the official GT |
| [LVIS](https://www.lvisdataset.org/) | Long-tail categories, more precise masks than COCO |
| [SA-1B](https://ai.meta.com/datasets/segment-anything/) | The data SAM was trained on; diverse images for click-to-segment demos |
| [SA-V](https://ai.meta.com/datasets/segment-anything-video/) | SAM2's video dataset; for the v0.4 propagation milestone |
| [DAVIS 2017](https://davischallenge.org/) | Video object segmentation; measure J&F after propagation |

## Results

### Mask fidelity (v0.2)

`uv run python -m benchmarks.mask_fidelity` on every ground-truth mask (2026-11-08):

| dataset | n | stored losslessly | JSON KB / mask | polygon IoU / vertices @ 0.5 px | @ 1 px | @ 2 px | @ 4 px |
|---|---|---|---|---|---|---|---|
| GrabCut | 50 | 100% | 2.2 | 0.986 / 253 | 0.980 / 90 | 0.973 / 51 | 0.958 / 31 |
| Berkeley | 100 | 100% | 1.6 | 0.987 / 224 | 0.978 / 86 | 0.965 / 48 | 0.941 / 29 |
| DAVIS | 345 | 100% | 2.5 | 0.962 / 372 | 0.950 / 133 | 0.936 / 80 | 0.909 / 48 |

Mask shapes keep every pixel. Polygons lose 1-9% IoU (holes and thin parts), more on
DAVIS's detailed objects: prefer masks when edges matter, polygons when files must be small.

### Classic pixel tools (v0.3)

`uv run python -m benchmarks.classic_tools` (2026-11-08, Apple Silicon laptop, single
thread). Each tool gets one prompt derived from the ground truth: GrabCut the tight
bounding box, Otsu (gray) the same box (the class owning less of the box border is the
object), magic wand (tolerance 32) one click at the interior point farthest from the edge.
Cells are mean IoU / milliseconds per image.

| dataset | n | GrabCut (box) | Otsu (box) | Magic wand (click) |
|---|---|---|---|---|
| GrabCut | 50 | 0.848 / 1491 | 0.543 / 4 | 0.254 / 3 |
| Berkeley | 100 | 0.756 / 1741 | 0.432 / 2 | 0.239 / 2 |
| DAVIS | 345 | 0.661 / 2307 | 0.374 / 4 | 0.168 / 3 |

GrabCut from a box is a usable first pass (refine with brush / eraser or Shift/Alt boxes);
one-shot threshold and wand clicks are only good on high-contrast or flat-coloured objects
and are meant to be combined with Shift+click / Alt+click and clean-up. These are the
baselines the AI click tool (v0.4) has to beat.

## Plan

1. **v0.1 (now):** unit and GUI tests on synthetic images (fast, no downloads, run in CI).
2. **AI branch:** add `benchmarks/clicks.py` that runs the simulated-click protocol
   on GrabCut + Berkeley + DAVIS and reports NoC@85/90 and mIoU@1. Datasets live outside
   the repo (`data/` is gitignored); the script takes a path.
3. **v0.2+:** timed manual sessions on a fixed 20-image COCO val subset to track
   seconds-per-object as the UI changes.

Check each dataset's licence before use; several are research-only.
