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
| [SA-V](https://ai.meta.com/datasets/segment-anything-video/) | SAM2's video dataset; a larger propagation test set |
| [DAVIS 2017](https://davischallenge.org/) (CC BY 4.0) | Video object segmentation; J&F after propagation. `DAVIS-2017-trainval-480p.zip` (833 MB) unzipped to `data/benchmarks/DAVIS2017/` |

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

### Video propagation (v0.5)

`uv run python -m benchmarks.video` on the first five DAVIS 2017 val sequences, 480p
(bike-packing, blackswan, bmx-trees, breakdance, camel; 373 frames, 7 objects), 2026-11-08.
Machine: MacBook Air M2 (8 cores, 16 GB), macOS 26.5, Python 3.12, PyTorch 2.15.
J = region IoU, F = boundary F-measure (DAVIS definitions; see the script docstring), each
averaged per object, then over objects. Seconds per frame include frame loading but not
the one-off model load (2-5 s).

**Propagation from the first frame's ground truth** (model sees frame 0 only):

| model | frames | device | J | F | J&F | s / frame |
|---|---|---|---|---|---|---|
| EfficientTAM-Ti 512 (default) | all 373 | CPU | 0.838 | 0.934 | 0.885 | 1.48 |
| EfficientTAM-Ti 512 (default) | all 373 | MPS | 0.838 | 0.934 | 0.885 | 0.34 |
| EfficientTAM-Ti 512 | first 30 per sequence | CPU | 0.875 | 0.953 | 0.914 | 1.51 |
| SAM 2.1 Hiera-Tiny | first 30 per sequence | CPU | 0.914 | 0.974 | 0.945 | 7.38 |

Per sequence (EfficientTAM, CPU, all frames):

| sequence | frames | J | F | J&F | s / frame |
|---|---|---|---|---|---|
| bike-packing | 69 | 0.778 | 0.860 | 0.819 | 2.14 |
| blackswan | 50 | 0.958 | 0.991 | 0.974 | 0.41 |
| bmx-trees | 80 | 0.606 | 0.915 | 0.760 | 1.71 |
| breakdance | 84 | 0.893 | 0.919 | 0.906 | 1.94 |
| camel | 90 | 0.953 | 0.983 | 0.968 | 1.22 |

**Keyframe interpolation, no model** (ground truth on frames 0 and 9 of each 10-frame
window, the 8 frames between scored; masks blended by signed distance after centring):

| sequence | objects | J | F | J&F |
|---|---|---|---|---|
| bike-packing | 2 | 0.673 | 0.780 | 0.726 |
| blackswan | 1 | 0.931 | 0.999 | 0.965 |
| bmx-trees | 2 | 0.316 | 0.755 | 0.535 |
| breakdance | 1 | 0.526 | 0.377 | 0.452 |
| camel | 1 | 0.865 | 0.896 | 0.880 |
| **mean** | | 0.662 | 0.761 | 0.712 |

Takeaways: EfficientTAM is the default because it is ~5x faster than SAM 2.1 on CPU
(and ~4x faster again on Apple MPS, where SAM 2.1 barely speeds up: 8.7 s/frame in a
separate run) for about 3 points of J&F. Pick SAM 2.1 with `--video-model
sam2.1_hiera_tiny` when quality matters more than time. Time grows with the number of
objects (bike-packing, 2 objects, is the slowest). Interpolation is free and good for
rigid or slow motion (blackswan, camel) but fails on articulated motion and occlusion
(breakdance, bmx-trees): propagate those instead, and use interpolation for short gaps.

## Plan

1. **v0.1 (now):** unit and GUI tests on synthetic images (fast, no downloads, run in CI).
2. **AI branch:** add `benchmarks/clicks.py` that runs the simulated-click protocol
   on GrabCut + Berkeley + DAVIS and reports NoC@85/90 and mIoU@1. Datasets live outside
   the repo (`data/` is gitignored); the script takes a path.
3. **v0.2+:** timed manual sessions on a fixed 20-image COCO val subset to track
   seconds-per-object as the UI changes.

Check each dataset's licence before use; several are research-only.
