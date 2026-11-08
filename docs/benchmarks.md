# Benchmarks and test data

How we measure whether easy-mask (and especially its AI click-to-segment tool) is
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

### COCO export: RLE vs polygons (v0.6)

`uv run python -m benchmarks.coco_rle`: each ground-truth mask is stored as a mask shape,
exported with `easy-mask export --format coco` (polygons traced at 1 px) and with `--rle`,
then decoded back to pixels (2026-11-08). "Compact" is the JSON without indentation.

| dataset | n | format | mean pixel IoU | exact | KB / object | KB / object (compact) |
|---|---|---|---|---|---|---|
| GrabCut | 50 | polygon | 0.9801 | 0% | 3.3 | 1.4 |
| GrabCut | 50 | RLE | 1.0000 | 100% | 1.3 | 1.2 |
| Berkeley | 100 | polygon | 0.9778 | 0% | 3.2 | 1.3 |
| Berkeley | 100 | RLE | 1.0000 | 100% | 1.2 | 1.1 |
| DAVIS | 345 | polygon | 0.9503 | 0% | 4.8 | 2.0 |
| DAVIS | 345 | RLE | 1.0000 | 100% | 2.0 | 1.9 |

RLE round-trips every mask exactly and is also smaller than polygons on all three
datasets, so `--rle` is the better choice for mask annotations whenever the training
code reads COCO RLE (anything built on pycocotools, e.g. Detectron2 and MMDetection).

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

### Click-to-segment: EfficientSAM ViT-T (v0.4)

`uv run python -m benchmarks.clicks --model efficientsam-vitt` on every image (2026-11-08),
simulated clicks as described above, at most 20, stopping once IoU >= 90% after at least
5 clicks. The app's own logic is used: at most 6 prompt points (most recent kept), lone
click -> largest candidate, negative clicks cut regions out (see below). GrabCut's
uncertain boundary band is ignored, as in RITM.

| dataset | n | NoC@85 | NoC@90 | NoF@90 | mIoU@1 | mIoU@3 | mIoU@5 | encode ms | click ms |
|---|---|---|---|---|---|---|---|---|---|
| GrabCut | 50 | 1.52 | 1.76 | 0 | 0.88 | 0.94 | 0.95 | 3320 | 125 |
| Berkeley | 100 | 2.99 | 4.42 | 10 | 0.78 | 0.85 | 0.88 | 2206 | 69 |
| DAVIS | 345 | 4.71 | 6.34 | 73 | 0.70 | 0.84 | 0.85 | 2013 | 60 |

Machine: Apple M2 (4 performance + 4 efficiency cores), 16 GB, macOS 26.5, onnxruntime
1.31.1 CPU execution provider, Python 3.12. **Timings are pessimistic:** other heavy jobs
were running (load average 6-12 on 8 cores); on an idle machine encoding is expected around
0.4-0.8 s. Encode time is per image (once, in the background); click time is per prompt
change and includes upsampling three masks to full image size.

Compare with the tables in the [RITM](https://arxiv.org/abs/2102.06583) and
[SimpleClick](https://github.com/uncbiag/SimpleClick) papers with care: those models run on
a GPU, feed the previous mask back into the network and use the same 20-click protocol;
EfficientSAM here gets only the clicks. Berkeley's fine boundaries are its weak spot
(10 of 100 objects never reach 90%).

**Why the app post-processes the model** (ablation, first 10 images per dataset):

| variant | NoC@90 GrabCut | Berkeley | DAVIS |
|---|---|---|---|
| top-scored mask, negative clicks passed to the model as label 0 | 3.7 | 8.8 | 18.1 |
| + negative clicks cut out the smallest region a click there selects | 1.4 | 3.8 | 4.9 |
| oracle: best of the 3 candidates by true IoU (upper bound for picking) | 3.1 | 6.9 | 2.9 |

EfficientSAM's prompt encoder has no embedding for label 0, and a "negative" point behaves
almost like a positive one (masks grow instead of shrinking), so passing negative clicks
straight through makes every correction worse. For a single click the predicted IoU
prefers small parts; the largest candidate is the whole object far more often
(mIoU@1 on 25 images: 0.89 / 0.76 / 0.60 vs 0.57 / 0.54 / 0.45).

**ViT-S vs ViT-T** (`--limit 20`: first 20 images per dataset only, same machine and load;
the first 20 DAVIS frames are harder than the full set):

| model | dataset | NoC@85 | NoC@90 | NoF@90 | mIoU@1 | mIoU@3 | mIoU@5 | encode ms | click ms |
|---|---|---|---|---|---|---|---|---|---|
| ViT-T | GrabCut | 1.30 | 1.55 | 0 | 0.89 | 0.96 | 0.97 | 2585 | 87 |
| ViT-S | GrabCut | 1.15 | 1.20 | 0 | 0.92 | 0.93 | 0.91 | 5795 | 79 |
| ViT-T | Berkeley | 2.80 | 5.95 | 4 | 0.77 | 0.87 | 0.88 | 3335 | 121 |
| ViT-S | Berkeley | 2.60 | 5.10 | 4 | 0.84 | 0.83 | 0.79 | 6064 | 68 |
| ViT-T | DAVIS | 9.95 | 12.45 | 11 | 0.61 | 0.73 | 0.77 | 2735 | 90 |
| ViT-S | DAVIS | 11.35 | 12.45 | 11 | 0.68 | 0.72 | 0.73 | 5796 | 71 |

ViT-S is better on the first click and about 2x slower to encode, but its corrections
help less (mIoU falls from 3 to 5 clicks), so ViT-T stays the default.

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
2. **AI branch (done):** `benchmarks/clicks.py` runs the simulated-click protocol on
   GrabCut + Berkeley + DAVIS (results above). Next: SBD / COCO-MVal and boundary IoU.
3. **v0.2+:** timed manual sessions on a fixed 20-image COCO val subset to track
   seconds-per-object as the UI changes.

Check each dataset's licence before use; several are research-only.

### AI Segment responsiveness and click filtering (v0.7)

- **Prefetch:** with the model loaded, the current and next two images are encoded in the
  background. Time from "next image" to AI ready, EfficientSAM ViT-T on GrabCut, Apple M2
  CPU, mean of 5 steps: **862 ms -> 8 ms**. The encode itself still takes ~0.85 s; it
  just happens before you get there.
- **Keep clicked regions:** for click-only prompts, blobs that contain no positive click
  are dropped. First 20 images per dataset, ViT-T:

  | dataset | NoC@90 before -> after | mIoU@1 | mIoU@5 |
  |---|---|---|---|
  | GrabCut | 1.55 -> 1.55 | 0.89 -> 0.90 | 0.97 -> 0.97 |
  | Berkeley | 5.95 -> 5.70 | 0.77 -> 0.77 | 0.88 -> 0.88 |
  | DAVIS | 12.45 -> 12.30 | 0.61 -> 0.61 | 0.77 -> 0.79 |

  Neutral on the numbers (DAVIS NoC@85 9.95 -> 10.35: some objects there are in separate
  parts; click each part), while visibly removing stray regions far from the click.

### Auto clean-up of AI masks (v0.7)

`uv run python -m benchmarks.cleanup`: one standard first click per object (EfficientSAM
ViT-T, CPU), IoU before / after the GUI's auto clean-up (remove specks and fill pinholes
below 0.05% of the image, at least 20 px; larger holes are kept):

| dataset | n | IoU raw | IoU cleaned | masks changed | better / worse |
|---|---|---|---|---|---|
| GrabCut | 50 | 0.8731 | 0.8740 | 30/50 | 23 / 7 |
| Berkeley | 100 | 0.7918 | 0.7927 | 62/100 | 36 / 26 |
| DAVIS | 345 | 0.6809 | 0.6819 | 205/345 | 159 / 46 |

A small but consistent gain that mostly removes visual noise (stray specks, pinholes), so
it is on by default (Mask > Auto Clean-Up). The brush and eraser are never cleaned.

### AI Propagate vs Copy from one polygon (v0.8)

Through the GUI: one polygon traced from the frame-0 ground truth of DAVIS 2017 `blackswan`,
then 15 frames of AI Propagate (EfficientTAM-Ti 512, Apple M2 GPU via MPS) or Copy; IoU
against the ground truth of frames 1-15:

| mode | mean IoU | frame 1 | frame 15 | time for 15 frames |
|---|---|---|---|---|
| AI Propagate | 0.965 | 0.970 | 0.963 | 9.4 s (incl. model load) |
| Copy | 0.635 | 0.914 | 0.503 | 0.1 s |

### Cut Through Frames (v0.9)

Through the GUI on DAVIS 2017 `dogs-jump`: object 2 was merged into object 1's mask on
every frame (a typical propagation mistake); one lasso cut around object 2 on frame 0,
then Cut Through Frames over 15 frames. Object-1 IoU against its ground truth, frames 1-15:

| mode | before | after | frame 15 | time |
|---|---|---|---|---|
| AI (EfficientTAM tracks the cut part) | 0.174 | 0.755 | 0.857 | 7.2 s |
| Copy (same pixels every frame) | 0.174 | 0.299 | 0.127 | 0.1 s |

One `Ctrl+Z` afterwards restored all 15 frames exactly (IoU 1.000 with the merged masks).
