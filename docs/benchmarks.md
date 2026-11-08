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

## Plan

1. **v0.1 (now):** unit and GUI tests on synthetic images (fast, no downloads, run in CI).
2. **AI branch (done):** `benchmarks/clicks.py` runs the simulated-click protocol on
   GrabCut + Berkeley + DAVIS (results above). Next: SBD / COCO-MVal and boundary IoU.
3. **v0.2+:** timed manual sessions on a fixed 20-image COCO val subset to track
   seconds-per-object as the UI changes.

Check each dataset's licence before use; several are research-only.
