# Command line (optional)

You don't need these to use easy-mask: the app does the same things with buttons. They
are for scripts and batch jobs.

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
uv run easy-mask export images/ --format objects --parts masks,centroid,corners --out objects/
uv run easy-mask export images/ --format coco --rle --out dataset/  # masks as exact COCO RLE

# progress and label check: per-label objects/area, annotated/reviewed counts
uv run easy-mask stats images/                     # plain table (or --json)
uv run easy-mask stats images/ --labels cat,dog    # exit 1 if other labels are used

# video: import frames (choose folder and format), then one JSON per frame
uv run easy-mask clip.mp4                       # asks: folder, PNG/JPEG/WebP, every n-th frame
uv run easy-mask export clip_frames/ --format davis --out davis/   # indexed PNG per frame
```

Options for AI: `--ai-model efficientsam-vits` (larger, slower) and, with the video extra,
`--video-model sam2.1_hiera_tiny` (better, slower) or `--video-model copy` (no AI).

## Annotation format


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
