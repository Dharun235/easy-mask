# Ideas for the future

Things that could make easy-mask more useful. Nothing here is promised. Each idea says
what it is, why it matters, and roughly how big it is (**S** a day or two, **M** a week,
**L** several weeks). Pick one, open an issue, and move it to the [roadmap](roadmap.md)
when work starts.

The guiding rule stays the same: easy for people who don't code. A new idea should add
few buttons, or none.

## 1. Getting started without code

- **Double-click app** (**L**): a downloadable app for macOS, Windows and Linux
  (PyInstaller or Briefcase), so nobody needs git, uv or a terminal. This is the biggest
  gap: today the "no-code" tool needs three commands to install.
- **First-run tour** (**S**): three short pop-ups on the first launch (open, click an
  object, export).
- **Sample video built in** (**S**): a "Try it" button on the start page that opens a
  short demo clip, so people can learn before using their own data.

## 2. Working through videos

- **Frame timeline** (**M**): a strip under the image with one cell per frame, coloured
  by state (no mask, propagated, hand-edited, reviewed); click to jump, drag to pick a
  frame range for Propagate or Delete.
- **Play with masks** (**M**): play / pause / scrub the video with the masks drawn on, to
  spot where tracking drifted.
- **"Check these frames"** (**M**): use the tracking model's confidence (and sudden jumps in
  mask area or position) to flag doubtful frames; Next jumps to the next doubtful one.
- **Watch propagation live** (**S**): show each frame as it is tracked, and let the user
  stop as soon as it goes wrong.
- **Re-propagate from a fix automatically** (**S**): after a hand fix, offer to redo just the
  frames after it.
- **Object list across the video** (**M**): every object with the frames it appears in,
  so a lost object is easy to find.

## 3. Faster labelling

- **Text prompts** (**L**): type "wing" or "dog" and a model (Grounding-DINO, SAM 3)
  pre-labels every frame; the user only fixes mistakes.
- **More AI models** (**M**): a bigger SAM / SAM 2 image model for hard edges, chosen in a
  simple Fast / Accurate switch rather than a model list.
- **Learn from my labels** (**L**): fine-tune a small model on what the user has labelled
  so far and pre-label the rest.
- **Smart edge snapping** (**M**): polygon points snap to strong edges while drawing.
- **Point tracking** (**M**): track hand-placed points (not just mask centroids and
  corners) through frames with a fast point tracker.

## 4. Measurements and export

- **Measurements panel** (**M**): area, length, angle and centroid speed per object over
  time, with a small plot and CSV export; useful for research like wing or limb motion.
- **Custom key points** (**M**): let the user define their own named points (tip, base,
  joint) per label, kept in order on every frame.
- **More formats** (**S** each): Pascal VOC export; COCO and YOLO import, so existing
  datasets can be fixed in easy-mask.
- **Export video with overlays** (**S**): an mp4 of the frames with masks and points
  drawn on, for slides and papers.

## 5. Ease of use

- **Usability tests** (**S**, ongoing): give the app to one or two people who have never
  seen it, give them one task, and note where they hesitate. Fix the top three problems.
- **Optional expert mode** (**S**): show all shortcuts and the hidden tools for people who
  label all day, without cluttering the default view.
- **Language support** (**M**): translatable UI text.
- **Accessibility** (**S**): colour-blind-safe label colours, larger UI option.

## 6. Bigger projects

- **Share with a team** (**L**): a shared folder mode where several people label different
  frames and a reviewer accepts or rejects them. Only if there is real demand; CVAT and
  Label Studio already do this well.
- **3D and microscopy stacks** (**L**): treat a z-stack like a video and propagate through
  slices.
- **Plugin system** (**L**): let others add models or exporters without changing the core.

## Checking the tool is worth it

Before building big features, measure: time one real clip (about 200 frames) in
easy-mask, X-AnyLabeling and CVAT, counting minutes, clicks per frame and final quality.
Build what closes the biggest gap.
