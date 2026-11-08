"""Command-line entry point.

Three forms::

    easy-mask [PATH] [--output DIR] [--labels ...] [--strict-labels] [--flags ...] [--autosave]
              [--ai-model M] [--video-model M] [--frame-step N] [--max-frames N]
    easy-mask export IMAGE_DIR --format {masks,coco,yolo,davis} [--rle] --out DIR   # headless
    easy-mask stats IMAGE_DIR [--labels ...] [--json]                                # statistics
"""

import argparse
import json
import sys
from pathlib import Path

from easy_mask import __version__
from easy_mask.ai import efficient_sam
from easy_mask.core.labels import parse_name_list

EXPORT_FORMATS = ("masks", "coco", "yolo", "davis")


def build_gui_parser() -> argparse.ArgumentParser:
    """Parser for launching the annotation GUI."""
    parser = argparse.ArgumentParser(
        prog="easy-mask",
        description="Annotate images with masks. "
        "See `easy-mask export -h` (dataset export) and `easy-mask stats -h` (statistics).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("path", nargs="?", type=Path, help="image file or folder to open")
    parser.add_argument(
        "--output", type=Path, help="folder for annotation JSON (default: next to images)"
    )
    parser.add_argument("--labels", help="comma-separated labels or a text file with one per line")
    parser.add_argument(
        "--strict-labels",
        action="store_true",
        help="only accept labels from --labels in the label dialog",
    )
    parser.add_argument("--flags", help="image-level flags: comma-separated or a text file")
    parser.add_argument(
        "--autosave", action="store_true", help="save automatically when switching images"
    )
    parser.add_argument(
        "--ai-model",
        default=efficient_sam.DEFAULT_MODEL,
        choices=sorted(efficient_sam.MODELS),
        help="model for the AI Segment tool (vits: slower, more accurate)",
    )
    parser.add_argument(
        "--frame-step", type=int, default=1, help="when opening a video: keep every n-th frame"
    )
    parser.add_argument(
        "--max-frames", type=int, help="when opening a video: extract at most this many frames"
    )
    parser.add_argument(
        "--video-model",
        choices=("efficienttam_ti_512", "sam2.1_hiera_tiny", "copy"),
        default="efficienttam_ti_512",
        help="mask propagation: EfficientTAM (fast default) or SAM 2.1 (~3 points better J&F, "
        "~5x slower) need the video extra; 'copy' repeats the mask unchanged, no model. "
        "Without the extra, propagation falls back to copy.",
    )
    return parser


def build_export_parser() -> argparse.ArgumentParser:
    """Parser for the ``export`` subcommand."""
    parser = argparse.ArgumentParser(
        prog="easy-mask export", description="Export annotations to a dataset."
    )
    parser.add_argument("image_dir", type=Path, help="folder of annotated images")
    parser.add_argument("--format", required=True, choices=EXPORT_FORMATS, help="dataset format")
    parser.add_argument("--out", required=True, type=Path, help="destination folder")
    parser.add_argument(
        "--annotations", type=Path, help="folder of JSON files (default: image_dir)"
    )
    parser.add_argument(
        "--labels", help="class order: comma-separated or a text file (default: sorted)"
    )
    parser.add_argument(
        "--rle",
        action="store_true",
        help="coco only: write mask shapes as exact COCO RLE instead of traced polygons",
    )
    return parser


def build_stats_parser() -> argparse.ArgumentParser:
    """Parser for the ``stats`` subcommand."""
    parser = argparse.ArgumentParser(
        prog="easy-mask stats",
        description="Count images, objects and per-label totals; optionally validate labels.",
    )
    parser.add_argument("image_dir", type=Path, help="folder of images")
    parser.add_argument(
        "--annotations", type=Path, help="folder of JSON files (default: image_dir)"
    )
    parser.add_argument(
        "--labels",
        help="allowed labels (comma-separated or a text file); exit 1 if others are used",
    )
    parser.add_argument("--json", action="store_true", help="print JSON instead of a table")
    return parser


def run_stats(args: argparse.Namespace) -> int:
    """Print folder statistics. Returns 1 if labels outside ``--labels`` are used."""
    from easy_mask.io import stats

    allowed = parse_name_list(args.labels) or None
    result = stats.compute(args.image_dir, args.annotations, allowed)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(stats.format_table(result, validated=allowed is not None))
    return 1 if result.unknown_labels else 0


def run_export(args: argparse.Namespace) -> int:
    """Run a headless export. Returns a process exit code."""
    # Imported here so `easy-mask export` works without loading Qt.
    from easy_mask.io import export

    if args.rle and args.format != "coco":
        print("--rle only applies to --format coco", file=sys.stderr)
        return 2
    items = export.collect(args.image_dir, args.annotations)
    if not items:
        print(f"No annotated images found in {args.image_dir}", file=sys.stderr)
        return 1
    fmt = "coco-rle" if args.rle else args.format
    export.EXPORTERS[fmt](items, args.out, parse_name_list(args.labels) or None)
    print(f"Exported {len(items)} images as {fmt} to {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the GUI, an export or stats. Returns a process exit code."""
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "export":
        return run_export(build_export_parser().parse_args(argv[1:]))
    if argv and argv[0] == "stats":
        return run_stats(build_stats_parser().parse_args(argv[1:]))

    parser = build_gui_parser()
    args = parser.parse_args(argv)
    if args.strict_labels and not args.labels:
        parser.error("--strict-labels needs --labels")
    from easy_mask.gui.app import run
    from easy_mask.gui.main_window import Settings

    settings = Settings(
        output_dir=args.output,
        labels=parse_name_list(args.labels),
        flags=parse_name_list(args.flags),
        autosave=args.autosave,
        strict_labels=args.strict_labels,
        ai_model=args.ai_model,
        frame_step=args.frame_step,
        max_frames=args.max_frames,
        video_model=args.video_model,
    )
    return run(args.path, settings)
