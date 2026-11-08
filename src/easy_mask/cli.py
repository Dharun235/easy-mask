"""Command-line entry point.

Two forms::

    easy-mask [PATH] [--output DIR] [--labels ...] [--flags ...] [--autosave]   # open the GUI
    easy-mask export IMAGE_DIR --format {masks,coco,yolo,davis} --out DIR    # headless export
"""

import argparse
import sys
from pathlib import Path

from easy_mask import __version__
from easy_mask.core.labels import parse_name_list

EXPORT_FORMATS = ("masks", "coco", "yolo", "davis")


def build_gui_parser() -> argparse.ArgumentParser:
    """Parser for launching the annotation GUI."""
    parser = argparse.ArgumentParser(
        prog="easy-mask",
        description="Annotate images with masks. Run `easy-mask export -h` for dataset export.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("path", nargs="?", type=Path, help="image file or folder to open")
    parser.add_argument(
        "--output", type=Path, help="folder for annotation JSON (default: next to images)"
    )
    parser.add_argument("--labels", help="comma-separated labels or a text file with one per line")
    parser.add_argument("--flags", help="image-level flags: comma-separated or a text file")
    parser.add_argument(
        "--autosave", action="store_true", help="save automatically when switching images"
    )
    parser.add_argument(
        "--frame-step", type=int, default=1, help="when opening a video: keep every n-th frame"
    )
    parser.add_argument(
        "--max-frames", type=int, help="when opening a video: extract at most this many frames"
    )
    parser.add_argument(
        "--video-model",
        choices=("efficienttam_ti_512", "sam2.1_hiera_tiny"),
        default="efficienttam_ti_512",
        help="mask propagation model (needs the video extra): fast default, or SAM 2.1 for "
        "~3 points better J&F at ~5x the time",
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
    return parser


def run_export(args: argparse.Namespace) -> int:
    """Run a headless export. Returns a process exit code."""
    # Imported here so `easy-mask export` works without loading Qt.
    from easy_mask.io import export

    items = export.collect(args.image_dir, args.annotations)
    if not items:
        print(f"No annotated images found in {args.image_dir}", file=sys.stderr)
        return 1
    export.EXPORTERS[args.format](items, args.out, parse_name_list(args.labels) or None)
    print(f"Exported {len(items)} images as {args.format} to {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the GUI or an export. Returns a process exit code."""
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "export":
        return run_export(build_export_parser().parse_args(argv[1:]))

    args = build_gui_parser().parse_args(argv)
    from easy_mask.gui.app import run
    from easy_mask.gui.main_window import Settings

    settings = Settings(
        output_dir=args.output,
        labels=parse_name_list(args.labels),
        flags=parse_name_list(args.flags),
        autosave=args.autosave,
        frame_step=args.frame_step,
        max_frames=args.max_frames,
        video_model=args.video_model,
    )
    return run(args.path, settings)
