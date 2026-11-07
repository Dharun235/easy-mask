"""Command-line entry point for easy-mask."""

import argparse

from easy_mask import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(prog="easy-mask", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run. Returns a process exit code."""
    build_parser().parse_args(argv)
    return 0
