"""CLI: ``python -m kitscenes.pack_sqfs``.

Build per-split SquashFS images from scene tars under ``$KITSCENES_ROOT/data/``::

    export KITSCENES_ROOT=/data/kitscenes
    python -m kitscenes.pack_sqfs

Creates ``/data/kitscenes_sqfs/{train,val,...}.sqfs``.  Mount with::

    source scripts/mount_kitscenes_sqfs.sh /data/kitscenes_sqfs
"""

from __future__ import annotations

import argparse
import os

from kitscenes.constants import KITSCENES_ROOT_ENV_VAR
from kitscenes.pack_sqfs.builder import SquashFsBuilder


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m kitscenes.pack_sqfs",
        description=(
            "Build one SquashFS image per split from scene tars "
            "(ratarmount + mksquashfs). Tars are removed after append by default."
        ),
    )
    parser.add_argument(
        "dataset_root",
        nargs="?",
        default=None,
        help=f"Dataset root (default: ${KITSCENES_ROOT_ENV_VAR}).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        metavar="DIR",
        help="Output directory for .sqfs files (default: <parent of root>/kitscenes_sqfs).",
    )
    parser.add_argument(
        "--split",
        action="append",
        dest="splits",
        metavar="SPLIT",
        help="Build only this split (repeatable). Default: all splits with tars.",
    )
    parser.add_argument(
        "--comp",
        default="lz4",
        help=(
            "SquashFS compressor (default: lz4). "
            "Use 'none' for -no-compression."
        ),
    )
    parser.add_argument(
        "--processors",
        type=int,
        default=1,
        metavar="N",
        help="mksquashfs -processors (default: 1).",
    )
    parser.add_argument(
        "--keep-tars",
        action="store_true",
        help="Do not delete .tar files after append.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove existing .sqfs images for selected splits before building.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List tars that would be packed without doing work.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    root = args.dataset_root or os.environ.get(KITSCENES_ROOT_ENV_VAR)
    if root is None:
        raise SystemExit(
            f"Pass dataset_root or set ${KITSCENES_ROOT_ENV_VAR}."
        )

    builder = SquashFsBuilder(
        root,
        output_dir=args.output_dir,
        compression=args.comp,
        processors=args.processors,
    )
    builder.build(
        splits=args.splits,
        delete_tars=not args.keep_tars,
        force=args.force,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
