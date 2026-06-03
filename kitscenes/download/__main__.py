"""CLI entry point: ``python -m kitscenes.download``.

Examples::

    # Download the full train split
    python -m kitscenes.download /data/kitscenes --split train

    # Limit to the first 50 GB of the validation split
    python -m kitscenes.download /data/kitscenes --split val --max-gb 50

    # Download specific scenes
    python -m kitscenes.download /data/kitscenes \\
        --scenes 008fba36-5e82-e02b-8edf-a55f5271758d \\
                 01ab4321-dead-beef-cafe-123456789abc

    # Preview without downloading
    python -m kitscenes.download /data/kitscenes --split train --dry-run
"""

from __future__ import annotations

import argparse
import sys

from kitscenes.download.downloader import KITScenesDownloader, _DEFAULT_REPO_ID


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m kitscenes.download",
        description=(
            "Download KITScenes scenes from HuggingFace Hub.\n\n"
            "Tars are verified via SHA-256, extracted in place, and deleted afterwards."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "output_dir",
        metavar="OUTPUT_DIR",
        help="Directory where extracted scene folders will be placed.",
    )

    # --- Scene selection ---
    selection = parser.add_argument_group("scene selection (mutually exclusive)")
    sel_grp = selection.add_mutually_exclusive_group()
    sel_grp.add_argument(
        "--split",
        choices=["train", "val", "test", "test_e2e", "overlap_train_val"],
        metavar="SPLIT",
        help=(
            "Download all scenes from this split "
            "(train / val / test / test_e2e / overlap_train_val)."
        ),
    )
    sel_grp.add_argument(
        "--scenes",
        nargs="+",
        metavar="SCENE_ID",
        help="One or more explicit scene IDs to download.",
    )

    parser.add_argument(
        "--max-gb",
        type=float,
        metavar="GB",
        help=(
            "Cap the total download at this many GB. "
            "Scenes are taken in sorted order until the cap is reached."
        ),
    )
    parser.add_argument(
        "--include-existing",
        action="store_true",
        help=(
            "Re-download and re-extract scenes whose output folder already exists. "
            "By default already-extracted scenes are skipped."
        ),
    )

    # --- Output / behaviour ---
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without actually downloading anything.",
    )
    parser.add_argument(
        "--extract-local",
        action="store_true",
        help=(
            "Extract scene tars already under OUTPUT_DIR (after hf download / "
            "git-xet). Skips HuggingFace tar downloads."
        ),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        metavar="N",
        help="Parallel extract workers for --extract-local (default: 1).",
    )
    parser.add_argument(
        "--keep-tars",
        action="store_true",
        help="Keep .tar files after --extract-local (default: delete).",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip SHA-256 verification during --extract-local.",
    )

    # --- HuggingFace ---
    hf = parser.add_argument_group("HuggingFace options")
    hf.add_argument(
        "--token",
        default=None,
        metavar="TOKEN",
        help=(
            "HuggingFace access token. "
            "Defaults to the token saved by ``huggingface-cli login``."
        ),
    )
    hf.add_argument(
        "--repo-id",
        default=_DEFAULT_REPO_ID,
        metavar="REPO_ID",
        help=f"HuggingFace repo ID (default: {_DEFAULT_REPO_ID}).",
    )
    hf.add_argument(
        "--staging-dir",
        default=None,
        metavar="DIR",
        help=(
            "Directory for temporary tar downloads before extraction. "
            "Defaults to OUTPUT_DIR/.kitscenes_staging or "
            "$KITSCENES_DOWNLOAD_STAGING. Use when /tmp is too small."
        ),
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    downloader = KITScenesDownloader(
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        token=args.token,
        staging_dir=args.staging_dir,
    )

    scene_ids = downloader.select_scenes(
        split=args.split,
        scene_ids=args.scenes,
        max_gb=args.max_gb,
        skip_existing=not args.include_existing,
    )

    if args.extract_local:
        downloader.extract_local(
            scene_ids,
            jobs=args.jobs,
            verify=not args.no_verify,
            delete_tars=not args.keep_tars,
            dry_run=args.dry_run,
        )
    else:
        downloader.download(scene_ids, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
