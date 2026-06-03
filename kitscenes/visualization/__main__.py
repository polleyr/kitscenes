"""CLI entry points for HD map visualization pipelines.

Usage::

    python -m kitscenes.visualization project --base-dir ... --map-path ...
    python -m kitscenes.visualization video --visdir ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from kitscenes.constants import KITSCENES_ROOT_ENV_VAR, KITSCENES_VIZ_OUTPUT_ENV_VAR


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="KITScenes HD map visualization tools",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    project_parser = subparsers.add_parser(
        "project",
        help="Project Lanelet2 map onto camera images (MapProjector)",
    )
    project_parser.add_argument("--base-dir", type=str, required=True)
    project_parser.add_argument("--map-path", type=str, required=True)
    project_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Output directory for projections (required unless "
            f"${KITSCENES_VIZ_OUTPUT_ENV_VAR} is set). Must not lie inside "
            f"${KITSCENES_ROOT_ENV_VAR}."
        ),
    )
    project_parser.add_argument("--config-path", type=str, default=None)
    project_parser.add_argument("--poses-path", type=str, default=None)
    project_parser.add_argument("--timestamp-path", type=str, default=None)
    project_parser.add_argument(
        "--lat-origin",
        type=float,
        default=None,
        help="UTM latitude origin override (default: read maps/origin.json).",
    )
    project_parser.add_argument(
        "--lon-origin",
        type=float,
        default=None,
        help="UTM longitude origin override (default: read maps/origin.json).",
    )
    project_parser.add_argument("--front-only", action="store_true")
    project_parser.add_argument("--frame-step", type=int, default=5)
    project_parser.add_argument("--skip-top-down", action="store_true")
    project_parser.add_argument("--top-down-only", action="store_true")
    project_parser.add_argument("--debug-local-submap", action="store_true")
    project_parser.add_argument(
        "--no-generate-grid",
        dest="generate_grid",
        action="store_false",
        default=True,
    )
    project_parser.add_argument("--grid-only", action="store_true")
    project_parser.add_argument("--grid-resize-factor", type=float, default=0.5)
    project_parser.add_argument("--num-processes", type=int, default=16)

    video_parser = subparsers.add_parser(
        "video",
        help="Build MP4 from pre-rendered projection frames",
    )
    video_parser.add_argument("--visdir", type=str, required=True)
    video_parser.add_argument("--fps", type=int, default=2)
    video_parser.add_argument("--video-name", type=str, default="map_projection")
    video_parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help=(
            "Output MP4 path (required unless "
            f"${KITSCENES_VIZ_OUTPUT_ENV_VAR} is set). Must not lie inside "
            f"${KITSCENES_ROOT_ENV_VAR}."
        ),
    )
    video_parser.add_argument("--resize-factor", type=float, default=0.5)
    video_parser.add_argument("--max-workers", type=int, default=8)
    video_parser.add_argument("--batch-size", type=int, default=100)

    return parser


def _run_project(args: argparse.Namespace) -> None:
    from pathlib import Path

    from kitscenes.visualization.map_projection import MapProjector
    from kitscenes.visualization.map_viz import resolve_map_projection_origin
    from kitscenes.visualization.output_paths import resolve_viz_output_dir

    base_dir = Path(args.base_dir)
    try:
        output_dir = resolve_viz_output_dir(args.output_dir)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    config_path = Path(args.config_path) if args.config_path else base_dir / "calibration" / "calib.json"
    poses_path = Path(args.poses_path) if args.poses_path else base_dir / "poses.txt"
    timestamp_path = (
        Path(args.timestamp_path) if args.timestamp_path else base_dir / "timestamp.reference.txt"
    )
    map_path = Path(args.map_path)
    lat_origin, lon_origin = resolve_map_projection_origin(
        base_dir,
        map_path=map_path,
        lat_origin=args.lat_origin,
        lon_origin=args.lon_origin,
    )

    if args.grid_only and args.top_down_only:
        raise SystemExit("--grid-only cannot be combined with --top-down-only.")
    if args.skip_top_down and args.top_down_only:
        raise SystemExit("--skip-top-down cannot be combined with --top-down-only.")

    generate_grid = args.generate_grid
    if args.top_down_only and generate_grid:
        generate_grid = False

    projector = MapProjector(
        None if args.top_down_only else str(config_path),
        str(map_path),
        str(poses_path),
        str(timestamp_path),
        str(base_dir),
        str(output_dir),
        lat_origin=lat_origin,
        lon_origin=lon_origin,
        front_only=args.front_only,
        skip_top_down=args.skip_top_down,
        top_down_only=args.top_down_only,
        debug_local_submap=args.debug_local_submap,
    )

    if args.grid_only:
        projector.generate_all_grid_images(
            resize_factor=args.grid_resize_factor,
            frame_step=args.frame_step,
            verbose=True,
        )
    else:
        projector.project_all_frames(
            frame_step=args.frame_step,
            num_processes=args.num_processes,
            generate_grid=generate_grid,
            grid_resize_factor=args.grid_resize_factor,
        )


def _run_video(args: argparse.Namespace) -> None:
    from kitscenes.visualization.output_paths import resolve_viz_output_file
    from kitscenes.visualization.video_generation import generate_map_projection_video

    visdir = Path(args.visdir)
    try:
        output_path = resolve_viz_output_file(
            args.output_path,
            default_name=f"{args.video_name}.mp4",
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    generate_map_projection_video(
        visdir,
        output_path,
        fps=args.fps,
        resize_factor=args.resize_factor,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
    )


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "project":
        _run_project(args)
    elif args.command == "video":
        _run_video(args)
    else:
        parser.print_help()
        sys.exit(2)


if __name__ == "__main__":
    main()
