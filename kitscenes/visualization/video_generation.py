"""Video generation from pre-rendered map projection frames.

Assembles 2×3 camera grid images (optionally with top-down BEV appended)
into an MP4 video.  Logic ported from ``lg_dataset_api/src/visualization/video_generation.py``.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[assignment,misc]


def get_camera_image(visdir: Path | str, camera_name: str, frame_idx: int) -> Optional[np.ndarray]:
    """Load a single camera projection image for a given frame."""
    img_path = Path(visdir) / camera_name / f"{frame_idx:010d}.jpg"
    if img_path.is_file():
        return cv2.imread(str(img_path))
    return None


def get_top_down_image(visdir: Path | str, frame_idx: int) -> Optional[np.ndarray]:
    """Load the top-down view image for a given frame."""
    img_path = Path(visdir) / "top_down" / f"{frame_idx:010d}.jpg"
    if img_path.is_file():
        return cv2.imread(str(img_path))
    return None


def create_composite_frame(
    visdir: Path | str,
    frame_idx: int,
    resize_factor: float = 0.5,
) -> Optional[np.ndarray]:
    """Load pre-generated grid image and optionally append top-down view on the right."""
    grid_path = Path(visdir) / "grid" / f"{frame_idx:010d}.jpg"
    if not grid_path.is_file():
        return None

    grid = cv2.imread(str(grid_path))
    if grid is None:
        return None

    top_down_img = get_top_down_image(visdir, frame_idx)
    if top_down_img is not None:
        target_height = grid.shape[0]
        aspect_ratio = top_down_img.shape[1] / top_down_img.shape[0]
        target_width = int(target_height * aspect_ratio)
        top_down_resized = cv2.resize(top_down_img, (target_width, target_height))
        grid = cv2.hconcat([grid, top_down_resized])

    return grid


def process_frame(
    visdir: Path | str,
    frame_idx: int,
    size: tuple[int, int],
    resize_factor: float,
) -> tuple[int, Optional[np.ndarray]]:
    """Process a single frame and return the composite grid image."""
    grid_img = create_composite_frame(visdir, frame_idx, resize_factor)
    if grid_img is not None:
        resized_img = cv2.resize(grid_img, size)
        return frame_idx, resized_img
    return frame_idx, None


def get_frame_indices(visdir: Path | str) -> list[int]:
    """Get all unique frame indices from the ``grid/`` directory."""
    frame_set: set[int] = set()
    grid_path = Path(visdir) / "grid"
    if grid_path.is_dir():
        for filename in os.listdir(grid_path):
            if filename.endswith(".jpg"):
                frame_str = filename.replace(".jpg", "")
                try:
                    frame_set.add(int(frame_str))
                except ValueError:
                    continue
    return sorted(frame_set)


def generate_map_projection_video(
    visdir: Path | str,
    output_path: Path | str,
    *,
    fps: int = 2,
    resize_factor: float = 0.5,
    max_workers: int = 8,
    batch_size: int = 100,
    frame_indices: Optional[Sequence[int]] = None,
    verbose: bool = True,
) -> Path:
    """Create an MP4 video from pre-rendered map projection images.

    Args:
        visdir: Directory containing ``grid/``, optional ``top_down/``, and
            per-camera projection subdirectories.
        output_path: Output ``.mp4`` file path.
        fps: Video frame rate.
        resize_factor: Passed through to frame processing (legacy parameter).
        max_workers: Thread pool size for parallel frame loading.
        batch_size: Number of frames loaded per batch to limit memory use.
        frame_indices: Explicit frame list.  Scanned from ``grid/`` when ``None``.
        verbose: Print progress information.

    Returns:
        Path to the written video file.
    """
    visdir = Path(visdir)
    output_path = Path(output_path)
    if not visdir.is_dir():
        raise FileNotFoundError(f"Projection directory not found: {visdir}")

    if frame_indices is None:
        frame_indices = get_frame_indices(visdir)
    else:
        frame_indices = sorted(int(idx) for idx in frame_indices)

    if not frame_indices:
        raise ValueError(f"No frames found under {visdir / 'grid'}")

    sample_grid = create_composite_frame(visdir, frame_indices[0], resize_factor)
    if sample_grid is None:
        raise RuntimeError(f"Could not load sample grid frame {frame_indices[0]}")
    size = (sample_grid.shape[1], sample_grid.shape[0])

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(str(output_path), fourcc, fps, size, True)
    if not video.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")

    frames_written = 0
    num_batches = (len(frame_indices) + batch_size - 1) // batch_size
    progress_iter: Iterable[int] = range(num_batches)
    if verbose and tqdm is not None:
        progress_iter = tqdm(progress_iter, desc="Video batches")

    try:
        for batch_num in progress_iter:
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, len(frame_indices))
            batch_frames = frame_indices[start_idx:end_idx]
            frames_dict: dict[int, np.ndarray] = {}

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        process_frame, visdir, frame_idx, size, resize_factor
                    ): frame_idx
                    for frame_idx in batch_frames
                }
                completed = as_completed(futures)
                if verbose and tqdm is not None:
                    completed = tqdm(completed, total=len(futures), desc="Loading batch", leave=False)
                for future in completed:
                    frame_idx, img = future.result()
                    if img is not None:
                        frames_dict[frame_idx] = img

            for frame_idx in batch_frames:
                img = frames_dict.get(frame_idx)
                if img is not None:
                    video.write(img)
                    frames_written += 1
    finally:
        video.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    if verbose:
        print(f"Wrote {frames_written}/{len(frame_indices)} frames to {output_path}")
    return output_path


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate video from map projection images",
    )
    parser.add_argument(
        "--visdir",
        required=True,
        help="Directory containing projection images (grid/, top_down/, cameras/)",
    )
    parser.add_argument("--fps", default=2, type=int, help="FPS for output video")
    parser.add_argument(
        "--video-name",
        default="map_projection",
        type=str,
        help="Output video filename (without extension)",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        type=str,
        help="Output MP4 path (required unless $KITSCENES_VIZ_OUTPUT is set; "
             "must not be inside $KITSCENES_ROOT).",
    )
    parser.add_argument(
        "--resize-factor",
        default=0.5,
        type=float,
        help="Resize factor for individual camera images",
    )
    parser.add_argument(
        "--max-workers",
        default=8,
        type=int,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--batch-size",
        default=100,
        type=int,
        help="Frames per batch (limits memory usage)",
    )
    args = parser.parse_args(argv)

    visdir = Path(args.visdir)
    from kitscenes.visualization.output_paths import resolve_viz_output_file

    try:
        video_path = resolve_viz_output_file(
            args.output_path,
            default_name=f"{args.video_name}.mp4",
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    generate_map_projection_video(
        visdir,
        video_path,
        fps=args.fps,
        resize_factor=args.resize_factor,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
