"""HD map visualization via the Lanelet2 ML converter.

Bridges :class:`~kitscenes.frame.Frame` / :class:`~kitscenes.schema.Scene`
to the tested ``ml_converter_vis_utils`` and ``map_projection`` pipelines.
"""

from __future__ import annotations

import json
import logging
import os
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from kitscenes.visualization.ml_converter_vis_utils import get_map_data, plot_map_data

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

    from kitscenes.frame import Frame
    from kitscenes.map_api import SceneMap
    from kitscenes.schema import EgoPose, Scene
    from kitscenes.visualization.map_projection import MapProjector

from kitscenes.visualization.output_paths import (
    resolve_viz_output_dir,
    resolve_viz_output_file,
)

logger = logging.getLogger(__name__)


def default_car_img_path() -> Optional[str]:
    """Return the default ego-car overlay image path, if present."""
    env_path = Path(os.environ["KITSCENES_CAR_IMG"]) if "KITSCENES_CAR_IMG" in os.environ else None
    candidates = [
        env_path,
        Path(__file__).resolve().parents[2] / "res" / "car_img.png",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return str(candidate)
    return None


def _require_ml_converter() -> None:
    try:
        from lanelet2.ml_converter import MapDataInterface  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "lanelet2 with ml_converter support is required for HD map visualization. "
            "Install a wheel from res/ml_converter_wheels/."
        ) from exc


def ego_pose_to_map_query(ego_pose: "EgoPose") -> tuple[np.ndarray, float, float, float]:
    """Convert an :class:`~kitscenes.schema.EgoPose` to ML-converter query parameters."""
    rot = Rotation.from_quat(ego_pose.rotation)
    euler_zyx = rot.as_euler("zyx", degrees=False)
    return ego_pose.translation, float(euler_zyx[0]), float(euler_zyx[1]), float(euler_zyx[2])


def get_map_data_for_ego_pose(
    scene_map: "SceneMap",
    ego_pose: "EgoPose",
    *,
    type_grouping: str = "default",
    extent_long: float = 150.0,
    extent_lat: float = 75.0,
    n_points_lanes: int = 60,
    ignore_map_ele: bool = False,
) -> Any:
    """Extract ML-converter :class:`MapData` for a single ego pose."""
    _require_ml_converter()
    position, yaw, pitch, roll = ego_pose_to_map_query(ego_pose)
    ll2_map = scene_map.lanelet_map
    return get_map_data(
        ll2_map,
        float(position[0]),
        float(position[1]),
        float(position[2]),
        yaw,
        pitch,
        roll,
        ignore_map_ele=ignore_map_ele,
        type_grouping=type_grouping,
        extent_long=extent_long,
        extent_lat=extent_lat,
        n_points_lanes=n_points_lanes,
    )


def get_map_data_for_frame(
    frame: "Frame",
    scene_map: "SceneMap",
    **kwargs: Any,
) -> Any:
    """Extract ML-converter map labels for a :class:`~kitscenes.frame.Frame`."""
    ego_pose = frame.ego_pose
    if ego_pose is None:
        raise ValueError(f"Frame {frame.frame_idx} has no ego pose")
    return get_map_data_for_ego_pose(scene_map, ego_pose, **kwargs)


def generate_map_labels_for_scene(
    scene: "Scene",
    *,
    type_grouping: str = "default",
    frame_step: int = 1,
    max_frames: int | None = None,
    extent_long: float = 150.0,
    extent_lat: float = 75.0,
    n_points_lanes: int = 60,
    ignore_map_ele: bool = False,
) -> tuple[list[Any], list["EgoPose"]]:
    """Generate ML-converter map labels along a scene's reference timeline.

    Uses :attr:`~kitscenes.schema.Scene.ego_poses` and
    :attr:`~kitscenes.schema.Scene.map` (Lanelet2 + ``maps/origin.json``).

    Args:
        scene: Loaded scene with map and ego poses on disk.
        type_grouping: ML-converter grouping preset (``default``, ``m3tr``, …).
        frame_step: Use every N-th reference frame (``1`` = all frames).
        max_frames: Optional cap on the number of labels generated.
        extent_long: Longitudinal submap extent passed to the ML converter.
        extent_lat: Lateral submap extent passed to the ML converter.
        n_points_lanes: Lane resampling density for MapData extraction.
        ignore_map_ele: Ignore map elevation in extraction.

    Returns:
        ``(map_data_list, ego_poses)`` — one MapData (or error string) per
        selected frame, and the corresponding ego poses.
    """
    if frame_step < 1:
        raise ValueError(f"frame_step must be >= 1, got {frame_step}")

    scene_map = scene.map
    if scene_map is None:
        raise FileNotFoundError(
            f"Scene {scene.scene_id!r} has no HD map at {scene.scene_path / 'maps'}"
        )

    ego_poses = scene.ego_poses
    if not ego_poses:
        raise ValueError(f"Scene {scene.scene_id!r} has no ego poses in poses.txt")

    num_frames = scene.metadata.num_frames
    if len(ego_poses) != num_frames:
        logger.warning(
            "Scene %s: %d reference frames but %d ego poses; using the first %d.",
            scene.scene_id,
            num_frames,
            len(ego_poses),
            min(num_frames, len(ego_poses)),
        )
    limit = min(num_frames, len(ego_poses))
    indices = list(range(0, limit, frame_step))
    if max_frames is not None:
        indices = indices[:max_frames]

    map_data_list: list[Any] = []
    poses_used: list[EgoPose] = []
    start = time.time()
    for frame_idx in indices:
        ego_pose = ego_poses[frame_idx]
        m_data = get_map_data_for_ego_pose(
            scene_map,
            ego_pose,
            type_grouping=type_grouping,
            extent_long=extent_long,
            extent_lat=extent_lat,
            n_points_lanes=n_points_lanes,
            ignore_map_ele=ignore_map_ele,
        )
        map_data_list.append(m_data)
        poses_used.append(ego_pose)

    logger.info(
        "Generated %d map labels for scene %s in %.2f s",
        len(map_data_list),
        scene.scene_id,
        time.time() - start,
    )
    return map_data_list, poses_used


def render_map_top_down(
    ego_pose: "EgoPose",
    scene_map: "SceneMap",
    *,
    ax: Optional["matplotlib.axes.Axes"] = None,
    type_grouping: str = "default",
    car_img_path: Optional[str] = None,
    **plot_kwargs: Any,
) -> tuple["matplotlib.figure.Figure", "matplotlib.axes.Axes"]:
    """Render an ego-centric top-down HD map view using the ML converter."""
    m_data = get_map_data_for_ego_pose(
        scene_map,
        ego_pose,
        type_grouping=type_grouping,
    )
    if isinstance(m_data, str):
        raise RuntimeError(f"MapData extraction failed: {m_data}")

    created_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 20))
    else:
        fig = ax.get_figure()

    if car_img_path is None:
        car_img_path = default_car_img_path()

    plot_map_data(
        m_data,
        car_img_path=car_img_path,
        orientation="vertical",
        show_drivable_area=True,
        max_arrow_length_fraction=0.3,
        **plot_kwargs,
    )

    if created_fig:
        return fig, plt.gca()
    return fig, ax


def render_map_top_down_for_frame(
    frame: "Frame",
    scene_map: "SceneMap",
    **kwargs: Any,
) -> tuple["matplotlib.figure.Figure", "matplotlib.axes.Axes"]:
    """Render a top-down HD map for a dataset frame."""
    ego_pose = frame.ego_pose
    if ego_pose is None:
        raise ValueError(f"Frame {frame.frame_idx} has no ego pose")
    return render_map_top_down(ego_pose, scene_map, **kwargs)


def read_map_origin(scene_path: Path) -> tuple[float, float]:
    """Read ``maps/origin.json`` lat/lon for a scene directory."""
    origin_path = scene_path / "maps" / "origin.json"
    data = json.loads(origin_path.read_text(encoding="utf-8"))
    return float(data["latitude"]), float(data["longitude"])


def resolve_map_projection_origin(
    scene_path: Path | str,
    map_path: Path | str | None = None,
    *,
    lat_origin: float | None = None,
    lon_origin: float | None = None,
) -> tuple[float, float]:
    """Resolve UTM projection origin for map loading.

    Priority:
    1. Explicit *lat_origin* / *lon_origin* when both are provided.
    2. ``<scene>/maps/origin.json`` (or next to *map_path*).
    3. Legacy Frankfurt fallback with a warning.
    """
    if lat_origin is not None and lon_origin is not None:
        return lat_origin, lon_origin

    scene_path = Path(scene_path)
    candidates = [scene_path / "maps" / "origin.json"]
    if map_path is not None:
        candidates.append(Path(map_path).parent / "origin.json")

    for origin_path in candidates:
        if origin_path.is_file():
            data = json.loads(origin_path.read_text(encoding="utf-8"))
            lat = float(data["latitude"])
            lon = float(data["longitude"])
            logger.info(
                "Using map projection origin from %s (lat=%.5f, lon=%.5f)",
                origin_path,
                lat,
                lon,
            )
            return lat, lon

    logger.warning(
        "maps/origin.json not found under %s; falling back to Frankfurt UTM origin. "
        "Map elements may be missing if the scene is elsewhere.",
        scene_path,
    )
    return 50.110423, 8.682138


def create_map_projector(
    scene_path: Path | str,
    output_dir: Path | str | None,
    *,
    type_grouping: str = "default",
    front_only: bool = False,
    skip_top_down: bool = False,
    top_down_only: bool = False,
    debug_local_submap: bool = False,
) -> "MapProjector":
    """Build a :class:`~kitscenes.visualization.map_projection.MapProjector` for a scene."""
    from kitscenes.visualization.map_projection import MapProjector

    scene_path = Path(scene_path)
    output_dir = resolve_viz_output_dir(output_dir)
    config_path = scene_path / "calibration" / "calib.json"
    map_path = scene_path / "maps" / "map.osm"
    lat_origin, lon_origin = resolve_map_projection_origin(scene_path, map_path=map_path)
    poses_path = scene_path / "poses.txt"
    timestamp_path = scene_path / "timestamp.reference.txt"

    return MapProjector(
        str(config_path) if config_path.is_file() else None,
        str(map_path),
        str(poses_path),
        str(timestamp_path) if timestamp_path.is_file() else None,
        str(scene_path),
        str(output_dir),
        type_grouping=type_grouping,
        front_only=front_only,
        skip_top_down=skip_top_down,
        top_down_only=top_down_only,
        debug_local_submap=debug_local_submap,
        lat_origin=lat_origin,
        lon_origin=lon_origin,
    )


def project_scene_maps(
    scene_path: Path | str,
    output_dir: Path | str | None,
    **kwargs: Any,
) -> "MapProjector":
    """Run the full map-on-camera + top-down projection pipeline for a scene."""
    projector = create_map_projector(scene_path, output_dir, **kwargs)
    projector.project_all_frames(**kwargs)
    return projector


def warn_deprecated_scene_bev_map() -> None:
    warnings.warn(
        "Drawing HD map layers inside render_scene_bev() was removed. "
        "Use render_map_top_down() or MapProjector for ML-converter map visualization.",
        DeprecationWarning,
        stacklevel=3,
    )
