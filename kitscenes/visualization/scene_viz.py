"""Scene visualization helpers for the kitscenes dataset.

All functions use **matplotlib** as the primary rendering backend.  OpenCV
(``cv2``) is used only for video writing in :func:`render_scene_animation`.
Plotly is an optional dependency for :func:`plot_map_interactive`.

Typical usage::

    from kitscenes.visualization import render_scene_bev, render_surround_view

    fig, ax = render_scene_bev(scene, timestep=50)
    surround = render_surround_view(sensor_loader, frame_idx=100)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from numpy.typing import NDArray

from kitscenes.schema import EgoPose, Scene
from kitscenes.sensors import SensorDataLoader

logger = logging.getLogger(__name__)

_EGO_COLOR = "#d62728"


# ---------------------------------------------------------------------------
# 1. Bird's-eye view
# ---------------------------------------------------------------------------


def render_scene_bev(
    scene: Scene,
    timestep: int | None = None,
    show_map: bool = False,
    show_ego: bool = True,
    roi_radius: float = 80.0,
    ax: matplotlib.axes.Axes | None = None,
    out_path: Path | None = None,
    scene_map: Optional["kitscenes.map_api.SceneMap"] = None,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Render a bird's-eye view of the scene at a given timestep.

    If *timestep* is ``None``, plot the **full** ego trajectory.

    Args:
        scene: The scene to visualise.
        timestep: Reference-timeline index (0-based).  ``None`` for all.
        show_map: Deprecated — use :func:`~kitscenes.visualization.render_map_top_down`
            for ML-converter HD map rendering.
        show_ego: If ``True``, draw the ego trajectory / pose.
        roi_radius: Radius (m) around the ego for the map ROI query.
        ax: Existing axes to draw on.  Created if ``None``.
        out_path: If given, save the figure to this path.
        scene_map: Deprecated — kept for API compatibility only.

    Returns:
        ``(fig, ax)`` tuple.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    else:
        fig = ax.get_figure()

    # -- ego poses ----------------------------------------------------------
    ego_xy = np.array(
        [ep.translation[:2] for ep in scene.ego_poses], dtype=np.float64
    )

    if timestep is not None:
        _validate_timestep(timestep, len(scene.ego_poses))
        ego_center = ego_xy[timestep]
    else:
        ego_center = ego_xy[len(ego_xy) // 2] if len(ego_xy) > 0 else np.zeros(2)

    # -- map layers (deprecated) --------------------------------------------
    if show_map:
        from kitscenes.visualization.map_viz import warn_deprecated_scene_bev_map
        warn_deprecated_scene_bev_map()

    # -- ego trajectory -----------------------------------------------------
    if show_ego and len(ego_xy) > 0:
        if timestep is not None:
            # Full trajectory as thin line, current position as marker
            ax.plot(ego_xy[:, 0], ego_xy[:, 1], "-", color=_EGO_COLOR,
                    linewidth=0.8, alpha=0.4, label="ego trajectory")
            ax.plot(ego_xy[timestep, 0], ego_xy[timestep, 1], "o",
                    color=_EGO_COLOR, markersize=8, zorder=5, label="ego")
            _draw_heading_arrow(ax, scene.ego_poses[timestep], color=_EGO_COLOR)
        else:
            ax.plot(ego_xy[:, 0], ego_xy[:, 1], "-", color=_EGO_COLOR,
                    linewidth=1.5, label="ego trajectory")
            # Start / end markers
            ax.plot(ego_xy[0, 0], ego_xy[0, 1], "s", color=_EGO_COLOR,
                    markersize=6, zorder=5)
            ax.plot(ego_xy[-1, 0], ego_xy[-1, 1], "^", color=_EGO_COLOR,
                    markersize=6, zorder=5)

    # -- formatting ---------------------------------------------------------
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize="small", framealpha=0.8)

    title = f"Scene {scene.scene_id}"
    if timestep is not None:
        title += f"  t={timestep}"
    ax.set_title(title)

    if out_path is not None:
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        logger.info("Saved BEV to %s", out_path)

    return fig, ax


# ---------------------------------------------------------------------------
# 2. Scene animation
# ---------------------------------------------------------------------------


def render_scene_animation(
    scene: Scene,
    out_path: Path,
    fps: int = 10,
    show_map: bool = False,
    roi_radius: float = 80.0,
    scene_map: Optional["kitscenes.map_api.SceneMap"] = None,
    follow_ego: bool = True,
) -> None:
    """Render an animated video of the full scene.

    Each frame corresponds to one reference-timeline timestep.  Requires
    ``matplotlib`` with a movie writer backend (``ffmpeg`` or ``pillow``).

    Args:
        scene: Scene to animate.
        out_path: Output file path (``.mp4``, ``.gif``, etc.).
        fps: Frames per second.
        show_map: Deprecated — use :func:`~kitscenes.visualization.render_map_top_down`.
        roi_radius: Map ROI radius around the ego.
        scene_map: Optional :class:`~kitscenes.map_api.SceneMap`.
        follow_ego: If ``True``, centre the viewport on the ego each frame.
    """
    from matplotlib.animation import FuncAnimation

    num_frames = len(scene.ego_poses)
    if num_frames == 0:
        logger.warning("No ego poses — cannot create animation")
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    ego_xy = np.array(
        [ep.translation[:2] for ep in scene.ego_poses], dtype=np.float64
    )

    def _update(frame_idx: int) -> None:
        ax.cla()

        ego_center = ego_xy[frame_idx]

        if show_map:
            from kitscenes.visualization.map_viz import warn_deprecated_scene_bev_map
            warn_deprecated_scene_bev_map()

        # Ego trajectory up to current frame
        ax.plot(ego_xy[: frame_idx + 1, 0], ego_xy[: frame_idx + 1, 1],
                "-", color=_EGO_COLOR, linewidth=1.0, alpha=0.5)
        ax.plot(ego_center[0], ego_center[1], "o", color=_EGO_COLOR,
                markersize=8, zorder=5)
        _draw_heading_arrow(ax, scene.ego_poses[frame_idx], color=_EGO_COLOR)

        if follow_ego:
            ax.set_xlim(ego_center[0] - roi_radius, ego_center[0] + roi_radius)
            ax.set_ylim(ego_center[1] - roi_radius, ego_center[1] + roi_radius)

        ax.set_aspect("equal")
        ax.set_title(f"Scene {scene.scene_id}  t={frame_idx}/{num_frames - 1}")

    anim = FuncAnimation(fig, _update, frames=num_frames, interval=1000 / fps)
    anim.save(str(out_path), fps=fps)
    plt.close(fig)
    logger.info("Saved animation (%d frames) to %s", num_frames, out_path)


# ---------------------------------------------------------------------------
# 3. Surround view
# ---------------------------------------------------------------------------

# Layout: 2 rows × 3 columns, physically arranged around the vehicle.
_SURROUND_LAYOUT: list[list[str]] = [
    ["camera_ring_front_left", "camera_ring_front", "camera_ring_front_right"],
    ["camera_ring_rear_left",  "camera_ring_rear",  "camera_ring_rear_right"],
]


def render_surround_view(
    sensor_loader: SensorDataLoader,
    frame_idx: int,
    camera_names: list[list[str]] | None = None,
    ax_grid: NDArray | None = None,
    out_path: Path | None = None,
) -> tuple[matplotlib.figure.Figure, NDArray]:
    """Compose a surround-view grid from ring cameras at a given frame.

    Args:
        sensor_loader: Sensor loader for the scene.
        frame_idx: Frame index to load images for.
        camera_names: 2-D list defining the grid layout.  Defaults to the
            standard 2×3 ring-camera arrangement.  Use ``""`` for empty cells.
        ax_grid: Optional pre-created array of axes (shape matching layout).
        out_path: If given, save the figure.

    Returns:
        ``(fig, axes)`` where *axes* is a 2-D numpy array of
        :class:`matplotlib.axes.Axes`.
    """
    layout = camera_names if camera_names is not None else _SURROUND_LAYOUT
    nrows = len(layout)
    ncols = max(len(row) for row in layout)

    if ax_grid is None:
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(5 * ncols, 4 * nrows),
        )
        axes = np.atleast_2d(axes)
    else:
        axes = ax_grid
        fig = axes.flat[0].get_figure()

    available = set(sensor_loader.get_camera_names())

    for r, row in enumerate(layout):
        for c, cam_name in enumerate(row):
            ax = axes[r, c]
            if not cam_name or cam_name not in available:
                ax.axis("off")
                if cam_name:
                    ax.set_title(cam_name.replace("camera_ring_", ""), fontsize=8)
                continue
            try:
                img = sensor_loader.get_camera_image(cam_name, frame_idx)
                ax.imshow(img)
            except (FileNotFoundError, RuntimeError) as exc:
                logger.debug("Could not load %s frame %d: %s", cam_name, frame_idx, exc)
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, fontsize=14)
            ax.axis("off")
            ax.set_title(cam_name.replace("camera_ring_", ""), fontsize=8)

    fig.tight_layout(pad=0.5)

    if out_path is not None:
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")

    return fig, axes


# ---------------------------------------------------------------------------
# 4. Interactive map (optional — requires plotly)
# ---------------------------------------------------------------------------


def plot_map_interactive(
    scene_map: "kitscenes.map_api.SceneMap",
    center: NDArray[np.float64] | None = None,
    radius: float = 100.0,
    ego_poses: Sequence[EgoPose] | None = None,
) -> "plotly.graph_objs.Figure":
    """Interactive Plotly visualisation of the HD map.

    For ego-centric ML-converter map views (traffic elements, styled lane
    markings), prefer :func:`~kitscenes.visualization.render_map_top_down`.

    Args:
        scene_map: The HD map to render.
        center: (2,) query centre.  If ``None``, uses the map centroid.
        radius: Radius in metres for the lane segment query.
        ego_poses: Optional GNSS/INS ego trajectory to overlay.

    Returns:
        A Plotly ``Figure`` (call ``.show()`` to display).

    Raises:
        ImportError: If plotly is not installed.
    """
    try:
        import plotly.graph_objects as go  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            "plotly is required for plot_map_interactive.  "
            "Install via: pip install plotly"
        )

    fig = go.Figure()

    if center is None:
        # Use map bounding box centroid as fallback
        center = np.array([0.0, 0.0], dtype=np.float64)

    # Offset to convert map-local coords to absolute UTM
    offset = scene_map.utm_origin  # (2,) [easting, northing]
    local_center = center - offset

    segments = scene_map.get_lane_segments_in_roi(local_center, radius)

    for seg in segments:
        for boundary, name, color in [
            (seg.left_boundary, "left", "#636EFA"),
            (seg.right_boundary, "right", "#636EFA"),
            (seg.centerline, "center", "#FFA15A"),
        ]:
            if len(boundary) == 0:
                continue
            abs_b = boundary + offset
            fig.add_trace(go.Scatter(
                x=abs_b[:, 0], y=abs_b[:, 1],
                mode="lines",
                line=dict(color=color, width=1 if name != "center" else 2,
                          dash="dash" if name == "center" else "solid"),
                name=f"lanelet {seg.lanelet_id} {name}",
                showlegend=False,
                hoverinfo="name",
            ))

    # Crosswalks
    for i, cw in enumerate(scene_map.get_crosswalks()):
        abs_cw = cw + offset
        closed = np.vstack([abs_cw, abs_cw[:1]])
        fig.add_trace(go.Scatter(
            x=closed[:, 0], y=closed[:, 1],
            mode="lines", fill="toself",
            fillcolor="rgba(227, 119, 194, 0.3)",
            line=dict(color="#e377c2", width=1),
            name=f"crosswalk {i}",
            showlegend=False,
        ))

    # Ego trajectory
    if ego_poses is not None and len(ego_poses) > 0:
        xy = np.array([ep.translation[:2] for ep in ego_poses])
        fig.add_trace(go.Scatter(
            x=xy[:, 0], y=xy[:, 1],
            mode="lines+markers",
            line=dict(color=_EGO_COLOR, width=3),
            marker=dict(size=2),
            name="ego",
        ))

    fig.update_layout(
        xaxis=dict(scaleanchor="y", scaleratio=1, title="x (m)"),
        yaxis=dict(title="y (m)"),
        template="plotly_white",
        title="HD Map",
        width=900,
        height=900,
    )

    return fig


# ===================================================================
# Private drawing helpers
# ===================================================================


def _validate_timestep(timestep: int, max_len: int) -> None:
    if timestep < 0 or timestep >= max_len:
        raise IndexError(
            f"timestep {timestep} out of range [0, {max_len})"
        )


def _draw_heading_arrow(
    ax: matplotlib.axes.Axes,
    ego_pose: EgoPose,
    color: str,
    length: float = 3.0,
) -> None:
    """Draw a heading arrow from the ego pose using its quaternion rotation."""
    x, y = ego_pose.translation[0], ego_pose.translation[1]

    # Extract yaw from quaternion (qx, qy, qz, qw)
    qx, qy, qz, qw = ego_pose.rotation
    # yaw = atan2(2*(qw*qz + qx*qy), 1 - 2*(qy^2 + qz^2))
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))

    dx = length * np.cos(yaw)
    dy = length * np.sin(yaw)
    ax.annotate(
        "", xy=(x + dx, y + dy), xytext=(x, y),
        arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
        zorder=6,
    )
