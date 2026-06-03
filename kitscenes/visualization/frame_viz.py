"""Per-frame visualization helpers for the kitscenes dataset.

All functions operate on a :class:`~kitscenes.frame.Frame` and use
**matplotlib** as the rendering backend.

Typical usage::

    from kitscenes.visualization import (
        render_lidar_bev,
        render_lidar_on_camera,
        render_radar_bev,
        render_radar_on_camera,
        render_frame_overview,
    )

    fig, ax = render_lidar_bev(frame)
    fig, ax = render_radar_bev(frame)
    fig, axes = render_frame_overview(frame, show_radar=True)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from kitscenes.frame import Frame

logger = logging.getLogger(__name__)

# Range-ring period for LiDAR BEV coloring: the turbo colormap repeats every
# this many metres, creating concentric "rings" that make distance easy to read.
_LIDAR_RANGE_RING_M = 100.0

# Default color range for LiDAR height coloring used in camera projections
_LIDAR_Z_MIN = -2.0
_LIDAR_Z_MAX = 4.0

# Range-rate color scale: symmetric around zero (m/s)
# Negative = approaching (radar moving towards target), positive = receding
_RADAR_RR_CLIM = 20.0

# Ring camera layout for surround grid (2 rows × 3 cols)
_SURROUND_LAYOUT = [
    ["camera_ring_front_left",  "camera_ring_front",  "camera_ring_front_right"],
    ["camera_ring_rear_left",   "camera_ring_rear",   "camera_ring_rear_right"],
]

# Marker shape per radar sensor — distinguishes sensors without a second color channel
_RADAR_SENSOR_MARKERS = {
    "radar_front": "o",   # circle
    "radar_left":  "^",   # triangle-up
    "radar_right": "s",   # square
}

_RADAR_POINT_SIZE = 45   # fixed size; RCS not encoded (no legend was provided)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_lidar_bev(
    frame: Frame,
    lidar_name: str = "lidar_top",
    *,
    ring_period: float = _LIDAR_RANGE_RING_M,
    point_size: float = 0.5,
    range_m: float = 60.0,
    ax: Optional[Axes] = None,
    out_path: Optional[Path | str] = None,
) -> tuple[Figure, Axes]:
    """Render a top-down bird's-eye view of a single LiDAR sweep.

    Points are coloured by ``range % ring_period`` using the ``turbo``
    colormap, creating concentric distance rings that make it easy to judge
    how far each point is from the sensor.

    Args:
        frame: Source frame.
        lidar_name: Which LiDAR sensor to visualise.
        ring_period: Colormap cycle length in metres (default 100 m).
        point_size: Scatter point size in pts².
        range_m: Half-width of the square view window around the ego vehicle.
        ax: Existing axes to draw into. A new figure is created if ``None``.
        out_path: If given, save the figure to this path.

    Returns:
        ``(fig, ax)`` tuple.
    """
    sweep = frame.lidar(lidar_name)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure  # type: ignore[union-attr]

    if sweep is None:
        ax.text(
            0.5, 0.5, f"{lidar_name}\nnot available",
            ha="center", va="center", transform=ax.transAxes, color="gray",
        )
    else:
        pts = sweep.points
        xyz = np.column_stack([
            pts["x"].astype(np.float64),
            pts["y"].astype(np.float64),
            pts["z"].astype(np.float64),
        ])

        # Range from reference origin (sensor offsets < 1 m, negligible for 100 m rings)
        point_range = np.linalg.norm(xyz, axis=1)

        # Points are already in the reference frame (deskewed output).
        x = xyz[:, 0].astype(np.float32)
        y = xyz[:, 1].astype(np.float32)

        # Modulo creates repeating rings; normalise to [0, 1] for colormap
        ring_color = point_range % ring_period / ring_period

        sc = ax.scatter(x, y, c=ring_color, cmap="turbo", s=point_size,
                        vmin=0.0, vmax=1.0, linewidths=0)
        cb = plt.colorbar(sc, ax=ax, label=f"range mod {ring_period:.0f} m",
                          fraction=0.03, pad=0.04)
        cb.set_ticks([0.0, 0.5, 1.0])
        cb.set_ticklabels([f"0", f"{ring_period/2:.0f}", f"{ring_period:.0f} m"])
        cb.ax.yaxis.label.set_color("white")
        cb.ax.tick_params(colors="white")

    # Ego vehicle marker
    ax.plot(0, 0, marker="^", color="#d62728", markersize=8, zorder=5,
            label="ego")
    ax.set_xlim(-range_m, range_m)
    ax.set_ylim(-range_m, range_m)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(
        f"{lidar_name} — frame {frame.frame_idx} "
        f"(t={frame.timestamp_s:.2f}s, "
        f"{'deskewed' if (sweep and sweep.deskewed) else 'raw'})"
    )
    ax.set_facecolor("#111111")
    fig.patch.set_facecolor("#1a1a1a")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")

    if out_path is not None:
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
    return fig, ax


def render_lidar_on_camera(
    frame: Frame,
    camera_name: str,
    lidar_name: str = "lidar_top",
    *,
    ring_period: float = _LIDAR_RANGE_RING_M,
    point_size: float = 2.0,
    ax: Optional[Axes] = None,
    out_path: Optional[Path | str] = None,
) -> tuple[Figure, Axes]:
    """Overlay projected LiDAR points onto a camera image.

    Points are transformed from the reference frame into the camera frame
    using the calibration extrinsic (T_camera←reference = inv(T_camera→ref))
    and then projected with the pinhole intrinsic matrix.  Only points with
    positive depth that land within the image boundary are shown.

    Points are coloured by ``range % ring_period`` with the ``turbo`` colormap
    (same scheme as :func:`render_lidar_bev`), making distance rings visible
    directly on the image.

    Args:
        frame: Source frame.
        camera_name: Camera to render (e.g. ``"camera_ring_front"``).
        lidar_name: LiDAR sensor whose points are projected.
        ring_period: Colormap cycle length in metres (default 100 m).
        point_size: Scatter point size in pts².
        ax: Existing axes to draw into. A new figure is created if ``None``.
        out_path: If given, save the figure to this path.

    Returns:
        ``(fig, ax)`` tuple.
    """
    image = frame.camera(camera_name)
    sweep = frame.lidar(lidar_name)

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
    else:
        fig = ax.figure  # type: ignore[union-attr]

    if image is None:
        ax.text(0.5, 0.5, f"{camera_name}\nnot available",
                ha="center", va="center", transform=ax.transAxes, color="gray")
        if out_path is not None:
            fig.savefig(out_path, bbox_inches="tight", dpi=150)
        return fig, ax

    ax.imshow(image)
    h_img, w_img = image.shape[:2]

    if sweep is None:
        ax.set_title(f"{camera_name} | {lidar_name} not available")
        if out_path is not None:
            fig.savefig(out_path, bbox_inches="tight", dpi=150)
        return fig, ax

    try:
        calib = frame._loader.get_camera_calibration(camera_name)
    except KeyError:
        logger.warning("No calibration for %s — skipping LiDAR projection", camera_name)
        ax.set_title(f"{camera_name} | no calibration")
        if out_path is not None:
            fig.savefig(out_path, bbox_inches="tight", dpi=150)
        return fig, ax

    pts = sweep.points
    # Points are in the reference frame (deskewed output).
    xyz_ref = np.column_stack([
        pts["x"].astype(np.float64),
        pts["y"].astype(np.float64),
        pts["z"].astype(np.float64),
    ])

    # Range from reference origin (sensor offsets < 1 m, negligible for 100 m rings)
    point_range = np.linalg.norm(xyz_ref, axis=1)

    # T_cam_to_ref → invert to get T_ref_to_cam
    T_cam_to_ref = calib.extrinsic          # (4, 4)
    T_ref_to_cam = np.linalg.inv(T_cam_to_ref)

    R = T_ref_to_cam[:3, :3]
    t = T_ref_to_cam[:3, 3]
    xyz_cam = (R @ xyz_ref.T).T + t         # (N, 3) in camera frame

    # Only keep points in front of the camera
    in_front = xyz_cam[:, 2] > 0.1
    if not np.any(in_front):
        ax.set_title(f"{camera_name} | no LiDAR points in front")
        if out_path is not None:
            fig.savefig(out_path, bbox_inches="tight", dpi=150)
        return fig, ax

    xyz_cam = xyz_cam[in_front]
    point_range_f = point_range[in_front]

    K = calib.intrinsic                     # (3, 3)
    uvw = (K @ xyz_cam.T).T                 # (N, 3)
    u = uvw[:, 0] / uvw[:, 2]
    v = uvw[:, 1] / uvw[:, 2]

    # Filter to image bounds
    in_bounds = (u >= 0) & (u < w_img) & (v >= 0) & (v < h_img)
    u = u[in_bounds]
    v = v[in_bounds]
    ring_color = point_range_f[in_bounds] % ring_period / ring_period

    sc = ax.scatter(u, v, c=ring_color, cmap="turbo", s=point_size,
                    vmin=0.0, vmax=1.0, linewidths=0, alpha=0.8)
    cb = plt.colorbar(sc, ax=ax, label=f"range mod {ring_period:.0f} m",
                      fraction=0.02, pad=0.02)
    cb.set_ticks([0.0, 0.5, 1.0])
    cb.set_ticklabels([f"0", f"{ring_period/2:.0f}", f"{ring_period:.0f} m"])

    ax.set_xlim(0, w_img)
    ax.set_ylim(h_img, 0)
    ax.set_title(
        f"{camera_name} ← {lidar_name}  "
        f"({in_bounds.sum()} pts visible / {len(pts)} total)"
    )
    ax.axis("off")

    if out_path is not None:
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
    return fig, ax


def render_radar_bev(
    frame: Frame,
    radar_names: Optional[Sequence[str]] = None,
    *,
    rr_clim: float = _RADAR_RR_CLIM,
    max_range_m: float = 150.0,
    out_path: Optional[Path | str] = None,
) -> tuple[Figure, np.ndarray]:
    """Render polar (range × azimuth) plots for each radar sensor.

    Radar xyz coordinates are stored in each sensor's local frame (x = sensor
    boresight), so a shared Cartesian BEV is only possible once extrinsic
    calibration is added to ``calib.json``.  Until then this function renders
    one polar subplot per radar — the native representation of the data.

    Point size encodes RCS (dBsm); colour encodes compensated range-rate on a
    diverging scale: **red = approaching** (negative), **blue = receding**
    (positive).

    Args:
        frame: Source frame.
        radar_names: Subset of radars to plot.  Defaults to all available.
        rr_clim: Symmetric colour limit for the range-rate scale (m/s).
        max_range_m: Outer radius of the polar plot in metres.
        out_path: If given, save the figure to this path.

    Returns:
        ``(fig, axes)`` where ``axes`` is a 1-D array of polar Axes, one per
        radar sensor.

    Note:
        Once radar extrinsic calibration is available in ``calib.json`` a
        full reference-frame BEV can be computed.  Add entries keyed by sensor
        name with a ``T_to_reference`` field (4 × 4, same format as cameras
        and lidars) and this function will automatically switch to a shared
        Cartesian BEV.
    """
    names = list(radar_names) if radar_names is not None else frame.available_radar_names()

    # Check whether reference-frame BEV is possible
    extrinsics: dict[str, np.ndarray] = {}
    for name in names:
        try:
            T = frame._loader.get_radar_extrinsic(name)
            extrinsics[name] = T
        except (KeyError, AttributeError):
            pass

    if extrinsics and len(extrinsics) == len(names):
        return _render_radar_bev_reference(
            frame, names, extrinsics, rr_clim=rr_clim,
            max_range_m=max_range_m, out_path=out_path,
        )

    # Fall back to per-sensor polar plots
    n = max(len(names), 1)
    fig, axes_list = plt.subplots(
        1, n, subplot_kw={"projection": "polar"},
        figsize=(5 * n, 5),
    )
    if n == 1:
        axes_list = [axes_list]
    axes = np.array(axes_list)

    sc_last = None
    for ax, radar_name in zip(axes_list, names):
        sweep = frame.radar(radar_name)
        ax.set_facecolor("#111111")
        ax.tick_params(colors="white", labelsize=7)
        ax.spines["polar"].set_color("#444444")

        if sweep is None or len(sweep.points) == 0:
            ax.set_title(f"{radar_name}\nN/A", color="white", fontsize=8)
            continue

        pts = sweep.points
        # azimuth is stored in radians (confirmed: az == arctan2(y, x))
        az = pts["azimuth"].astype(np.float64) if "azimuth" in pts.dtype.names \
            else np.arctan2(pts["y"].astype(np.float64), pts["x"].astype(np.float64))
        r = pts["range"].astype(np.float64) if "range" in pts.dtype.names \
            else np.linalg.norm(
                np.column_stack([pts["x"], pts["y"], pts["z"]]).astype(np.float64), axis=1
            )
        rr = pts["range_rate"].astype(np.float64)

        in_range = r <= max_range_m
        az = az[in_range]; r = r[in_range]
        rr_c = np.clip(rr[in_range], -rr_clim, rr_clim)

        sc_last = ax.scatter(
            az, r, c=rr_c, cmap="RdBu_r",
            vmin=-rr_clim, vmax=rr_clim,
            s=_RADAR_POINT_SIZE, linewidths=0, alpha=0.9, zorder=3,
        )
        ax.set_rmax(max_range_m)
        ax.set_theta_zero_location("N")   # boresight (az=0) points up
        ax.set_theta_direction(-1)         # clockwise (matches sensor x-axis right convention)
        short_name = radar_name.replace("radar_", "")
        ax.set_title(
            f"{short_name}\n{in_range.sum()} pts",
            color="white", fontsize=9, pad=12,
        )

    if sc_last is not None:
        cb = fig.colorbar(sc_last, ax=axes_list, label="range-rate (m/s)",
                          fraction=0.02, pad=0.08, shrink=0.6)
        cb.ax.yaxis.label.set_color("white")
        cb.ax.tick_params(colors="white")
        cb.ax.text(0.5, 1.03, "receding →", transform=cb.ax.transAxes,
                   ha="center", va="bottom", fontsize=7, color="white")
        cb.ax.text(0.5, -0.03, "← approaching", transform=cb.ax.transAxes,
                   ha="center", va="top", fontsize=7, color="white")

    fig.suptitle(
        f"radar — frame {frame.frame_idx}  (t={frame.timestamp_s:.2f}s)  "
        f"[sensor-local frame, boresight = up]",
        color="white", fontsize=9,
    )
    fig.patch.set_facecolor("#1a1a1a")

    if out_path is not None:
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
    return fig, axes


def _render_radar_bev_reference(
    frame: Frame,
    names: list[str],
    extrinsics: dict[str, np.ndarray],
    *,
    rr_clim: float,
    max_range_m: float,
    out_path: Optional[Path | str],
) -> tuple[Figure, np.ndarray]:
    """Reference-frame radar BEV used when extrinsic calibration is available."""
    fig, ax = plt.subplots(figsize=(8, 8))
    sc_last = None
    total_pts = 0

    for radar_name in names:
        sweep = frame.radar(radar_name)
        if sweep is None or len(sweep.points) == 0:
            continue
        pts = sweep.points
        xyz_sensor = np.column_stack([
            pts["x"].astype(np.float64),
            pts["y"].astype(np.float64),
            pts["z"].astype(np.float64),
        ])
        T = extrinsics[radar_name]          # T_sensor_to_reference (4×4)
        R, t = T[:3, :3], T[:3, 3]
        xyz_ref = (R @ xyz_sensor.T).T + t

        rr = np.clip(pts["range_rate"].astype(np.float64), -rr_clim, rr_clim)
        in_range = np.linalg.norm(xyz_ref[:, :2], axis=1) <= max_range_m

        marker = _RADAR_SENSOR_MARKERS.get(radar_name, "o")
        sc_last = ax.scatter(
            xyz_ref[in_range, 0], xyz_ref[in_range, 1],
            c=rr[in_range], cmap="RdBu_r",
            vmin=-rr_clim, vmax=rr_clim,
            s=_RADAR_POINT_SIZE, marker=marker,
            linewidths=0, alpha=0.9, zorder=3,
            label=f"{radar_name.replace('radar_', '')} ({marker})",
        )
        total_pts += int(in_range.sum())

    if sc_last is not None:
        cb = plt.colorbar(sc_last, ax=ax, label="range-rate (m/s)",
                          fraction=0.03, pad=0.04)
        cb.ax.yaxis.label.set_color("white")
        cb.ax.tick_params(colors="white")
        cb.ax.text(0.5, 1.03, "receding →", transform=cb.ax.transAxes,
                   ha="center", va="bottom", fontsize=7, color="white")
        cb.ax.text(0.5, -0.03, "← approaching", transform=cb.ax.transAxes,
                   ha="center", va="top", fontsize=7, color="white")

    ax.plot(0, 0, marker="^", color="#d62728", markersize=8, zorder=5, label="ego")
    leg = ax.legend(loc="upper right", fontsize=7, framealpha=0.3,
                    facecolor="#333333")
    for text in leg.get_texts():
        text.set_color("white")
    lim = min(max_range_m, 150.0)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(
        f"radar BEV — frame {frame.frame_idx} "
        f"(t={frame.timestamp_s:.2f}s, {total_pts} detections)"
    )
    ax.set_facecolor("#111111")
    fig.patch.set_facecolor("#1a1a1a")
    for item in [ax.xaxis.label, ax.yaxis.label, ax.title]:
        item.set_color("white")
    ax.tick_params(colors="white")

    if out_path is not None:
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
    return fig, np.array([ax])


def render_radar_on_camera(
    frame: Frame,
    camera_name: str,
    radar_names: Optional[Sequence[str]] = None,
    *,
    rr_clim: float = _RADAR_RR_CLIM,
    ax: Optional[Axes] = None,
    out_path: Optional[Path | str] = None,
) -> tuple[Figure, Axes]:
    """Overlay projected radar detections onto a camera image.

    Radar points are transformed from the reference frame into the camera
    frame and projected with the pinhole intrinsic.  Colour encodes
    compensated range-rate (blue = receding, red = approaching).  Marker
    shape encodes the sensor (circle = front, triangle = left, square = right).

    Args:
        frame: Source frame.
        camera_name: Camera to render onto.
        radar_names: Subset of radar sensors to project.  Defaults to all
            available radars.
        rr_clim: Symmetric colour limit for the range-rate scale (m/s).
        ax: Existing axes to draw into.
        out_path: If given, save the figure to this path.

    Returns:
        ``(fig, ax)`` tuple.
    """
    image = frame.camera(camera_name)

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
    else:
        fig = ax.figure  # type: ignore[union-attr]

    if image is None:
        ax.text(0.5, 0.5, f"{camera_name}\nnot available",
                ha="center", va="center", transform=ax.transAxes, color="gray")
        if out_path is not None:
            fig.savefig(out_path, bbox_inches="tight", dpi=150)
        return fig, ax

    ax.imshow(image)
    h_img, w_img = image.shape[:2]

    try:
        calib = frame._loader.get_camera_calibration(camera_name)
    except KeyError:
        logger.warning("No calibration for %s — skipping radar projection", camera_name)
        ax.set_title(f"{camera_name} | no calibration")
        if out_path is not None:
            fig.savefig(out_path, bbox_inches="tight", dpi=150)
        return fig, ax

    T_cam_to_ref = calib.extrinsic
    T_ref_to_cam = np.linalg.inv(T_cam_to_ref)
    R = T_ref_to_cam[:3, :3]
    t = T_ref_to_cam[:3, 3]
    K = calib.intrinsic

    names = list(radar_names) if radar_names is not None else frame.available_radar_names()
    sc_last = None
    total_visible = 0

    for radar_name in names:
        sweep = frame.radar(radar_name)
        if sweep is None or len(sweep.points) == 0:
            continue

        try:
            T_radar_to_ref = frame._loader.get_radar_extrinsic(radar_name)
        except (KeyError, AttributeError):
            T_radar_to_ref = None

        pts = sweep.points
        xyz_sensor = np.column_stack([
            pts["x"].astype(np.float64),
            pts["y"].astype(np.float64),
            pts["z"].astype(np.float64),
        ])
        if T_radar_to_ref is not None:
            xyz_ref = (T_radar_to_ref[:3, :3] @ xyz_sensor.T).T + T_radar_to_ref[:3, 3]
        else:
            xyz_ref = xyz_sensor
        rr = pts["range_rate"].astype(np.float64)

        xyz_cam = (R @ xyz_ref.T).T + t
        in_front = xyz_cam[:, 2] > 0.1
        if not np.any(in_front):
            continue

        xyz_cam_f = xyz_cam[in_front]
        rr_f = rr[in_front]

        uvw = (K @ xyz_cam_f.T).T
        u = uvw[:, 0] / uvw[:, 2]
        v = uvw[:, 1] / uvw[:, 2]

        in_bounds = (u >= 0) & (u < w_img) & (v >= 0) & (v < h_img)
        u = u[in_bounds]
        v = v[in_bounds]
        rr_vis = np.clip(rr_f[in_bounds], -rr_clim, rr_clim)
        marker = _RADAR_SENSOR_MARKERS.get(radar_name, "o")

        if not np.any(in_bounds):
            continue

        sc_last = ax.scatter(
            u, v, c=rr_vis, cmap="RdBu_r",
            vmin=-rr_clim, vmax=rr_clim,
            s=_RADAR_POINT_SIZE, marker=marker,
            linewidths=0, alpha=0.9, zorder=3,
            label=f"{radar_name.replace('radar_', '')} ({marker})",
        )
        total_visible += int(in_bounds.sum())

    if sc_last is not None:
        cb = plt.colorbar(sc_last, ax=ax, label="range-rate (m/s)",
                          fraction=0.02, pad=0.02)
        cb.ax.yaxis.label.set_color("white")
        cb.ax.tick_params(colors="white")
        cb.ax.text(0.5, 1.03, "receding →", transform=cb.ax.transAxes,
                   ha="center", va="bottom", fontsize=7, color="white")
        cb.ax.text(0.5, -0.03, "← approaching", transform=cb.ax.transAxes,
                   ha="center", va="top", fontsize=7, color="white")
        leg = ax.legend(loc="lower left", fontsize=7, framealpha=0.5,
                        facecolor="#333333")
        for text in leg.get_texts():
            text.set_color("white")

    ax.set_xlim(0, w_img)
    ax.set_ylim(h_img, 0)
    ax.set_title(
        f"{camera_name} ← radar  ({total_visible} detections visible)"
    )
    ax.axis("off")

    if out_path is not None:
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
    return fig, ax


def render_frame_overview(
    frame: Frame,
    *,
    camera_layout: list[list[str]] = _SURROUND_LAYOUT,
    lidar_name: str = "lidar_top",
    show_radar: bool = True,
    range_m: float = 60.0,
    out_path: Optional[Path | str] = None,
) -> tuple[Figure, np.ndarray]:
    """Render a grid overview of one frame: surround cameras + sensor BEV.

    The layout is a ``(n_rows, n_cols + 1)`` grid.  The rightmost column
    spans all rows and shows a combined LiDAR + radar BEV panel (when
    ``show_radar=True``) or LiDAR only.

    Args:
        frame: Source frame.
        camera_layout: 2-D list of camera names (empty string = blank cell).
            Defaults to the 6-camera ring layout (2 rows × 3 cols).
        lidar_name: LiDAR sensor for the BEV panel.
        show_radar: Overlay radar detections on the BEV panel.
        range_m: BEV view half-width in metres.
        out_path: If given, save the figure to this path.

    Returns:
        ``(fig, axes)`` where ``axes`` is a numpy array of shape
        ``(n_rows, n_cols + 1)``.
    """
    n_rows = len(camera_layout)
    n_cols = max(len(row) for row in camera_layout)
    total_cols = n_cols + 1          # +1 for the sensor BEV column

    fig = plt.figure(figsize=(5 * total_cols, 4 * n_rows))
    gs = fig.add_gridspec(n_rows, total_cols, hspace=0.05, wspace=0.05)

    axes = np.empty((n_rows, total_cols), dtype=object)

    # Camera cells
    for r, row in enumerate(camera_layout):
        for c, cam_name in enumerate(row):
            ax = fig.add_subplot(gs[r, c])
            axes[r, c] = ax
            if not cam_name:
                ax.axis("off")
                continue
            img = frame.camera(cam_name)
            if img is None:
                ax.text(0.5, 0.5, f"{cam_name}\nN/A",
                        ha="center", va="center", transform=ax.transAxes,
                        color="gray", fontsize=8)
                ax.set_facecolor("#1a1a1a")
            else:
                ax.imshow(img)
            ax.set_title(cam_name.replace("camera_ring_", ""), fontsize=7)
            ax.axis("off")

    # Fill unused camera columns in each row
    for r in range(n_rows):
        for c in range(len(camera_layout[r]), n_cols):
            ax = fig.add_subplot(gs[r, c])
            axes[r, c] = ax
            ax.axis("off")

    # Combined BEV — spans all rows in the last column
    ax_bev = fig.add_subplot(gs[:, n_cols])
    axes[:, n_cols] = ax_bev
    render_lidar_bev(frame, lidar_name=lidar_name, range_m=range_m, ax=ax_bev)

    if show_radar and frame.available_radar_names():
        _overlay_radar_on_bev(frame, ax_bev, range_m=range_m)

    fig.suptitle(
        f"Frame {frame.frame_idx}  |  scene {frame.scene_id[:8]}…  "
        f"|  t={frame.timestamp_s:.3f}s",
        color="white", fontsize=10,
    )
    fig.patch.set_facecolor("#1a1a1a")

    if out_path is not None:
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
    return fig, axes


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _overlay_radar_on_bev(
    frame: Frame,
    ax: Axes,
    *,
    range_m: float = 60.0,
    rr_clim: float = _RADAR_RR_CLIM,
) -> None:
    """Overlay radar detections onto a reference-frame LiDAR BEV.

    Requires radar extrinsic calibration in ``calib.json``.  Silently skips
    sensors whose calibration is missing — radar xyz is in sensor-local frame
    and cannot be meaningfully mixed with reference-frame LiDAR without it.
    """
    any_plotted = False
    for radar_name in frame.available_radar_names():
        try:
            T = frame._loader.get_radar_extrinsic(radar_name)
        except KeyError:
            continue  # no calibration — skip this sensor

        sweep = frame.radar(radar_name)
        if sweep is None or len(sweep.points) == 0:
            continue

        pts = sweep.points
        xyz_sensor = np.column_stack([
            pts["x"].astype(np.float64),
            pts["y"].astype(np.float64),
            pts["z"].astype(np.float64),
        ])
        R, t = T[:3, :3], T[:3, 3]
        xyz_ref = (R @ xyz_sensor.T).T + t

        x = xyz_ref[:, 0]; y = xyz_ref[:, 1]
        rr = np.clip(pts["range_rate"].astype(np.float64), -rr_clim, rr_clim)
        in_range = (np.abs(x) <= range_m) & (np.abs(y) <= range_m)
        if not np.any(in_range):
            continue

        marker = _RADAR_SENSOR_MARKERS.get(radar_name, "o")
        ax.scatter(
            x[in_range], y[in_range], c=rr[in_range],
            cmap="RdBu_r", vmin=-rr_clim, vmax=rr_clim,
            s=_RADAR_POINT_SIZE, marker=marker,
            linewidths=0, alpha=0.9, zorder=4,
            label=f"{radar_name.replace('radar_', '')} ({marker})",
        )
        any_plotted = True

    if any_plotted:
        leg = ax.legend(loc="upper right", fontsize=6, framealpha=0.3,
                        facecolor="#333333")
        for text in leg.get_texts():
            text.set_color("white")
