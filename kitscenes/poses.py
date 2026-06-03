"""Pose loading and ego-motion estimation utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from kitscenes.constants import (
    TUM_POSES_FILENAME,
)
from kitscenes.schema import EgoPose

logger = logging.getLogger(__name__)


class _EgoMotion(NamedTuple):
    timestamp_ns: int
    pose_index: int
    linear_velocity_world: np.ndarray
    linear_velocity_reference: np.ndarray
    angular_velocity_world: np.ndarray
    angular_velocity_reference: np.ndarray
    method: str


def load_ego_poses(scene_path: Path) -> list[EgoPose]:
    """Load ego poses from the scene-root ``poses.txt`` file."""
    tum_path = scene_path / TUM_POSES_FILENAME
    if not tum_path.exists():
        logger.debug("No TUM poses file at %s", tum_path)
        return []

    poses: list[EgoPose] = []
    with open(tum_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 8:
                logger.warning("Skipping malformed TUM line: %r", line)
                continue
            ts_s, tx, ty, tz, qx, qy, qz, qw = (float(p) for p in parts)
            poses.append(
                EgoPose(
                    timestamp_ns=int(ts_s * 1e9),
                    translation=np.array([tx, ty, tz], dtype=np.float64),
                    rotation=np.array([qx, qy, qz, qw], dtype=np.float64),
                )
            )

    logger.info("Loaded %d ego poses from %s", len(poses), tum_path)
    return poses


def estimate_ego_motion(
    poses: tuple[EgoPose, ...] | list[EgoPose],
    timestamp_ns: int,
) -> Optional[_EgoMotion]:
    """Estimate ego linear and angular velocity around ``timestamp_ns``.

    The closest pose to the query timestamp is used as the current pose.
    Central differences are preferred; one-sided differences are used at the
    sequence boundaries. Velocities are returned in both world and current
    reference/body coordinates.
    """
    if len(poses) < 2:
        return None

    pose_timestamps = np.array([pose.timestamp_ns for pose in poses], dtype=np.int64)
    pose_index = int(np.argmin(np.abs(pose_timestamps - np.int64(timestamp_ns))))

    if pose_index == 0:
        start_idx = 0
        end_idx = 1
        method = "forward"
    elif pose_index == len(poses) - 1:
        start_idx = len(poses) - 2
        end_idx = len(poses) - 1
        method = "backward"
    else:
        start_idx = pose_index - 1
        end_idx = pose_index + 1
        method = "central"

    start_pose = poses[start_idx]
    end_pose = poses[end_idx]
    current_pose = poses[pose_index]

    dt_ns = end_pose.timestamp_ns - start_pose.timestamp_ns
    if dt_ns <= 0:
        logger.warning(
            "Cannot estimate ego motion around %d due to non-positive dt between poses %d and %d.",
            timestamp_ns,
            start_idx,
            end_idx,
        )
        return None

    dt_s = dt_ns / 1e9
    linear_velocity_world = (
        end_pose.translation.astype(np.float64) - start_pose.translation.astype(np.float64)
    ) / dt_s

    start_rot = Rotation.from_quat(start_pose.rotation)
    end_rot = Rotation.from_quat(end_pose.rotation)
    current_rot = Rotation.from_quat(current_pose.rotation)
    angular_velocity_world = (end_rot * start_rot.inv()).as_rotvec() / dt_s

    linear_velocity_reference = current_rot.inv().apply(linear_velocity_world)
    angular_velocity_reference = current_rot.inv().apply(angular_velocity_world)

    return _EgoMotion(
        timestamp_ns=timestamp_ns,
        pose_index=pose_index,
        linear_velocity_world=linear_velocity_world.astype(np.float64),
        linear_velocity_reference=linear_velocity_reference.astype(np.float64),
        angular_velocity_world=angular_velocity_world.astype(np.float64),
        angular_velocity_reference=angular_velocity_reference.astype(np.float64),
        method=method,
    )


def deskew_lidar(
    raw_points: np.ndarray,
    point_timestamps_s: np.ndarray,
    ego_poses: tuple[EgoPose, ...] | list[EgoPose],
    sweep_timestamp_ns: int,
    T_sensor_to_reference: np.ndarray | None = None,
) -> tuple[np.ndarray, bool, Optional[str]]:
    """Deskew a LiDAR sweep by interpolating ego poses at each point's timestamp.

    Each point is acquired at a slightly different time during the ~100 ms sweep.
    This function transforms all points into the ego reference frame as it was at
    ``sweep_timestamp_ns``, removing the motion blur caused by vehicle movement.

    The math is:
        p_ref_i   = R_s @ p_sensor + t_s           (sensor → reference, via extrinsic)
        p_world   = R_ego_i @ p_ref_i + t_ego_i    (reference → world at acquisition time)
        p_deskewed = R0^T @ (p_world - t0)          (world → reference at sweep time)

    where (R0, t0) is the ego pose at sweep time and (R_ego_i, t_ego_i) is the
    interpolated ego pose at each point's acquisition time.

    Args:
        raw_points: Structured numpy array with fields ``x``, ``y``, ``z``
            and at minimum a ``timestamp`` field (Unix seconds, float64).
        point_timestamps_s: Per-point acquisition timestamps in seconds.
            Points with timestamp == 0.0 are treated as invalid and left
            unchanged (same convention as zero-range returns).
        ego_poses: Sequence of ego poses covering the sweep duration.
            At least 2 poses are required; SLERP is used for rotation
            and linear interpolation for translation.
        sweep_timestamp_ns: Reference timestamp of the sweep in nanoseconds.
        T_sensor_to_reference: Optional (4, 4) sensor-to-reference extrinsic.
            When provided, sensor-local points are first transformed into the
            reference frame before ego-pose motion correction is applied.
            The output is then in the reference frame at ``sweep_timestamp_ns``.
            When ``None``, the input is assumed to already be in the reference
            frame (legacy behaviour for callers without extrinsic data).

    Returns:
        Tuple of (deskewed_points, success, reason).
        On failure, deskewed_points is a copy of raw_points and success=False.
    """
    if len(ego_poses) < 2:
        return raw_points.copy(), False, "insufficient ego poses for deskewing"

    pose_times_s = np.array([p.timestamp_ns / 1e9 for p in ego_poses], dtype=np.float64)
    rots = Rotation.from_quat([p.rotation for p in ego_poses])
    trans = np.array([p.translation for p in ego_poses], dtype=np.float64)
    slerp = Slerp(pose_times_s, rots)

    # Pose at sweep reference time
    sweep_time_s = sweep_timestamp_ns / 1e9
    t0_clamped = float(np.clip(sweep_time_s, pose_times_s[0], pose_times_s[-1]))
    R0 = slerp(t0_clamped).as_matrix()   # (3, 3)
    t0 = np.array([np.interp(t0_clamped, pose_times_s, trans[:, i]) for i in range(3)])
    R0_inv = R0.T                          # orthogonal matrix: inverse == transpose

    # Extract xyz as float64 for numerics
    xyz = np.column_stack([
        raw_points["x"].astype(np.float64),
        raw_points["y"].astype(np.float64),
        raw_points["z"].astype(np.float64),
    ])

    # Valid: non-zero xyz AND non-zero per-point timestamp
    valid = np.any(xyz != 0.0, axis=1) & (point_timestamps_s > 0.0)
    if not np.any(valid):
        return raw_points.copy(), True, None

    pts = xyz[valid]
    ts = point_timestamps_s[valid].clip(pose_times_s[0], pose_times_s[-1])

    # Apply sensor extrinsic: sensor-local → reference frame
    if T_sensor_to_reference is not None:
        R_s = T_sensor_to_reference[:3, :3]
        t_s = T_sensor_to_reference[:3, 3]
        pts = (R_s @ pts.T).T + t_s

    # Interpolate pose at each point's acquisition time
    interp_rot = slerp(ts)
    interp_trans = np.column_stack([
        np.interp(ts, pose_times_s, trans[:, i]) for i in range(3)
    ])

    # p_world = R_ego_i @ p_ref_i + t_ego_i  →  p_deskewed = R0^T @ (p_world - t0)
    p_world = interp_rot.apply(pts) + interp_trans          # (N, 3)
    deskewed = (R0_inv @ (p_world - t0).T).T                # (N, 3)

    result = raw_points.copy()
    result["x"][valid] = deskewed[:, 0].astype(result.dtype["x"])
    result["y"][valid] = deskewed[:, 1].astype(result.dtype["y"])
    result["z"][valid] = deskewed[:, 2].astype(result.dtype["z"])

    return result, True, None