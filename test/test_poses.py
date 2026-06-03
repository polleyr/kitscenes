"""Tests for kitscenes.poses."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kitscenes.poses import deskew_lidar, estimate_ego_motion, load_ego_poses
from kitscenes.schema import EgoPose


def test_load_ego_poses(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    scene.mkdir()
    (scene / "poses.txt").write_text(
        "1.0 1 2 3 0 0 0 1\n"
        "2.0 4 5 6 0 0 0 1\n"
    )
    poses = load_ego_poses(scene)
    assert len(poses) == 2
    assert poses[0].timestamp_ns == 1_000_000_000
    np.testing.assert_array_equal(poses[0].translation, [1, 2, 3])


def test_estimate_ego_motion_central() -> None:
    poses = tuple(
        EgoPose(
            int(i * 1e9),
            np.array([float(i), 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 1.0]),
        )
        for i in range(3)
    )
    motion = estimate_ego_motion(poses, timestamp_ns=1_000_000_000)
    assert motion is not None
    assert motion.method == "central"
    np.testing.assert_allclose(motion.linear_velocity_world[0], 1.0, rtol=0.1)


def test_deskew_lidar_requires_poses() -> None:
    points = np.array([(1.0, 0.0, 0.0, 0.5, 1.0)], dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"), ("reflectivity", "f4"), ("timestamp", "f8")
    ])
    result, success, reason = deskew_lidar(
        points,
        points["timestamp"],
        [],
        sweep_timestamp_ns=1_000_000_000,
    )
    assert not success
    assert reason is not None
