"""Tests for kitscenes.frame — Frame accessor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kitscenes.frame import Frame
from kitscenes.sensors import SensorDataLoader


def _make_scene_dir(tmp_path: Path) -> Path:
    scene = tmp_path / "scene"
    scene.mkdir()
    (scene / "timestamp.reference.txt").write_text("1000\n2000\n")
    (scene / "poses.txt").write_text(
        "1.000000000 0 0 0 0 0 0 1\n"
        "2.000000000 1 0 0 0 0 0 1\n"
    )
    return scene


def test_frame_lidar_returns_none_when_missing(tmp_path: Path) -> None:
    scene = _make_scene_dir(tmp_path)
    loader = SensorDataLoader(scene)
    frame = Frame(
        scene_id="scene",
        frame_idx=0,
        timestamp_ns=1000,
        loader=loader,
        ego_poses=(),
    )
    assert frame.lidar("lidar_top") is None


def test_frame_ego_pose_indexing(tmp_path: Path) -> None:
    from kitscenes.schema import EgoPose

    scene = _make_scene_dir(tmp_path)
    loader = SensorDataLoader(scene)
    poses = (
        EgoPose(1000, np.zeros(3), np.array([0, 0, 0, 1], dtype=np.float64)),
        EgoPose(2000, np.ones(3), np.array([0, 0, 0, 1], dtype=np.float64)),
    )
    frame = Frame(
        scene_id="scene",
        frame_idx=1,
        timestamp_ns=2000,
        loader=loader,
        ego_poses=poses,
    )
    assert frame.ego_pose is not None
    np.testing.assert_array_equal(frame.ego_pose.translation, np.ones(3))
