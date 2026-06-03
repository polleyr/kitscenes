"""Tests for kitscenes.frame_dataset — FrameDataset."""

from __future__ import annotations

from pathlib import Path

import pytest

from kitscenes.dataset import KITScenesDataset
from kitscenes.frame_dataset import FrameDataset


def _create_fake_scene(root: Path, scene_id: str, num_frames: int = 3) -> None:
    scene = root / scene_id
    scene.mkdir(parents=True)
    start_ns = 1_700_000_000_000_000_000
    step_ns = 100_000_000
    lines = [str(start_ns + i * step_ns) for i in range(num_frames)]
    (scene / "timestamp.reference.txt").write_text("\n".join(lines) + "\n")


def test_frame_dataset_length(tmp_path: Path) -> None:
    root = tmp_path / "kitscenes"
    _create_fake_scene(root, "scene_a", num_frames=2)
    _create_fake_scene(root, "scene_b", num_frames=3)
    ds = KITScenesDataset(root=root)
    frames = FrameDataset(ds)
    assert len(frames) == 5


def test_frame_dataset_getitem(tmp_path: Path) -> None:
    root = tmp_path / "kitscenes"
    _create_fake_scene(root, "scene_a", num_frames=2)
    ds = KITScenesDataset(root=root)
    frames = FrameDataset(ds)
    frame = frames[0]
    assert frame.scene_id == "scene_a"
    assert frame.frame_idx == 0
    assert frame.timestamp_ns == 1_700_000_000_000_000_000
