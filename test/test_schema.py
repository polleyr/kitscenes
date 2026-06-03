"""Tests for kitscenes.schema — dataclasses."""

import numpy as np
import pytest

from kitscenes.schema import (
    EgoPose,
    Scene,
    SceneMetadata,
)
from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures — reusable building blocks for tests
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_gnss_ego_pose() -> EgoPose:
    """A minimal valid EgoPose."""
    return EgoPose(
        timestamp_ns=1_000_000_000,
        translation=np.array([100.0, 200.0, 0.5], dtype=np.float64),
        rotation=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
    )


@pytest.fixture
def sample_metadata() -> SceneMetadata:
    """A minimal valid SceneMetadata."""
    return SceneMetadata(
        num_frames=100,
        duration_s=10.0,
        sensor_names=("lidar_top", "camera_ring_front"),
    )


@pytest.fixture
def sample_scene(
    sample_gnss_ego_pose: EgoPose,
    sample_metadata: SceneMetadata,
) -> Scene:
    """A minimal valid Scene composed from other fixtures."""
    return Scene(
        scene_id="scene_001",
        scene_path=Path("/data/kitscenes/scene_001"),
        timestamps_ns=np.arange(100, dtype=np.int64) * 100_000_000,
        metadata=sample_metadata,
        _ego_loader=lambda: (sample_gnss_ego_pose,),
        _map_loader=lambda: None,
    )


# ---------------------------------------------------------------------------
# EgoPose tests
# ---------------------------------------------------------------------------


class TestEgoPose:
    """Verify EgoPose construction and immutability."""

    def test_field_access(self, sample_gnss_ego_pose: EgoPose) -> None:
        assert sample_gnss_ego_pose.timestamp_ns == 1_000_000_000
        np.testing.assert_array_equal(
            sample_gnss_ego_pose.translation, [100.0, 200.0, 0.5]
        )

    def test_rotation_shape(self, sample_gnss_ego_pose: EgoPose) -> None:
        """Rotation quaternion must be a 4-element array."""
        assert sample_gnss_ego_pose.rotation.shape == (4,)

    def test_frozen(self, sample_gnss_ego_pose: EgoPose) -> None:
        with pytest.raises(AttributeError):
            sample_gnss_ego_pose.timestamp_ns = 0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SceneMetadata tests
# ---------------------------------------------------------------------------


class TestSceneMetadata:
    """Verify SceneMetadata construction."""

    def test_field_access(self, sample_metadata: SceneMetadata) -> None:
        assert sample_metadata.num_frames == 100
        assert sample_metadata.duration_s == 10.0
        assert "lidar_top" in sample_metadata.sensor_names

    def test_sensor_names_is_tuple(self, sample_metadata: SceneMetadata) -> None:
        assert isinstance(sample_metadata.sensor_names, tuple)


# ---------------------------------------------------------------------------
# Scene tests
# ---------------------------------------------------------------------------


class TestScene:
    """Verify Scene construction and field access."""

    def test_field_access(self, sample_scene: Scene) -> None:
        assert sample_scene.scene_id == "scene_001"
        assert sample_scene.scene_path == Path("/data/kitscenes/scene_001")
        assert len(sample_scene.ego_poses) == 1

    def test_timestamps_shape(self, sample_scene: Scene) -> None:
        """Timestamps must match num_frames from metadata."""
        assert sample_scene.timestamps_ns.shape == (100,)
        assert sample_scene.timestamps_ns.dtype == np.int64

    def test_timestamp_s(self, sample_scene: Scene) -> None:
        """Scene.timestamp_s should expose the full reference timeline in seconds."""
        timestamp_s = sample_scene.timestamp_s

        assert isinstance(timestamp_s, np.ndarray)
        assert timestamp_s.shape == (100,)
        assert timestamp_s.dtype == np.float64
        np.testing.assert_allclose(timestamp_s[:3], [0.0, 0.1, 0.2])

    def test_frozen(self, sample_scene: Scene) -> None:
        with pytest.raises(AttributeError):
            sample_scene.scene_id = "other"  # type: ignore[misc]

    def test_ego_poses_is_tuple(self, sample_scene: Scene) -> None:
        assert isinstance(sample_scene.ego_poses, tuple)
