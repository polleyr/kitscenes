"""Tests for kitscenes.dataset — KITScenesDataset loader."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from kitscenes.constants import KITSCENES_ROOT_ENV_VAR
from kitscenes.dataset import KITScenesDataset
from kitscenes.schema import EgoPose, Scene, SceneMetadata

# Integration tests require $KITSCENES_ROOT — no default path.
_REAL_DATASET_ROOT = os.environ.get(KITSCENES_ROOT_ENV_VAR)


# ---------------------------------------------------------------------------
# Helper — build a synthetic dataset directory tree
# ---------------------------------------------------------------------------

_START_NS = 1700000000_000000000
_STEP_NS = 100_000_000  # 100 ms = 10 Hz


def _create_fake_scene(
    scene_dir: Path,
    num_timestamps: int = 5,
    write_poses: bool = True,
    sensor_dirs: tuple[str, ...] = ("lidar_top", "camera_ring_front"),
) -> None:
    """Populate a scene directory with minimal valid files.

    Creates:
      - timestamp.reference.txt  with ``num_timestamps`` lines
      - poses.txt  with TUM-format ego poses (one per timestamp)
      - subdirectories for each sensor in ``sensor_dirs``
    """
    scene_dir.mkdir(parents=True, exist_ok=True)

    # --- Reference timestamps (nanosecond integers, one per line) -----------
    timestamps_ns = [_START_NS + i * _STEP_NS for i in range(num_timestamps)]
    ts_file = scene_dir / "timestamp.reference.txt"
    ts_file.write_text("\n".join(str(t) for t in timestamps_ns) + "\n")

    # --- TUM ego poses: "timestamp_s tx ty tz qx qy qz qw" -----------------
    if write_poses:
        lines = []
        for i, t_ns in enumerate(timestamps_ns):
            ts_s = t_ns / 1e9
            tx, ty, tz = float(i), float(i) * 2, 0.5
            lines.append(f"{ts_s} {tx} {ty} {tz} 0.0 0.0 0.0 1.0")
        (scene_dir / "poses.txt").write_text("\n".join(lines) + "\n")

    # --- Sensor directories -------------------------------------------------
    for sensor in sensor_dirs:
        (scene_dir / sensor).mkdir(parents=True, exist_ok=True)


def _create_fake_dataset(
    root: Path,
    scene_names: tuple[str, ...] = ("scene_001", "scene_002"),
    **scene_kwargs,
) -> Path:
    """Create a complete fake dataset with multiple scenes."""
    root.mkdir(parents=True, exist_ok=True)
    for name in scene_names:
        _create_fake_scene(root / name, **scene_kwargs)
    return root


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_dataset(tmp_path: Path) -> Path:
    """A two-scene fake dataset in a temporary directory."""
    return _create_fake_dataset(tmp_path / "dataset")


@pytest.fixture
def env_clean(monkeypatch):
    """Ensure KITSCENES_ROOT is unset for the duration of a test."""
    monkeypatch.delenv(KITSCENES_ROOT_ENV_VAR, raising=False)


# ---------------------------------------------------------------------------
# Root resolution tests
# ---------------------------------------------------------------------------


class TestRootResolution:
    """Verify the three-tier root resolution: arg > env > error."""

    def test_explicit_root(self, fake_dataset: Path) -> None:
        """An explicit root path takes priority."""
        ds = KITScenesDataset(root=fake_dataset)
        assert ds.root == fake_dataset

    def test_env_variable_fallback(
        self, fake_dataset: Path, monkeypatch
    ) -> None:
        """When root is omitted, fall back to $KITSCENES_ROOT."""
        monkeypatch.setenv(KITSCENES_ROOT_ENV_VAR, str(fake_dataset))
        ds = KITScenesDataset()
        assert ds.root == fake_dataset

    def test_explicit_overrides_env(
        self, fake_dataset: Path, tmp_path: Path, monkeypatch
    ) -> None:
        """Explicit root wins even if the env var is also set."""
        other = _create_fake_dataset(tmp_path / "other")
        monkeypatch.setenv(KITSCENES_ROOT_ENV_VAR, str(other))
        ds = KITScenesDataset(root=fake_dataset)
        assert ds.root == fake_dataset

    def test_no_root_no_env_raises(self, env_clean) -> None:
        """Must raise ValueError when neither root nor env var is provided."""
        with pytest.raises(ValueError, match="KITSCENES_ROOT"):
            KITScenesDataset()

    def test_nonexistent_root_raises(self, env_clean) -> None:
        """Must raise FileNotFoundError for a path that does not exist."""
        with pytest.raises(FileNotFoundError):
            KITScenesDataset(root="/this/path/does/not/exist")


# ---------------------------------------------------------------------------
# Scene discovery tests
# ---------------------------------------------------------------------------


class TestSceneDiscovery:
    """Verify that scenes are discovered from subdirectories."""

    def test_scene_ids_sorted(self, fake_dataset: Path) -> None:
        """scene_ids must be sorted alphabetically."""
        ds = KITScenesDataset(root=fake_dataset)
        assert ds.scene_ids == ["scene_001", "scene_002"]

    def test_len(self, fake_dataset: Path) -> None:
        ds = KITScenesDataset(root=fake_dataset)
        assert len(ds) == 2

    def test_files_ignored(self, fake_dataset: Path) -> None:
        """Regular files in the root must not be treated as scenes."""
        (fake_dataset / "README.txt").write_text("not a scene")
        ds = KITScenesDataset(root=fake_dataset)
        assert "README.txt" not in ds.scene_ids
        assert len(ds) == 2

    def test_empty_root(self, tmp_path: Path) -> None:
        """An empty root directory yields zero scenes."""
        empty = tmp_path / "empty"
        empty.mkdir()
        ds = KITScenesDataset(root=empty)
        assert ds.scene_ids == []
        assert len(ds) == 0

    def test_hf_data_layout(self, tmp_path: Path) -> None:
        """Scenes under root/data/<split>/ are discovered (HF download layout)."""
        root = tmp_path / "kitscenes"
        train = root / "data" / "train"
        _create_fake_dataset(train, ("scene_001", "scene_002"))
        ds = KITScenesDataset(root=root)
        assert ds.scene_ids == ["scene_001", "scene_002"]
        assert ds.get_scene("scene_001").scene_path == train / "scene_001"


# ---------------------------------------------------------------------------
# Scene loading tests
# ---------------------------------------------------------------------------


class TestGetScene:
    """Verify that get_scene loads scene data correctly."""

    def test_basic_load(self, fake_dataset: Path) -> None:
        """A scene loaded from the fake dataset has expected fields."""
        ds = KITScenesDataset(root=fake_dataset)
        scene = ds.get_scene("scene_001")

        assert isinstance(scene, Scene)
        assert scene.scene_id == "scene_001"
        assert scene.scene_path == fake_dataset / "scene_001"

    def test_unknown_scene_raises(self, fake_dataset: Path) -> None:
        """Requesting a non-existent scene must raise KeyError."""
        ds = KITScenesDataset(root=fake_dataset)
        with pytest.raises(KeyError, match="no_such_scene"):
            ds.get_scene("no_such_scene")

    def test_timestamps_loaded(self, fake_dataset: Path) -> None:
        """Reference timestamps must be loaded as int64 nanoseconds."""
        ds = KITScenesDataset(root=fake_dataset)
        scene = ds.get_scene("scene_001")

        assert scene.timestamps_ns.dtype == np.int64
        assert len(scene.timestamps_ns) == 5
        assert np.all(np.diff(scene.timestamps_ns) > 0)

    def test_ego_poses_loaded(self, fake_dataset: Path) -> None:
        """Ego poses must be loaded from poses.txt."""
        ds = KITScenesDataset(root=fake_dataset)
        scene = ds.get_scene("scene_001")

        assert len(scene.ego_poses) == 5
        for pose in scene.ego_poses:
            assert isinstance(pose, EgoPose)
            assert pose.translation.shape == (3,)
            assert pose.rotation.shape == (4,)

    def test_ego_pose_values(self, fake_dataset: Path) -> None:
        """Verify that parsed ego pose values match what we wrote."""
        ds = KITScenesDataset(root=fake_dataset)
        scene = ds.get_scene("scene_001")

        first = scene.ego_poses[0]
        np.testing.assert_array_almost_equal(first.translation, [0.0, 0.0, 0.5])
        np.testing.assert_array_almost_equal(first.rotation, [0.0, 0.0, 0.0, 1.0])

        second = scene.ego_poses[1]
        np.testing.assert_array_almost_equal(second.translation, [1.0, 2.0, 0.5])

    def test_metadata(self, fake_dataset: Path) -> None:
        """SceneMetadata must reflect the loaded scene."""
        ds = KITScenesDataset(root=fake_dataset)
        scene = ds.get_scene("scene_001")

        assert isinstance(scene.metadata, SceneMetadata)
        assert scene.metadata.num_frames == 5
        assert scene.metadata.duration_s == pytest.approx(0.4, abs=1e-6)
        assert "lidar_top" in scene.metadata.sensor_names
        assert "camera_ring_front" in scene.metadata.sensor_names

    def test_sensors_discovered(self, fake_dataset: Path) -> None:
        """Only sensors with existing directories should appear."""
        ds = KITScenesDataset(root=fake_dataset)
        scene = ds.get_scene("scene_001")
        assert set(scene.metadata.sensor_names) == {"lidar_top", "camera_ring_front"}

# ---------------------------------------------------------------------------
# Missing file graceful handling tests
# ---------------------------------------------------------------------------


class TestMissingFiles:
    """Verify that missing files don't crash, just return empty data."""

    def test_no_timestamp_file(self, tmp_path: Path) -> None:
        """A scene without timestamp.reference.txt gets an empty array."""
        root = tmp_path / "ds"
        scene_dir = root / "scene_no_ts"
        scene_dir.mkdir(parents=True)

        ds = KITScenesDataset(root=root)
        scene = ds.get_scene("scene_no_ts")
        assert len(scene.timestamps_ns) == 0

    def test_no_pose_file(self, tmp_path: Path) -> None:
        """A scene without poses.txt gets an empty pose tuple."""
        root = tmp_path / "ds"
        _create_fake_scene(
            root / "scene_no_poses",
            num_timestamps=3,
            write_poses=False,
        )

        ds = KITScenesDataset(root=root)
        scene = ds.get_scene("scene_no_poses")
        assert len(scene.ego_poses) == 0
        assert len(scene.timestamps_ns) == 3


# ---------------------------------------------------------------------------
# Container protocol tests
# ---------------------------------------------------------------------------


class TestContainerProtocol:
    """Verify __iter__, __getitem__, and __repr__."""

    def test_iter(self, fake_dataset: Path) -> None:
        """Iterating yields one Scene per scene_id."""
        ds = KITScenesDataset(root=fake_dataset)
        scenes = list(ds)
        assert len(scenes) == 2
        assert scenes[0].scene_id == "scene_001"
        assert scenes[1].scene_id == "scene_002"

    def test_getitem_int(self, fake_dataset: Path) -> None:
        """Integer indexing returns scenes in sorted order."""
        ds = KITScenesDataset(root=fake_dataset)
        assert ds[0].scene_id == "scene_001"
        assert ds[1].scene_id == "scene_002"

    def test_getitem_str(self, fake_dataset: Path) -> None:
        """String indexing returns the matching scene."""
        ds = KITScenesDataset(root=fake_dataset)
        assert ds["scene_002"].scene_id == "scene_002"

    def test_repr(self, fake_dataset: Path) -> None:
        """__repr__ must include the root path and scene count."""
        ds = KITScenesDataset(root=fake_dataset)
        r = repr(ds)
        assert "num_scenes=2" in r
        assert str(fake_dataset) in r


# ---------------------------------------------------------------------------
# Integration test (only runs when real dataset is mounted)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    _REAL_DATASET_ROOT is None,
    reason=f"${KITSCENES_ROOT_ENV_VAR} is not set",
)
@pytest.mark.skipif(
    _REAL_DATASET_ROOT is not None
    and (
        not Path(_REAL_DATASET_ROOT).is_dir()
        or not any(Path(_REAL_DATASET_ROOT).iterdir())
    ),
    reason=f"${KITSCENES_ROOT_ENV_VAR} does not point to a mounted dataset",
)
class TestRealDataset:
    """Smoke tests against the real dataset (requires $KITSCENES_ROOT)."""

    @property
    def root(self) -> Path:
        assert _REAL_DATASET_ROOT is not None
        return Path(_REAL_DATASET_ROOT)

    def test_discover_scenes(self) -> None:
        ds = KITScenesDataset(root=self.root)
        assert len(ds) > 0, "Expected at least one scene"
        print(f"Found {len(ds)} scenes: {ds.scene_ids[:5]}...")

    def test_load_first_scene(self) -> None:
        ds = KITScenesDataset(root=self.root)
        scene = ds[0]
        assert scene.scene_id
        assert len(scene.timestamps_ns) > 0
        print(
            f"Scene {scene.scene_id}: "
            f"{scene.metadata.num_frames} frames, "
            f"{scene.metadata.duration_s:.1f}s, "
            f"sensors={scene.metadata.sensor_names}"
        )

    def test_lidar_sweep_filters_invalid_points(self) -> None:
        ds = KITScenesDataset(root=self.root, split="val")
        assert len(ds) > 0, "Expected at least one val scene under $KITSCENES_ROOT"
        loader = ds.get_sensor_loader(ds.scene_ids[0])
        sweep = loader.get_lidar_sweep("lidar_top", 0)
        pts = sweep.points
        assert len(pts) > 0
        invalid = np.isclose(pts["reflectivity"], -1.0) & (
            np.isclose(pts["x"], 0.0)
            & np.isclose(pts["y"], 0.0)
            & np.isclose(pts["z"], 0.0)
        )
        assert not np.any(invalid)
