"""Tests for kitscenes.sensors — SensorDataLoader and data containers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kitscenes.poses import estimate_ego_motion
from kitscenes.sensors import (
    CameraCalibration,
    LidarSweep,
    RadarSweep,
    SensorDataLoader,
)

# ---------------------------------------------------------------------------
# Helpers — build synthetic scene data
# ---------------------------------------------------------------------------

_START_NS = 1700000000_000000000
_STEP_NS = 100_000_000  # 100 ms


def _write_tum_poses(
    scene_dir: Path,
    positions: tuple[tuple[float, float, float], ...],
    yaws_rad: tuple[float, ...] | None = None,
    start_ns: int = _START_NS,
    step_ns: int = _STEP_NS,
) -> None:
    """Write a scene-root poses.txt file matching the synthetic timestamps."""
    if yaws_rad is None:
        yaws_rad = tuple(0.0 for _ in positions)
    if len(yaws_rad) != len(positions):
        raise ValueError("positions and yaws_rad must have the same length")

    lines: list[str] = []
    for idx, ((x, y, z), yaw) in enumerate(zip(positions, yaws_rad)):
        ts_s = (start_ns + idx * step_ns) / 1e9
        qx = 0.0
        qy = 0.0
        qz = np.sin(yaw / 2.0)
        qw = np.cos(yaw / 2.0)
        lines.append(f"{ts_s:.9f} {x:.6f} {y:.6f} {z:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}")
    (scene_dir / "poses.txt").write_text("\n".join(lines) + "\n")


def _write_timestamp_file(path: Path, count: int, start_ns: int = _START_NS) -> None:
    """Write *count* nanosecond-integer timestamps, one per line."""
    lines = [str(start_ns + i * _STEP_NS) for i in range(count)]
    path.write_text("\n".join(lines) + "\n")


def _make_calib_json(scene_dir: Path) -> None:
    """Write a minimal ``calibration/calib.json`` with one camera and one lidar."""
    calib = {
        "camera_ring_front_pinhole": {
            "T_to_reference": np.eye(4).tolist(),
            "intrinsics": {
                "focal_length": 1000.0,
                "principal_point_u": 960.0,
                "principal_point_v": 540.0,
            },
            "resolution": {
                "width": 1920,
                "height": 1080,
            },
        },
        "lidar_top": {
            "T_to_reference": np.eye(4).tolist(),
        },
    }
    calib_dir = scene_dir / "calibration"
    calib_dir.mkdir(parents=True, exist_ok=True)
    (calib_dir / "calib.json").write_text(json.dumps(calib))


def _make_camera_images(
    scene_dir: Path,
    camera_name: str = "camera_ring_front",
    frame_indices: tuple[int, ...] = (0, 1, 2),
    width: int = 64,
    height: int = 48,
) -> None:
    """Write tiny JPEG images for the given frame indices."""
    cam_dir = scene_dir / camera_name
    cam_dir.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image

        for idx in frame_indices:
            img = Image.new("RGB", (width, height), color=(idx % 256, 128, 64))
            img.save(cam_dir / f"{idx:010d}.jpg")
    except ImportError:
        # Fallback: write a dummy file (image loading tests will be skipped)
        for idx in frame_indices:
            (cam_dir / f"{idx:010d}.jpg").write_bytes(b"\xff\xd8\xff\xe0")


def _make_lidar_parquets(
    scene_dir: Path,
    lidar_name: str = "lidar_top",
    frame_indices: tuple[int, ...] = (0, 1, 2),
    num_points: int = 10,
) -> None:
    """Write minimal lidar Parquet files with correct metadata."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    lidar_dir = scene_dir / lidar_name
    lidar_dir.mkdir(parents=True, exist_ok=True)

    resolution = 0.005
    for idx in frame_indices:
        # Random integer-discretized xyz + valid reflectivity
        rng = np.random.RandomState(idx)
        x = rng.randint(-1000, 1000, num_points, dtype=np.int32)
        y = rng.randint(-1000, 1000, num_points, dtype=np.int32)
        z = rng.randint(-100, 100, num_points, dtype=np.int32)
        columns = {
            "x": x,
            "y": y,
            "z": z,
            "reflectivity": rng.uniform(0, 1, num_points).astype(np.float32),
            "timestamp": np.full(num_points, float(idx), dtype=np.float64),
            "ring": np.zeros(num_points, dtype=np.uint8),
            "sensor_id": np.zeros(num_points, dtype=np.uint8),
            "sensor_specific_data": np.zeros(num_points, dtype=np.uint16),
        }
        table = pa.Table.from_pydict(columns)
        metadata = {
            "discretization_resolution": str(resolution),
            "point_type_name": "PointLGDataset",
            "pcl_width": str(num_points),
            "pcl_height": "1",
        }
        table = table.cast(table.schema.with_metadata(metadata))
        pq.write_table(
            table,
            lidar_dir / f"{idx:010d}.parquet",
            compression="zstd",
        )


def _make_radar_parquets(
    scene_dir: Path,
    radar_name: str = "radar_front",
    frame_indices: tuple[int, ...] = (0, 1, 2),
    num_points: int = 5,
    point_overrides: dict[int, dict[str, np.ndarray]] | None = None,
) -> None:
    """Write minimal radar Parquet files with correct metadata."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    radar_dir = scene_dir / radar_name
    radar_dir.mkdir(parents=True, exist_ok=True)

    for idx in frame_indices:
        rng = np.random.RandomState(idx + 1000)
        columns = {
            "x": rng.uniform(-10, 10, num_points).astype(np.float32),
            "y": rng.uniform(-10, 10, num_points).astype(np.float32),
            "z": rng.uniform(-1, 1, num_points).astype(np.float32),
            "azimuth": rng.uniform(-3, 3, num_points).astype(np.float32),
            "azimuth_std": np.zeros(num_points, dtype=np.float32),
            "elevation": rng.uniform(-1, 1, num_points).astype(np.float32),
            "elevation_std": np.zeros(num_points, dtype=np.float32),
            "range": rng.uniform(0, 100, num_points).astype(np.float32),
            "range_std": np.zeros(num_points, dtype=np.float32),
            "range_rate": rng.uniform(-5, 5, num_points).astype(np.float32),
            "range_rate_std": np.zeros(num_points, dtype=np.float32),
            "rcs": rng.uniform(-10, 30, num_points).astype(np.float32),
            "timestamp": np.full(num_points, float(idx), dtype=np.float64),
            "sensor_id": np.zeros(num_points, dtype=np.uint8),
            "detection_id": np.arange(num_points, dtype=np.uint16),
            "object_id": np.zeros(num_points, dtype=np.uint16),
            "classification": np.zeros(num_points, dtype=np.uint8),
            "existence_probability": np.ones(num_points, dtype=np.float32),
            "resolved_velocity_probability": np.ones(num_points, dtype=np.float32),
            "multi_target_probability": np.zeros(num_points, dtype=np.float32),
        }
        if point_overrides and idx in point_overrides:
            columns.update(point_overrides[idx])
        table = pa.Table.from_pydict(columns)
        metadata = {
            "discretization_resolution": "0.005",
            "point_type_name": "PointLGDatasetRadar",
            "pcl_width": str(num_points),
            "pcl_height": "1",
        }
        table = table.cast(table.schema.with_metadata(metadata))
        pq.write_table(
            table,
            radar_dir / f"{idx:010d}.parquet",
            compression="zstd",
        )


def _create_scene(
    scene_dir: Path,
    num_frames: int = 3,
    cameras: tuple[str, ...] = ("camera_ring_front",),
    lidars: tuple[str, ...] = ("lidar_top",),
    radars: tuple[str, ...] = ("radar_front",),
) -> Path:
    """Build a complete synthetic scene directory."""
    scene_dir.mkdir(parents=True, exist_ok=True)
    frame_indices = tuple(range(num_frames))

    # Timestamps
    _write_timestamp_file(
        scene_dir / "timestamp.reference.txt", num_frames,
    )
    for sensor in list(cameras) + list(lidars) + list(radars):
        _write_timestamp_file(
            scene_dir / f"timestamp.{sensor}.txt", num_frames,
        )

    # Calibration
    _make_calib_json(scene_dir)
    _write_tum_poses(
        scene_dir,
        positions=tuple((float(i), 0.0, 0.0) for i in frame_indices),
    )

    # Sensor data
    for cam in cameras:
        _make_camera_images(scene_dir, cam, frame_indices)
    for lid in lidars:
        _make_lidar_parquets(scene_dir, lid, frame_indices)
    for rad in radars:
        _make_radar_parquets(scene_dir, rad, frame_indices)

    return scene_dir


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def scene_dir(tmp_path: Path) -> Path:
    """A synthetic scene with 3 frames, 1 camera, 1 lidar, 1 radar."""
    return _create_scene(tmp_path / "scene_001")


@pytest.fixture
def loader(scene_dir: Path) -> SensorDataLoader:
    return SensorDataLoader(scene_dir)


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_valid_path(self, scene_dir: Path) -> None:
        loader = SensorDataLoader(scene_dir)
        assert loader.scene_path == scene_dir

    def test_string_path(self, scene_dir: Path) -> None:
        loader = SensorDataLoader(str(scene_dir))
        assert loader.scene_path == scene_dir

    def test_nonexistent_path_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            SensorDataLoader(Path("/nonexistent/path"))


# ---------------------------------------------------------------------------
# Calibration tests
# ---------------------------------------------------------------------------


class TestCalibration:
    def test_camera_calibration_loads(self, loader: SensorDataLoader) -> None:
        calib = loader.get_camera_calibration("camera_ring_front")
        assert isinstance(calib, CameraCalibration)
        assert calib.sensor_name == "camera_ring_front"
        assert calib.intrinsic.shape == (3, 3)
        assert calib.extrinsic.shape == (4, 4)
        assert calib.image_size == (1920, 1080)

    def test_camera_intrinsic_values(self, loader: SensorDataLoader) -> None:
        calib = loader.get_camera_calibration("camera_ring_front")
        assert calib.intrinsic[0, 0] == 1000.0  # focal_length
        assert calib.intrinsic[1, 1] == 1000.0  # focal_length
        assert calib.intrinsic[0, 2] == 960.0   # principal_point_u
        assert calib.intrinsic[1, 2] == 540.0   # principal_point_v
        assert calib.intrinsic[2, 2] == 1.0

    def test_camera_extrinsic_is_identity(self, loader: SensorDataLoader) -> None:
        calib = loader.get_camera_calibration("camera_ring_front")
        np.testing.assert_array_equal(calib.extrinsic, np.eye(4))

    def test_unknown_camera_raises(self, loader: SensorDataLoader) -> None:
        with pytest.raises(KeyError, match="no_such_camera"):
            loader.get_camera_calibration("no_such_camera")

    def test_lidar_extrinsic(self, loader: SensorDataLoader) -> None:
        ext = loader.get_lidar_extrinsic("lidar_top")
        assert ext.shape == (4, 4)
        np.testing.assert_array_equal(ext, np.eye(4))

    def test_unknown_lidar_raises(self, loader: SensorDataLoader) -> None:
        with pytest.raises(KeyError):
            loader.get_lidar_extrinsic("lidar_nonexistent")

    def test_missing_calib_file(self, tmp_path: Path) -> None:
        """A scene without calib.json returns empty calibration data."""
        scene = tmp_path / "no_calib"
        scene.mkdir()
        loader = SensorDataLoader(scene)
        with pytest.raises(KeyError):
            loader.get_camera_calibration("camera_ring_front")


# ---------------------------------------------------------------------------
# Timestamp tests
# ---------------------------------------------------------------------------


class TestTimestamps:
    def test_reference_timestamps(self, loader: SensorDataLoader) -> None:
        ts = loader.get_reference_timestamps()
        assert ts.dtype == np.int64
        assert len(ts) == 3
        assert np.all(np.diff(ts) > 0)

    def test_sensor_timestamps(self, loader: SensorDataLoader) -> None:
        ts = loader.get_sensor_timestamps("lidar_top")
        assert ts.dtype == np.int64
        assert len(ts) == 3

    def test_missing_sensor_timestamps_raises(
        self, loader: SensorDataLoader,
    ) -> None:
        with pytest.raises(FileNotFoundError):
            loader.get_sensor_timestamps("nonexistent_sensor")


# ---------------------------------------------------------------------------
# Sensor discovery tests
# ---------------------------------------------------------------------------


class TestSensorDiscovery:
    def test_camera_names(self, loader: SensorDataLoader) -> None:
        names = loader.get_camera_names()
        assert "camera_ring_front" in names

    def test_lidar_names(self, loader: SensorDataLoader) -> None:
        names = loader.get_lidar_names()
        assert "lidar_top" in names

    def test_radar_names(self, loader: SensorDataLoader) -> None:
        names = loader.get_radar_names()
        assert "radar_front" in names

    def test_all_sensor_names(self, loader: SensorDataLoader) -> None:
        names = loader.get_all_sensor_names()
        assert "camera_ring_front" in names
        assert "lidar_top" in names
        assert "radar_front" in names

    def test_empty_scene(self, tmp_path: Path) -> None:
        scene = tmp_path / "empty_scene"
        scene.mkdir()
        loader = SensorDataLoader(scene)
        assert loader.get_camera_names() == []
        assert loader.get_lidar_names() == []
        assert loader.get_radar_names() == []


# ---------------------------------------------------------------------------
# Frame index listing tests
# ---------------------------------------------------------------------------


class TestFrameIndices:
    def test_camera_frame_indices(self, loader: SensorDataLoader) -> None:
        indices = loader.get_frame_indices("camera_ring_front")
        assert indices == [0, 1, 2]

    def test_lidar_frame_indices(self, loader: SensorDataLoader) -> None:
        indices = loader.get_frame_indices("lidar_top")
        assert indices == [0, 1, 2]

    def test_radar_frame_indices(self, loader: SensorDataLoader) -> None:
        indices = loader.get_frame_indices("radar_front")
        assert indices == [0, 1, 2]

    def test_nonexistent_sensor(self, loader: SensorDataLoader) -> None:
        assert loader.get_frame_indices("no_such_sensor") == []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _has_pil() -> bool:
    try:
        import PIL  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Camera image tests
# ---------------------------------------------------------------------------


class TestCameraImages:
    def test_image_path(self, loader: SensorDataLoader) -> None:
        path = loader.get_camera_image_path("camera_ring_front", 0)
        assert path.exists()
        assert path.name == "0000000000.jpg"

    def test_image_path_missing_raises(self, loader: SensorDataLoader) -> None:
        with pytest.raises(FileNotFoundError):
            loader.get_camera_image_path("camera_ring_front", 999)

    @pytest.mark.skipif(
        not _has_pil(), reason="Pillow not installed",
    )
    def test_load_image(self, loader: SensorDataLoader) -> None:
        img = loader.get_camera_image("camera_ring_front", 0)
        assert img.dtype == np.uint8
        assert img.ndim == 3
        assert img.shape[2] == 3  # RGB

    @pytest.mark.skipif(
        not _has_pil(), reason="Pillow not installed",
    )
    def test_image_size(self, loader: SensorDataLoader) -> None:
        w, h = loader.get_camera_image_size("camera_ring_front", 0)
        assert w == 64
        assert h == 48


# ---------------------------------------------------------------------------
# LiDAR sweep tests
# ---------------------------------------------------------------------------


class TestLidarSweep:
    def test_load_sweep(self, loader: SensorDataLoader) -> None:
        sweep = loader.get_lidar_sweep("lidar_top", 0)
        assert isinstance(sweep, LidarSweep)
        assert sweep.sensor_name == "lidar_top"
        assert sweep.points.dtype.names is not None  # structured array
        assert "x" in sweep.points.dtype.names
        assert "y" in sweep.points.dtype.names
        assert "z" in sweep.points.dtype.names
        assert len(sweep.points) > 0

    def test_sweep_timestamp(self, loader: SensorDataLoader) -> None:
        sweep = loader.get_lidar_sweep("lidar_top", 0)
        assert sweep.timestamp_ns == _START_NS

    def test_missing_frame_raises(self, loader: SensorDataLoader) -> None:
        with pytest.raises(FileNotFoundError):
            loader.get_lidar_sweep("lidar_top", 999)


# ---------------------------------------------------------------------------
# Radar sweep tests
# ---------------------------------------------------------------------------


class TestRadarSweep:
    def test_load_sweep(self, loader: SensorDataLoader) -> None:
        sweep = loader.get_radar_sweep("radar_front", 0)
        assert isinstance(sweep, RadarSweep)
        assert sweep.sensor_name == "radar_front"
        assert sweep.points.dtype.names is not None
        assert "x" in sweep.points.dtype.names
        assert len(sweep.points) > 0

    def test_sweep_timestamp(self, loader: SensorDataLoader) -> None:
        sweep = loader.get_radar_sweep("radar_front", 0)
        assert sweep.timestamp_ns == _START_NS

    def test_missing_frame_raises(self, loader: SensorDataLoader) -> None:
        with pytest.raises(FileNotFoundError):
            loader.get_radar_sweep("radar_front", 999)

    def test_raw_points_remain_accessible(self, loader: SensorDataLoader) -> None:
        sweep = loader.get_radar_sweep("radar_front", 1)
        assert sweep.raw().dtype == sweep.points.dtype
        assert len(sweep.raw()) == len(sweep.points)

    def test_linear_ego_motion_compensates_range_rate(self, tmp_path: Path) -> None:
        scene_dir = _create_scene(tmp_path / "scene_radar_linear")
        _make_radar_parquets(
            scene_dir,
            radar_name="radar_front",
            frame_indices=(0, 1, 2),
            num_points=1,
            point_overrides={
                1: {
                    "x": np.array([50.0], dtype=np.float32),
                    "y": np.array([0.0], dtype=np.float32),
                    "z": np.array([0.0], dtype=np.float32),
                    "azimuth": np.array([0.0], dtype=np.float32),
                    "elevation": np.array([0.0], dtype=np.float32),
                    "range": np.array([50.0], dtype=np.float32),
                    "range_rate": np.array([-10.0], dtype=np.float32),
                },
            },
        )
        loader = SensorDataLoader(scene_dir)

        sweep = loader.get_radar_sweep("radar_front", 1)

        assert sweep.ego_motion_compensated is True
        np.testing.assert_allclose(sweep.raw()["range_rate"], [-10.0], atol=1e-6)
        np.testing.assert_allclose(sweep.points["range_rate"], [0.0], atol=1e-5)

    def test_missing_poses_softly_falls_back_to_raw(self, tmp_path: Path) -> None:
        scene_dir = _create_scene(tmp_path / "scene_radar_raw_fallback")
        (scene_dir / "poses.txt").unlink()
        loader = SensorDataLoader(scene_dir)

        sweep = loader.get_radar_sweep("radar_front", 1)

        assert sweep.ego_motion_compensated is False
        assert sweep.compensation_reason == "missing or insufficient poses.txt data"
        np.testing.assert_array_equal(sweep.points, sweep.raw())




# ---------------------------------------------------------------------------
# Sync alignment tests
# ---------------------------------------------------------------------------


