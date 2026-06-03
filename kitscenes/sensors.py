"""Sensor data loading for the kitscenes API.

Provides :class:`SensorDataLoader` for accessing camera images, LiDAR sweeps,
radar sweeps, calibration data, and per-sensor timestamps within a single scene
directory.
"""

from __future__ import annotations

import logging
import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Optional

import numpy as np

from kitscenes.poses import deskew_lidar, estimate_ego_motion, load_ego_poses
from kitscenes.parquet_read import load_point_cloud_parquet

from kitscenes.constants import (
    ALL_SENSOR_NAMES,
    CALIBRATION_FILENAME,
    CAMERA_IMAGE_EXTENSIONS,
    CAMERA_NAMES,
    FRAME_INDEX_WIDTH,
    LIDAR_NAMES,
    RADAR_NAMES,
    TIMESTAMP_FILE_PATTERN,
    REFERENCE_TIMESTAMP_FILENAME,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CameraCalibration:
    """Pinhole camera calibration loaded from ``calib.json``.

    Attributes:
        sensor_name: Camera sensor name (e.g. ``"camera_ring_front"``).
        intrinsic: (3, 3) pinhole camera matrix built from focal_length,
            principal_point_u, principal_point_v.
        extrinsic: (4, 4) transformation matrix *T_camera_to_reference*.
        image_size: (width, height) in pixels, or ``None`` if not yet
            resolved.  ``calib.json`` does not store image dimensions, so this
            is populated lazily by
            :meth:`SensorDataLoader.get_camera_image_size`.
    """

    sensor_name: str
    intrinsic: np.ndarray       # (3, 3) float64
    extrinsic: np.ndarray       # (4, 4) float64
    image_size: Optional[tuple[int, int]]  # (width, height) — None until resolved


@dataclass(frozen=True)
class LidarSweep:
    """A single LiDAR sweep loaded from a ``.parquet`` file.

    Attributes:
        timestamp_ns: Acquisition timestamp in nanoseconds.
        sensor_name: Sensor that produced this sweep.
        points: Ego-motion deskewed point cloud in the ego reference frame.
        Access the raw parquet
            payload through :meth:`raw`.
        deskewed: Whether deskewing was applied successfully.
        deskew_reason: Description when deskewing falls back to raw points.
    """

    timestamp_ns: int
    sensor_name: str
    _points: np.ndarray = field(repr=False)
    _raw_points: np.ndarray = field(repr=False)
    deskewed: bool = False
    deskew_reason: Optional[str] = None

    @property
    def points(self) -> np.ndarray:
        """Return the default ego-motion deskewed point cloud."""
        return self._points

    def raw(self) -> np.ndarray:
        """Return the unmodified point cloud loaded from parquet."""
        return self._raw_points


@dataclass(frozen=True)
class RadarSweep:
    """A single radar sweep loaded from a ``.parquet`` file.

    Attributes:
        timestamp_ns: Acquisition timestamp in nanoseconds.
        sensor_name: Sensor that produced this sweep.
        points: Ego-motion compensated radar points. Access the raw parquet
            payload through :meth:`raw`.
        ego_motion_compensated: Whether compensation was applied successfully.
        compensation_reason: Optional description when compensation falls back
            to raw values.
    """

    timestamp_ns: int
    sensor_name: str
    _points: np.ndarray = field(repr=False)
    _raw_points: np.ndarray = field(repr=False)
    ego_motion_compensated: bool = False
    compensation_reason: Optional[str] = None

    @property
    def points(self) -> np.ndarray:
        """Return the default ego-motion compensated radar points."""
        return self._points

    def raw(self) -> np.ndarray:
        """Return the unmodified radar point cloud loaded from parquet."""
        return self._raw_points


# ---------------------------------------------------------------------------
# Calibration key convention
# ---------------------------------------------------------------------------

# In calib.json, camera entries are stored with a ``_pinhole`` suffix
# (e.g. ``"camera_ring_front_pinhole"``), while lidar/radar entries use the
# bare sensor name.  This constant makes the convention easy to change.
_CAMERA_CALIB_SUFFIX: str = "_pinhole"


# ---------------------------------------------------------------------------
# SensorDataLoader
# ---------------------------------------------------------------------------


class SensorDataLoader:
    """Loads sensor data (cameras, LiDARs, radars) for a given scene.

    Args:
        scene_path: Path to the scene directory containing sensor
            subdirectories, timestamp files, and ``calibration/calib.json``.
    """

    def __init__(
        self,
        scene_path: Path | str,
    ) -> None:
        self._scene_path = Path(scene_path)
        if not self._scene_path.is_dir():
            raise FileNotFoundError(
                f"Scene directory not found: {self._scene_path}"
            )
        self._sensor_ts_cache: dict[str, np.ndarray] = {}
        self._frame_indices_cache: dict[str, list[int]] = {}
        self._frame_ts_lookup_cache: dict[str, dict[int, int]] = {}
        self._radar_fallback_warnings: set[str] = set()
        self._lidar_deskew_warnings: set[str] = set()
        self._radar_extrinsic_warnings: set[str] = set()

    @property
    def scene_path(self) -> Path:
        """Scene directory path."""
        return self._scene_path

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    @cached_property
    def _calibration_data(self) -> dict:
        """Raw calibration dict loaded from ``calib.json``."""
        calib_path = self._scene_path / CALIBRATION_FILENAME
        if not calib_path.exists():
            return {}
        with open(calib_path) as f:
            return json.load(f)

    def get_camera_calibration(self, camera_name: str) -> CameraCalibration:
        """Return :class:`CameraCalibration` for the given camera.

        Args:
            camera_name: One of the camera names from
                :data:`kitscenes.constants.CAMERA_NAMES`.

        Raises:
            KeyError: If the camera is not found in ``calib.json``.
        """
        # Try bare name first, then with _pinhole suffix (7/9 cameras use
        # the suffix, 2 base cameras use bare names).
        key = camera_name
        if key not in self._calibration_data:
            key = camera_name + _CAMERA_CALIB_SUFFIX
        if key not in self._calibration_data:
            raise KeyError(
                f"Camera {camera_name!r} not found in calibration data. "
                f"Tried keys: {camera_name!r}, "
                f"{camera_name + _CAMERA_CALIB_SUFFIX!r}. "
                f"Available keys: {sorted(self._calibration_data.keys())}"
            )

        entry = self._calibration_data[key]
        extrinsic = np.array(entry["T_to_reference"], dtype=np.float64)

        intr = entry["intrinsics"]
        f = float(intr["focal_length"])
        cu = float(intr["principal_point_u"])
        cv = float(intr["principal_point_v"])
        intrinsic = np.array(
            [[f, 0.0, cu],
             [0.0, f, cv],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

        # Read image_size from resolution field if present
        image_size = None
        if "resolution" in entry:
            image_size = (
                int(entry["resolution"]["width"]),
                int(entry["resolution"]["height"]),
            )

        return CameraCalibration(
            sensor_name=camera_name,
            intrinsic=intrinsic,
            extrinsic=extrinsic,
            image_size=image_size,
        )

    def get_lidar_extrinsic(self, lidar_name: str) -> np.ndarray:
        """Return the (4, 4) T_lidar_to_reference transform.

        Raises:
            KeyError: If the lidar is not in ``calib.json``.
        """
        if lidar_name not in self._calibration_data:
            raise KeyError(
                f"Lidar {lidar_name!r} not found in calibration data."
            )
        return np.array(
            self._calibration_data[lidar_name]["T_to_reference"],
            dtype=np.float64,
        )

    def get_radar_extrinsic(self, radar_name: str) -> np.ndarray:
        """Return the (4, 4) T_radar_to_reference transform.

        Raises:
            KeyError: If the radar is not found in ``calib.json``.
        """
        if radar_name not in self._calibration_data:
            raise KeyError(
                f"Radar {radar_name!r} not found in calibration data."
            )
        entry = self._calibration_data[radar_name]
        if "T_to_reference" not in entry:
            raise KeyError(
                f"Radar {radar_name!r} has no T_to_reference in calib.json."
            )
        return np.array(entry["T_to_reference"], dtype=np.float64)

    # ------------------------------------------------------------------
    # Timestamps
    # ------------------------------------------------------------------

    @cached_property
    def _reference_timestamps(self) -> np.ndarray:
        """Reference timestamps cached on first access."""
        return self._load_timestamp_file(
            self._scene_path / REFERENCE_TIMESTAMP_FILENAME
        )

    def get_reference_timestamps(self) -> np.ndarray:
        """Return (T,) array of reference timestamps in nanoseconds."""
        return self._reference_timestamps

    def get_sensor_timestamps(self, sensor_name: str) -> np.ndarray:
        """Return (T,) array of per-sensor timestamps in nanoseconds.

        Args:
            sensor_name: Any sensor name from
                :data:`kitscenes.constants.ALL_SENSOR_NAMES`.

        Raises:
            FileNotFoundError: If the sensor's timestamp file is missing.
        """
        if sensor_name not in self._sensor_ts_cache:
            ts_filename = TIMESTAMP_FILE_PATTERN.format(sensor_name=sensor_name)
            ts_path = self._scene_path / ts_filename
            if not ts_path.exists():
                raise FileNotFoundError(
                    f"Timestamp file not found: {ts_path}"
                )
            self._sensor_ts_cache[sensor_name] = self._load_timestamp_file(ts_path)
        return self._sensor_ts_cache[sensor_name]

    # ------------------------------------------------------------------
    # Sensor discovery
    # ------------------------------------------------------------------

    def get_camera_names(self) -> list[str]:
        """Return camera names whose directories exist in the scene."""
        return [
            n for n in CAMERA_NAMES
            if (self._scene_path / n).is_dir()
        ]

    def get_lidar_names(self) -> list[str]:
        """Return lidar names whose directories exist in the scene."""
        return [
            n for n in LIDAR_NAMES
            if (self._scene_path / n).is_dir()
        ]

    def get_radar_names(self) -> list[str]:
        """Return radar names whose directories exist in the scene."""
        return [
            n for n in RADAR_NAMES
            if (self._scene_path / n).is_dir()
        ]

    def get_all_sensor_names(self) -> list[str]:
        """Return all sensor names whose directories exist in the scene."""
        return [
            n for n in ALL_SENSOR_NAMES
            if (self._scene_path / n).is_dir()
        ]

    # ------------------------------------------------------------------
    # Camera images
    # ------------------------------------------------------------------

    def get_camera_image_path(
        self, camera_name: str, frame_idx: int,
    ) -> Path:
        """Resolve the file path for a camera image by frame index.

        The file name is ``{frame_idx:010d}.{ext}`` where *ext* is tried in
        order: jpg, jpeg, png.

        Raises:
            FileNotFoundError: If no matching image file exists.
        """
        camera_dir = self._scene_path / camera_name
        stem = f"{frame_idx:0{FRAME_INDEX_WIDTH}d}"
        for ext in CAMERA_IMAGE_EXTENSIONS:
            path = camera_dir / f"{stem}{ext}"
            if path.exists():
                return path
        raise FileNotFoundError(
            f"No image found for camera={camera_name!r}, "
            f"frame_idx={frame_idx} in {camera_dir}"
        )

    def get_camera_image(
        self, camera_name: str, frame_idx: int,
    ) -> np.ndarray:
        """Load a camera image as an (H, W, 3) uint8 RGB numpy array.

        Raises:
            FileNotFoundError: If no matching image file exists.
            ImportError: If neither PIL nor cv2 is available.
        """
        path = self.get_camera_image_path(camera_name, frame_idx)
        return _load_image_as_rgb(path)

    def get_camera_image_size(
        self, camera_name: str, frame_idx: int,
    ) -> tuple[int, int]:
        """Return (width, height) of a camera image without full decode.

        Useful for populating :attr:`CameraCalibration.image_size`.
        """
        path = self.get_camera_image_path(camera_name, frame_idx)
        return _read_image_size(path)

    # ------------------------------------------------------------------
    # LiDAR
    # ------------------------------------------------------------------

    def get_lidar_sweep(
        self, lidar_name: str, frame_idx: int,
    ) -> LidarSweep:
        """Load a single LiDAR sweep from its ``.parquet`` file.

        Args:
            lidar_name: One of :data:`kitscenes.constants.LIDAR_NAMES`.
            frame_idx: Zero-padded frame index (e.g. 5221).

        Raises:
            FileNotFoundError: If the parquet file does not exist.
            RuntimeError: If the parquet file cannot be decoded.
        """
        parquet_path = self._sensor_file_path(lidar_name, frame_idx, ".parquet")
        raw_pcd = load_point_cloud_parquet(parquet_path)
        timestamp_ns = self._get_frame_timestamp(lidar_name, frame_idx)

        if "timestamp" in raw_pcd.dtype.names:
            point_timestamps_s = raw_pcd["timestamp"].astype(np.float64)
        else:
            point_timestamps_s = np.zeros(len(raw_pcd), dtype=np.float64)

        try:
            T_sensor_to_ref = self.get_lidar_extrinsic(lidar_name)
        except KeyError:
            T_sensor_to_ref = None

        deskewed_pcd, success, reason = deskew_lidar(
            raw_pcd, point_timestamps_s, self._ego_poses, timestamp_ns,
            T_sensor_to_reference=T_sensor_to_ref,
        )
        if not success:
            self._warn_lidar_deskew_once(reason, lidar_name)

        return LidarSweep(
            timestamp_ns=timestamp_ns,
            sensor_name=lidar_name,
            _points=deskewed_pcd,
            _raw_points=raw_pcd,
            deskewed=success,
            deskew_reason=reason,
        )

    # ------------------------------------------------------------------
    # Radar
    # ------------------------------------------------------------------

    def get_radar_sweep(
        self, radar_name: str, frame_idx: int,
    ) -> RadarSweep:
        """Load a single radar sweep from its ``.parquet`` file.

        Args:
            radar_name: One of :data:`kitscenes.constants.RADAR_NAMES`.
            frame_idx: Zero-padded frame index.

        Raises:
            FileNotFoundError: If the parquet file does not exist.
            RuntimeError: If the parquet file cannot be decoded.
        """
        parquet_path = self._sensor_file_path(radar_name, frame_idx, ".parquet")
        raw_points = load_point_cloud_parquet(parquet_path)
        timestamp_ns = self._get_frame_timestamp(radar_name, frame_idx)
        if (
            timestamp_ns <= 0
            and "timestamp" in raw_points.dtype.names
            and len(raw_points) > 0
        ):
            scene_start_ns = int(self._reference_timestamps[0])
            relative_ns = int(np.median(raw_points["timestamp"]) * 1e9)
            timestamp_ns = scene_start_ns + relative_ns
        points, compensated, reason = self._compensate_radar_points(
            raw_points,
            timestamp_ns=timestamp_ns,
            radar_name=radar_name,
        )
        return RadarSweep(
            timestamp_ns=timestamp_ns,
            sensor_name=radar_name,
            _points=points,
            _raw_points=raw_points,
            ego_motion_compensated=compensated,
            compensation_reason=reason,
        )

    # ------------------------------------------------------------------
    # Frame listing
    # ------------------------------------------------------------------

    def get_frame_indices(self, sensor_name: str) -> list[int]:
        """Return sorted list of available frame indices for a sensor.

        Scans the sensor directory for files and extracts the integer stem.
        For cameras this means ``*.jpg``/``*.jpeg``/``*.png``; for
        lidar/radar ``*.parquet``.
        """
        sensor_dir = self._scene_path / sensor_name
        if not sensor_dir.is_dir():
            return []

        if sensor_name in CAMERA_NAMES:
            extensions = set(CAMERA_IMAGE_EXTENSIONS)
        else:
            extensions = {".parquet"}

        if sensor_name in self._frame_indices_cache:
            return self._frame_indices_cache[sensor_name]

        indices: list[int] = []
        for entry in sensor_dir.iterdir():
            if entry.suffix in extensions and entry.stem.isdigit():
                indices.append(int(entry.stem))
        indices.sort()
        self._frame_indices_cache[sensor_name] = indices
        return indices

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sensor_file_path(
        self, sensor_name: str, frame_idx: int, suffix: str,
    ) -> Path:
        """Build ``<scene>/<sensor>/<frame_idx:010d><suffix>``."""
        stem = f"{frame_idx:0{FRAME_INDEX_WIDTH}d}"
        path = self._scene_path / sensor_name / f"{stem}{suffix}"
        if not path.exists():
            raise FileNotFoundError(f"Sensor file not found: {path}")
        return path

    def _get_frame_timestamp(self, sensor_name: str, frame_idx: int) -> int:
        """Look up the timestamp for a given frame index."""
        lookup = self._frame_timestamp_lookup(sensor_name)
        return lookup.get(frame_idx, 0)

    def _frame_timestamp_lookup(self, sensor_name: str) -> dict[int, int]:
        if sensor_name not in self._frame_ts_lookup_cache:
            try:
                timestamps = self.get_sensor_timestamps(sensor_name)
            except FileNotFoundError:
                timestamps = self._reference_timestamps
            frame_indices = self.get_frame_indices(sensor_name)
            lookup = {
                frame_idx: int(timestamps[line_pos])
                for line_pos, frame_idx in enumerate(frame_indices)
                if line_pos < len(timestamps)
            }
            self._frame_ts_lookup_cache[sensor_name] = lookup
        return self._frame_ts_lookup_cache[sensor_name]

    @cached_property
    def _ego_poses(self) -> tuple:
        """Scene-root TUM poses used as the radar compensation reference."""
        return tuple(load_ego_poses(self._scene_path))

    def _compensate_radar_points(
        self,
        raw_points: np.ndarray,
        timestamp_ns: int,
        radar_name: str,
    ) -> tuple[np.ndarray, bool, Optional[str]]:
        """Return ego-motion compensated radar points or a raw fallback.

        Radar detections are stored in the sensor-local frame.  We rotate them
        into the scene reference frame with the sensor extrinsic before projecting
        the ego rigid-body velocity onto each detection's radial direction.
        """
        if timestamp_ns <= 0:
            reason = "missing radar timestamp"
            self._warn_radar_fallback_once(reason, radar_name)
            return raw_points.copy(), False, reason

        motion = estimate_ego_motion(self._ego_poses, timestamp_ns)
        if motion is None:
            reason = "missing or insufficient poses.txt data"
            self._warn_radar_fallback_once(reason, radar_name)
            return raw_points.copy(), False, reason

        compensated = raw_points.copy()
        if len(compensated) == 0:
            return compensated, True, None

        # xyz in sensor-local frame (as stored in the parquet file).
        xyz_sensor = np.column_stack([
            compensated["x"].astype(np.float64),
            compensated["y"].astype(np.float64),
            compensated["z"].astype(np.float64),
        ])

        # Sensor-to-reference extrinsic: rotate sensor-frame xyz into reference
        # frame so that the radial unit vectors and the rigid-body velocity are
        # expressed in the same coordinate system.
        try:
            T = self.get_radar_extrinsic(radar_name)
            R = T[:3, :3]
            t = T[:3, 3]
        except KeyError as exc:
            if radar_name not in self._radar_extrinsic_warnings:
                logger.warning(
                    "Radar extrinsic missing for %s (%s); "
                    "ego-motion compensation uses sensor-local frame.",
                    radar_name,
                    exc,
                )
                self._radar_extrinsic_warnings.add(radar_name)
            R = np.eye(3)
            t = np.zeros(3)
        xyz_ref = (R @ xyz_sensor.T).T + t

        # Range (distance from sensor) is invariant under rigid transforms.
        point_ranges = np.linalg.norm(xyz_sensor, axis=1)
        valid = point_ranges > 1e-6
        if not np.any(valid):
            return compensated, True, None

        # Radial unit in reference frame = rotation of the sensor-frame direction.
        radial_unit = np.zeros_like(xyz_ref)
        radial_unit[valid] = (R @ (xyz_sensor[valid] / point_ranges[valid, None]).T).T

        rigid_velocity = (
            motion.linear_velocity_reference[None, :]
            + np.cross(motion.angular_velocity_reference[None, :], xyz_ref)
        )
        ego_radial = np.einsum("ij,ij->i", radial_unit, rigid_velocity)

        compensated_range_rate = compensated["range_rate"].astype(np.float64)
        compensated_range_rate[valid] = compensated_range_rate[valid] + ego_radial[valid]
        compensated["range_rate"] = compensated_range_rate.astype(compensated.dtype["range_rate"])
        return compensated, True, None

    def _warn_radar_fallback_once(self, reason: str, radar_name: str) -> None:
        """Log radar compensation fallbacks once per loader instance and reason."""
        key = f"{radar_name}:{reason}"
        if key in self._radar_fallback_warnings:
            return
        self._radar_fallback_warnings.add(key)
        logger.warning(
            "Falling back to raw radar sweep for %s in %s: %s.",
            radar_name,
            self._scene_path,
            reason,
        )

    def _warn_lidar_deskew_once(self, reason: str, lidar_name: str) -> None:
        """Log LiDAR deskewing fallbacks once per loader instance and reason."""
        key = f"{lidar_name}:{reason}"
        if key in self._lidar_deskew_warnings:
            return
        self._lidar_deskew_warnings.add(key)
        logger.warning(
            "Falling back to raw LiDAR sweep for %s in %s: %s.",
            lidar_name,
            self._scene_path,
            reason,
        )

    @staticmethod
    def _load_timestamp_file(path: Path) -> np.ndarray:
        """Read a timestamp file (one nanosecond integer per line)."""
        if not path.exists():
            return np.array([], dtype=np.int64)
        with open(path) as f:
            lines = f.read().strip().splitlines()
        if not lines:
            return np.array([], dtype=np.int64)
        # Values > 1e12 are already in nanoseconds; smaller values are relative seconds.
        def _to_ns(s: str) -> int:
            v = float(s.strip())
            return int(v) if v > 1e12 else int(v * 1e9)

        return np.array([_to_ns(line) for line in lines], dtype=np.int64)


# ---------------------------------------------------------------------------
# Module-private helpers (image I/O, parquet loading)
# ---------------------------------------------------------------------------


def _load_image_as_rgb(path: Path) -> np.ndarray:
    """Load an image file as (H, W, 3) uint8 RGB array.

    Tries PIL first, falls back to cv2.
    """
    try:
        from PIL import Image

        img = Image.open(path).convert("RGB")
        return np.asarray(img, dtype=np.uint8)
    except ImportError:
        pass

    try:
        import cv2

        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"cv2 failed to read image: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except ImportError:
        raise ImportError(
            "Either 'Pillow' or 'opencv-python' is required to load images. "
            "Install one via: pip install Pillow"
        )


def _read_image_size(path: Path) -> tuple[int, int]:
    """Return (width, height) of an image without fully decoding it."""
    try:
        from PIL import Image

        with Image.open(path) as img:
            return img.size  # (width, height)
    except ImportError:
        pass

    try:
        import cv2

        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"cv2 failed to read image: {path}")
        h, w = img.shape[:2]
        return (w, h)
    except ImportError:
        raise ImportError(
            "Either 'Pillow' or 'opencv-python' is required to read image "
            "dimensions. Install one via: pip install Pillow"
        )
