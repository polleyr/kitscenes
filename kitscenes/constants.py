"""Dataset-wide constants for the kitscenes API."""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# Sensor names
# ---------------------------------------------------------------------------

CAMERA_BASE_NAMES: Final[tuple[str, ...]] = (
    "camera_base_front_center",
    "camera_base_front_left_rect",
    "camera_base_front_right_rect",
)

CAMERA_RING_NAMES: Final[tuple[str, ...]] = (
    "camera_ring_front",
    "camera_ring_front_left",
    "camera_ring_front_right",
    "camera_ring_rear",
    "camera_ring_rear_left",
    "camera_ring_rear_right",
)

CAMERA_NAMES: Final[tuple[str, ...]] = CAMERA_BASE_NAMES + CAMERA_RING_NAMES

LIDAR_NAMES: Final[tuple[str, ...]] = (
    "lidar_top",
    "lidar_front",
    "lidar_left",
    "lidar_right",
    "lidar_rear",
    "lidar_corner_left",
    "lidar_corner_right",
)

RADAR_NAMES: Final[tuple[str, ...]] = (
    "radar_front",
    "radar_left",
    "radar_right",
)

ALL_SENSOR_NAMES: Final[tuple[str, ...]] = CAMERA_NAMES + LIDAR_NAMES + RADAR_NAMES

# ---------------------------------------------------------------------------
# Sensor frame rates (Hz)
# ---------------------------------------------------------------------------

LIDAR_FRAMERATE_HZ: Final[int] = 10
CAMERA_RING_FRAMERATE_HZ: Final[int] = 10
CAMERA_BASE_FRAMERATE_HZ: Final[int] = 10
RADAR_FRAMERATE_HZ: Final[int] = 10

# ---------------------------------------------------------------------------
# File naming conventions
# ---------------------------------------------------------------------------

REFERENCE_TIMESTAMP_FILENAME: Final[str] = "timestamp.reference.txt"
CALIBRATION_FILENAME: Final[str] = "calibration/calib.json"
TIMESTAMP_FILE_PATTERN: Final[str] = "timestamp.{sensor_name}.txt" # relevant for async directory

# Frame index formatting: files are named ``{index:010d}.{ext}``.
FRAME_INDEX_WIDTH: Final[int] = 10

# Camera image file extensions, tried in order.
CAMERA_IMAGE_EXTENSIONS: Final[tuple[str, ...]] = (".jpg", ".jpeg", ".png") # actually always .jpg

# ---------------------------------------------------------------------------
# Ego-pose source
# ---------------------------------------------------------------------------

# TUM-format pose file: one line per pose — "timestamp tx ty tz qx qy qz qw"
# Timestamps are in seconds (float); will be converted to nanoseconds on load.
TUM_POSES_FILENAME: Final[str] = "poses.txt"

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------

KITSCENES_ROOT_ENV_VAR: Final[str] = "KITSCENES_ROOT"
KITSCENES_VIZ_OUTPUT_ENV_VAR: Final[str] = "KITSCENES_VIZ_OUTPUT"
