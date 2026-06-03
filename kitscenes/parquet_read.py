"""Shared Parquet read path for consumer-facing sensor loading.

Pipeline write utilities live in :mod:`kitscenes.parquet_io`.  This module
implements the canonical decode + invalid-point filtering used by
:class:`~kitscenes.sensors.SensorDataLoader`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

_LIDAR_POINT_TYPES = frozenset({"PointLGDataset", "PointKITScenes"})
_RADAR_POINT_TYPES = frozenset({"PointLGDatasetRadar", "PointKITScenesRadar"})


def filter_invalid_lidar_points(points: np.ndarray) -> np.ndarray:
    """Drop C++ sentinel returns (reflectivity -1 with zero xyz)."""
    if len(points) == 0 or "reflectivity" not in points.dtype.names:
        return points

    mask_refl = np.isclose(points["reflectivity"], -1.0)
    mask_zeros = (
        np.isclose(points["x"], 0.0)
        & np.isclose(points["y"], 0.0)
        & np.isclose(points["z"], 0.0)
    )
    valid_mask = ~mask_refl | ~mask_zeros
    return points[valid_mask]


def load_lidar_parquet(
    path: Path | str,
    *,
    remove_invalid_points: bool = True,
) -> np.ndarray:
    """Load a discretized LiDAR parquet file as a structured numpy array."""
    table = pq.read_table(str(path))
    metadata = table.schema.metadata or {}

    point_type_name = metadata.get(b"point_type_name", b"").decode()
    if point_type_name not in _LIDAR_POINT_TYPES:
        raise RuntimeError(
            f"Expected LiDAR point type, got {point_type_name!r} in {path}"
        )

    resolution = float(metadata.get(b"discretization_resolution", b"0.005"))
    columns = {col: table.column(col).to_numpy() for col in table.column_names}

    for axis in ("x", "y", "z"):
        if axis in columns:
            columns[axis] = (columns[axis] * resolution).astype(np.float32)

    dtype = np.dtype([(col, columns[col].dtype) for col in table.column_names])
    result = np.empty(table.num_rows, dtype=dtype)
    for col in table.column_names:
        result[col] = columns[col]

    if remove_invalid_points:
        result = filter_invalid_lidar_points(result)
    return result


def load_radar_parquet(path: Path | str) -> np.ndarray:
    """Load a radar parquet file as a structured numpy array."""
    table = pq.read_table(str(path))
    metadata = table.schema.metadata or {}
    point_type_name = metadata.get(b"point_type_name", b"").decode()
    if point_type_name and point_type_name not in _RADAR_POINT_TYPES:
        raise RuntimeError(
            f"Expected radar point type, got {point_type_name!r} in {path}"
        )

    columns = {col: table.column(col).to_numpy() for col in table.column_names}
    dtype = np.dtype([(col, columns[col].dtype) for col in table.column_names])
    result = np.empty(table.num_rows, dtype=dtype)
    for col in table.column_names:
        result[col] = columns[col]
    return result


def load_point_cloud_parquet(path: Path | str) -> np.ndarray:
    """Load a LiDAR or radar parquet file, auto-detecting the point type."""
    table = pq.read_table(str(path))
    metadata = table.schema.metadata or {}
    point_type_name = metadata.get(b"point_type_name", b"").decode()

    if point_type_name in _RADAR_POINT_TYPES:
        return load_radar_parquet(path)
    if point_type_name in _LIDAR_POINT_TYPES or not point_type_name:
        return load_lidar_parquet(path)
    raise RuntimeError(f"Unknown point type {point_type_name!r} in {path}")
