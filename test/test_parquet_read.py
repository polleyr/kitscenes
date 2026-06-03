"""Tests for kitscenes.parquet_read."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kitscenes.parquet_read import (
    filter_invalid_lidar_points,
    load_lidar_parquet,
)


def _write_lidar_parquet(path: Path, *, include_invalid: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    num_points = 4
    x = np.array([100, 0, 200, 50], dtype=np.int32)
    y = np.array([0, 0, -100, 10], dtype=np.int32)
    z = np.array([0, 0, 20, 5], dtype=np.int32)
    reflectivity = np.array([0.5, -1.0, 0.2, 0.8], dtype=np.float32)
    if not include_invalid:
        reflectivity = np.array([0.5, 0.1, 0.2, 0.8], dtype=np.float32)

    table = pa.Table.from_pydict(
        {
            "x": x,
            "y": y,
            "z": z,
            "reflectivity": reflectivity,
            "timestamp": np.zeros(num_points, dtype=np.float64),
            "ring": np.zeros(num_points, dtype=np.uint8),
            "sensor_id": np.zeros(num_points, dtype=np.uint8),
            "sensor_specific_data": np.zeros(num_points, dtype=np.uint16),
        }
    )
    metadata = {
        "discretization_resolution": "0.005",
        "point_type_name": "PointLGDataset",
        "pcl_width": str(num_points),
        "pcl_height": "1",
    }
    table = table.cast(table.schema.with_metadata(metadata))
    pq.write_table(table, path, compression="zstd")


def test_filter_invalid_lidar_points() -> None:
    points = np.array(
        [(0.0, 0.0, 0.0, -1.0), (1.0, 0.0, 0.0, 0.5)],
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("reflectivity", "f4")],
    )
    filtered = filter_invalid_lidar_points(points)
    assert len(filtered) == 1
    assert filtered[0]["reflectivity"] == pytest.approx(0.5)


def test_load_lidar_parquet_filters_invalid(tmp_path: Path) -> None:
    parquet_path = tmp_path / "0000000000.parquet"
    _write_lidar_parquet(parquet_path, include_invalid=True)
    points = load_lidar_parquet(parquet_path)
    assert len(points) == 3
    assert not np.any(np.isclose(points["reflectivity"], -1.0))
