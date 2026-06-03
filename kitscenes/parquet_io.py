# parquet_io.py

from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# pypcd is used for the PointCloud data structure
from pypcd4 import PointCloud

# --- NumPy dtypes matching C++ PointLGDataset and PointLGDatasetRadar ---
point_lg_dataset_dtype = np.dtype([
    ('x', 'f4'),  # int32 (discretized in storage)
    ('y', 'f4'),  # int32 (discretized in storage)
    ('z', 'f4'),  # int32 (discretized in storage)
    ('reflectivity', 'f4'),  # float32 (NOT discretized)
    ('timestamp', 'f8'),  # float64
    ('ring', 'u1'),  # uint8
    ('sensor_id', 'u1'),  # uint8
    ('sensor_specific_data', 'u2'),  # uint16
])

# Keep old name for backwards compatibility
point_kitscenes_dtype = point_lg_dataset_dtype

point_lg_dataset_uncompressed_dtype = np.dtype([
    ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
    ('reflectivity', 'f4'),
    ('timestamp', 'f8'),
    ('elongation', 'u1'), ('confidence', 'u1'),
    ('is2ndReturn', 'u1'), ('roi', 'u1'),
    ('facet', 'u1'), ('channel', 'u1'),
    ('sensorModel', 'u1'), ('sensorID', 'u1'),
])

# Keep old name for backwards compatibility
point_kitscenes_uncompressed_dtype = point_lg_dataset_uncompressed_dtype

point_lg_dataset_radar_dtype = np.dtype([
    ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
    ('azimuth', 'f4'), ('azimuth_std', 'f4'),
    ('elevation', 'f4'), ('elevation_std', 'f4'),
    ('range', 'f4'), ('range_std', 'f4'),
    ('range_rate', 'f4'), ('range_rate_std', 'f4'),
    ('rcs', 'f4'), ('timestamp', 'f8'),
    ('sensor_id', 'u1'),  # Added to match C++
    ('detection_id', 'u2'), ('object_id', 'u2'),
    ('classification', 'u1'),
    ('existence_probability', 'f4'),
    ('resolved_velocity_probability', 'f4'), ('multi_target_probability', 'f4'),
])

# Keep old name for backwards compatibility
point_kitscenes_radar_dtype = point_lg_dataset_radar_dtype

point_type_map = {
    "PointLGDataset": point_lg_dataset_dtype,
    "PointKITScenes": point_kitscenes_dtype,
    "PointLGDatasetUncompressed": point_lg_dataset_uncompressed_dtype,
    "PointKITScenesUncompressed": point_kitscenes_uncompressed_dtype,
    "PointLGDatasetRadar": point_lg_dataset_radar_dtype,
    "PointKITScenesRadar": point_kitscenes_radar_dtype,
}

def convert_pcd_cloud_to_numpy_array_with_different_dtype(
    cloud: PointCloud,
    target_dtype: np.dtype
) -> np.ndarray:
    """
    Converts a pypcd4.PointCloud to a NumPy array with a custom dtype.

    Only fields present in both the source and target dtypes are copied.
    """
    cloud_np = cloud.pc_data

    numpy_array_out = np.empty(cloud.points, dtype=target_dtype)

    common_fields = [name for name in cloud_np.dtype.names if name in target_dtype.names]

    for field in common_fields:
        numpy_array_out[field] = cloud_np[field]

    return numpy_array_out

def save_pcd_to_discretized_parquet(
    cloud: PointCloud,
    parquet_path: str,
    general_discretization_resolution: float = 0.005,
):
    """
    Saves a pypcd4.PointCloud to a discretized Parquet file with feature parity to the C++ version.

    For PointLGDataset:
    - Discretizes x, y, z with spatialDiscretizationResolution, stores as int32
    - Keeps reflectivity as float32 (NOT discretized)
    - Stores timestamp as float64
    - Handles NaN values by replacing them with (0, 0, 0) and reflectivity -1.0

    For PointLGDatasetRadar:
    - NO discretization - all fields stored as-is
    - All float fields kept as float32
    - Handles NaN values by replacing them with 0
    """
    if cloud.points == 0:
        print("Warning: Input cloud is empty.")

    numpy_cloud = convert_pcd_cloud_to_numpy_array_with_different_dtype(
        cloud, target_dtype=cloud.pc_data.dtype
    )
    
    point_dtype = numpy_cloud.dtype
    point_type_name = next((name for name, dtype in point_type_map.items() if dtype == point_dtype), None)
    
    if not point_type_name:
         raise ValueError(f"Unsupported point cloud dtype: {point_dtype.name}")

    # --- Invalid Point Handling (before discretization) ---
    if point_dtype == point_lg_dataset_dtype:
        # Check x, y, z, and reflectivity for NaN/inf
        invalid_mask = (
            ~np.isfinite(numpy_cloud['x']) |
            ~np.isfinite(numpy_cloud['y']) |
            ~np.isfinite(numpy_cloud['z']) |
            ~np.isfinite(numpy_cloud['reflectivity'])
        )
        if np.any(invalid_mask):
            numpy_cloud['x'][invalid_mask] = 0.0
            numpy_cloud['y'][invalid_mask] = 0.0
            numpy_cloud['z'][invalid_mask] = 0.0
            numpy_cloud['reflectivity'][invalid_mask] = -1.0
    
    elif point_dtype == point_lg_dataset_radar_dtype:
        # For radar, check all float fields for NaN/inf
        float_fields = [name for name, dt in point_dtype.fields.items() if dt[0].kind == 'f']
        invalid_mask = np.zeros(cloud.points, dtype=bool)
        for field in float_fields:
            invalid_mask |= ~np.isfinite(numpy_cloud[field])
        
        if np.any(invalid_mask):
            for field in numpy_cloud.dtype.names:
                numpy_cloud[field][invalid_mask] = 0

    # --- Discretization Logic ---
    columns = {}
    if point_dtype == point_lg_dataset_dtype:
        # LIDAR: Only discretize x, y, z; keep reflectivity as float32
        columns['x'] = np.round(numpy_cloud['x'] / general_discretization_resolution).astype(np.int32)
        columns['y'] = np.round(numpy_cloud['y'] / general_discretization_resolution).astype(np.int32)
        columns['z'] = np.round(numpy_cloud['z'] / general_discretization_resolution).astype(np.int32)
        columns['reflectivity'] = numpy_cloud['reflectivity']  # Keep as float32 (NOT discretized)
        columns['timestamp'] = numpy_cloud['timestamp']  # Keep as float64
        columns['ring'] = numpy_cloud['ring']
        columns['sensor_id'] = numpy_cloud['sensor_id']
        columns['sensor_specific_data'] = numpy_cloud['sensor_specific_data']

    elif point_dtype == point_lg_dataset_radar_dtype:
        # RADAR: NO discretization - store all fields as-is
        for field in numpy_cloud.dtype.names:
            columns[field] = numpy_cloud[field]

    # --- Metadata and Writing ---
    table = pa.Table.from_pydict(columns)
    metadata = {
        'discretization_resolution': str(general_discretization_resolution),
        'point_type_name': point_type_name,
        'pcl_width': str(cloud.metadata.width),
        'pcl_height': str(cloud.metadata.height)
    }

    table = table.cast(table.schema.with_metadata(metadata))

    # --- Configure column encoding and compression ---
    column_encoding = {}
    if point_dtype == point_lg_dataset_dtype:
        # LIDAR: x, y, z use DELTA_BINARY_PACKED; timestamp uses BYTE_STREAM_SPLIT
        column_encoding['x'] = 'DELTA_BINARY_PACKED'
        column_encoding['y'] = 'DELTA_BINARY_PACKED'
        column_encoding['z'] = 'DELTA_BINARY_PACKED'
        column_encoding['timestamp'] = 'BYTE_STREAM_SPLIT'
    elif point_dtype == point_lg_dataset_radar_dtype:
        # RADAR: All float fields use BYTE_STREAM_SPLIT
        float_fields = [field.name for field in table.schema if pa.types.is_floating(field.type)]
        for field in float_fields:
            column_encoding[field] = 'BYTE_STREAM_SPLIT'
    
    pq.write_table(
        table, parquet_path, use_dictionary=False, column_encoding=column_encoding,
        compression='zstd'
    )

def save_ground_classification_to_parquet(
    ground_classification: np.ndarray,
    parquet_path: str
):
    """
    Saves ground classification (boolean array) to a Parquet file.

    The Parquet file will contain a single boolean column 'isground' and metadata indicating it's a ground classification.
    """
    table = pa.Table.from_pydict({'isground': ground_classification.astype(bool)})
    metadata = {'classification_type': 'ground'}
    table = table.cast(table.schema.with_metadata(metadata))
    pq.write_table(table, parquet_path, compression='zstd')


def load_discretized_parquet_to_pcd(
    parquet_path: str,
    remove_invalid_points: bool = True,
) -> Optional[PointCloud]:
    """
    Loads a discretized Parquet file into a pypcd4.PointCloud object.

    Delegates decode and invalid-point filtering to :mod:`kitscenes.parquet_read`.
    """
    from kitscenes.parquet_read import load_lidar_parquet, load_radar_parquet

    try:
        table = pq.read_table(parquet_path)
    except Exception:
        return None

    metadata = table.schema.metadata or {}
    point_type_name = metadata.get(b"point_type_name", b"").decode()
    if not point_type_name:
        print("Error: Required metadata not found.")
        return None

    target_dtype = point_type_map.get(point_type_name)
    if target_dtype is None:
        print(f"Error: Unknown point type '{point_type_name}'.")
        return None

    try:
        if target_dtype == point_lg_dataset_dtype:
            numpy_cloud = load_lidar_parquet(
                parquet_path,
                remove_invalid_points=remove_invalid_points,
            )
        elif target_dtype == point_lg_dataset_radar_dtype:
            numpy_cloud = load_radar_parquet(parquet_path)
        else:
            print(f"Error: Unsupported point type '{point_type_name}'.")
            return None
    except RuntimeError:
        return None

    fields = numpy_cloud.dtype.names
    types = [numpy_cloud.dtype[name] for name in fields]
    points_list = [numpy_cloud[name] for name in fields]
    pcd_cloud = PointCloud.from_points(points_list, fields, types)

    width_str = metadata.get(b"pcl_width")
    if width_str:
        if (
            remove_invalid_points
            and target_dtype == point_lg_dataset_dtype
            and len(numpy_cloud) != int(width_str)
        ):
            pcd_cloud.metadata.width = len(numpy_cloud)
            pcd_cloud.metadata.height = 1
        else:
            pcd_cloud.metadata.width = int(width_str)
            pcd_cloud.metadata.height = int(metadata.get(b"pcl_height", b"1"))

    return pcd_cloud

def load_ground_classification_from_parquet(
    parquet_path: str
) -> Optional[np.ndarray]:
    """
    Loads ground classification from a discretized Parquet file.

    Returns a boolean NumPy array indicating ground points.
    """
    try:
        table = pq.read_table(parquet_path)
    except Exception as e:
        print(f"Error reading Parquet file: {e}")
        return None

    if 'isground' not in table.column_names:
        print("Error: 'isground' column not found in Parquet file.")
        return None

    isground_column = table.column('isground').to_numpy()
    ground_classification = isground_column.astype(bool)

    return ground_classification


# --- convert_lg_to_uncompressed (Unchanged from previous correct version) ---
def convert_lg_to_uncompressed(cloud_in: PointCloud) -> PointCloud:
    # ... This function remains correct ...
    cloud_in_np = cloud_in.pc_data
    if cloud_in_np.dtype != point_kitscenes_dtype:
        raise TypeError("Input cloud must have PointKITScenes dtype")
    cloud_out_np = np.empty(cloud_in.points, dtype=point_kitscenes_uncompressed_dtype)
    common_fields = [name for name in cloud_in_np.dtype.names if name in cloud_out_np.dtype.names]
    for field in common_fields:
        cloud_out_np[field] = cloud_in_np[field]
    ssd_col = cloud_in_np['sensorSpecificData']
    cloud_out_np['elongation']  = ssd_col & 0b1111
    cloud_out_np['confidence']  = (ssd_col >> 4) & 0b11
    cloud_out_np['is2ndReturn'] = (ssd_col >> 6) & 0b1
    cloud_out_np['roi']         = (ssd_col >> 7) & 0b11
    cloud_out_np['facet']       = (ssd_col >> 9) & 0b111
    cloud_out_np['channel']     = (ssd_col >> 12) & 0b11
    fields = cloud_out_np.dtype.names
    types = [cloud_out_np.dtype[name] for name in fields]
    points_list = [cloud_out_np[name] for name in fields]
    cloud_out = PointCloud.from_points(points_list, fields, types)
    return cloud_out