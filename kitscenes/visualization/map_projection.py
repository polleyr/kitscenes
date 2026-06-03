import numpy as np
import warnings

# Compatibility shims for older SciPy / dependencies running on newer NumPy.
# Some legacy modules still access removed aliases (e.g. `np.int`, `np.typeDict`).
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    _NUMPY_LEGACY_ALIASES = {
        'int': int,
        'float': float,
        'complex': complex,
        'bool': bool,
        'object': object,
        'str': str,
    }
    for _alias_name, _alias_type in _NUMPY_LEGACY_ALIASES.items():
        if not hasattr(np, _alias_name):
            setattr(np, _alias_name, _alias_type)

    if not hasattr(np, 'typeDict'):
        np.typeDict = np.sctypeDict

from lanelet2.ml_converter import LineStringType, TEType
from kitscenes.visualization.ml_converter_vis_utils import (
    get_map_data,
    plot_map_data,
    ls_type_to_color,
    te_type_to_icon_path,
    TYPE_GROUPINGS,
)
from PIL import Image
import io
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for multiprocessing
import matplotlib.pyplot as plt
from lanelet2.projection import UtmProjector
import lanelet2
import os
import json
import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import cv2
from scipy.spatial.transform import Rotation
import warnings
import multiprocessing as mp
from collections import deque
import argparse
import cairosvg
import piexif
import traceback
import queue
import time

# Limit internal threading in C libraries to avoid core contention with multiprocessing
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')

warnings.filterwarnings('ignore')

# Module-level worker state for multiprocessing (avoids pickling the entire MapProjector per task)
_worker_projector: Optional['MapProjector'] = None
_worker_slot: int = -1           # This worker's index into the shared arrays
_shared_slots_frame = None       # mp.Array('l') — frame_idx per slot (-1 = idle)
_shared_slots_pid = None         # mp.Array('l') — PID per slot
_shared_slots_heartbeat = None   # mp.Array('d') — last progress timestamp per slot
_shared_slots_phase = None       # mp.Array('i') — current phase code per slot

PHASE_IDLE = 0
PHASE_STARTUP = 1
PHASE_MAP_LOAD = 2
PHASE_MAP_QUERY = 3
PHASE_TOP_DOWN = 4
PHASE_ELEMENT_EXTRACT = 5
PHASE_CAMERA_FIND_IMAGE = 6
PHASE_CAMERA_READ_IMAGE = 7
PHASE_CAMERA_PREPARE = 8
PHASE_CAMERA_DRAW = 9
PHASE_IMAGE_WRITE = 10
PHASE_SHUTDOWN = 11

PHASE_NAMES = {
    PHASE_IDLE: 'idle',
    PHASE_STARTUP: 'startup',
    PHASE_MAP_LOAD: 'map_load',
    PHASE_MAP_QUERY: 'map_query',
    PHASE_TOP_DOWN: 'top_down',
    PHASE_ELEMENT_EXTRACT: 'element_extract',
    PHASE_CAMERA_FIND_IMAGE: 'camera_find_image',
    PHASE_CAMERA_READ_IMAGE: 'camera_read_image',
    PHASE_CAMERA_PREPARE: 'camera_prepare',
    PHASE_CAMERA_DRAW: 'camera_draw',
    PHASE_IMAGE_WRITE: 'image_write',
    PHASE_SHUTDOWN: 'shutdown',
}

PHASE_TIMEOUTS_SEC = {
    PHASE_STARTUP: 180.0,
    PHASE_MAP_LOAD: 180.0,
    PHASE_MAP_QUERY: 180.0,
    PHASE_TOP_DOWN: 300.0,
    PHASE_ELEMENT_EXTRACT: 120.0,
    PHASE_CAMERA_FIND_IMAGE: 30.0,
    PHASE_CAMERA_READ_IMAGE: 60.0,
    PHASE_CAMERA_PREPARE: 30.0,
    PHASE_CAMERA_DRAW: 180.0,
    PHASE_IMAGE_WRITE: 60.0,
    PHASE_SHUTDOWN: 30.0,
}

PHASE_HARD_TIMEOUTS_SEC = {
    phase: max(timeout_sec * 4.0, timeout_sec + 120.0)
    for phase, timeout_sec in PHASE_TIMEOUTS_SEC.items()
}


def _worker_bind_shared_state(slot: int, slots_frame, slots_pid, slots_heartbeat, slots_phase):
    """Bind shared worker-status arrays in the current process."""
    global _worker_slot
    global _shared_slots_frame, _shared_slots_pid, _shared_slots_heartbeat, _shared_slots_phase
    _worker_slot = slot
    _shared_slots_frame = slots_frame
    _shared_slots_pid = slots_pid
    _shared_slots_heartbeat = slots_heartbeat
    _shared_slots_phase = slots_phase


def _worker_update_status(frame_idx: Optional[int] = None, phase: Optional[int] = None):
    """Update this worker's shared status for watchdog monitoring."""
    if _worker_slot < 0 or _shared_slots_heartbeat is None:
        return
    if frame_idx is not None:
        _shared_slots_frame[_worker_slot] = frame_idx
    if phase is not None:
        _shared_slots_phase[_worker_slot] = phase
    _shared_slots_pid[_worker_slot] = os.getpid()
    _shared_slots_heartbeat[_worker_slot] = time.time()


def _worker_mark_idle():
    """Mark this worker as idle in shared state."""
    if _worker_slot < 0 or _shared_slots_heartbeat is None:
        return
    _shared_slots_frame[_worker_slot] = -1
    _shared_slots_phase[_worker_slot] = PHASE_IDLE
    _shared_slots_pid[_worker_slot] = os.getpid()
    _shared_slots_heartbeat[_worker_slot] = time.time()


def _read_proc_cpu_seconds(pid: int) -> Optional[Tuple[float, str]]:
    """Read total CPU seconds and state for a process from /proc."""
    if pid <= 0:
        return None
    stat_path = f'/proc/{pid}/stat'
    try:
        with open(stat_path, 'r') as stat_file:
            stat_line = stat_file.read().strip()
    except OSError:
        return None

    try:
        closing_paren = stat_line.rfind(')')
        rest = stat_line[closing_paren + 2:].split()
        proc_state = rest[0]
        utime_ticks = int(rest[11])
        stime_ticks = int(rest[12])
        clk_tck = os.sysconf(os.sysconf_names['SC_CLK_TCK'])
        return (utime_ticks + stime_ticks) / clk_tck, proc_state
    except (IndexError, KeyError, ValueError, OSError):
        return None


def _worker_main(slot: int, projector_kwargs: Dict, task_queue, result_queue,
                 slots_frame, slots_pid, slots_heartbeat, slots_phase):
    """Spawn-safe worker loop with local projector state and heartbeat updates."""
    global _worker_projector

    _worker_bind_shared_state(slot, slots_frame, slots_pid,
                              slots_heartbeat, slots_phase)
    _worker_update_status(frame_idx=-1, phase=PHASE_STARTUP)

    try:
        _worker_projector = MapProjector(**projector_kwargs)
    except Exception:
        result_queue.put({
            'kind': 'worker_init_failed',
            'slot': slot,
            'error': traceback.format_exc(),
        })
        _worker_update_status(frame_idx=-1, phase=PHASE_SHUTDOWN)
        return

    _worker_mark_idle()

    while True:
        try:
            task = task_queue.get(timeout=0.5)
        except queue.Empty:
            _worker_mark_idle()
            continue

        if task is None:
            _worker_update_status(frame_idx=-1, phase=PHASE_SHUTDOWN)
            break

        frame_idx = int(task)
        _worker_update_status(frame_idx=frame_idx, phase=PHASE_STARTUP)

        try:
            results = _worker_projector.project_frame(frame_idx)
            result_queue.put({
                'kind': 'result',
                'slot': slot,
                'frame_idx': frame_idx,
                'results': results,
                'error': None,
            })
        except Exception:
            result_queue.put({
                'kind': 'result',
                'slot': slot,
                'frame_idx': frame_idx,
                'results': None,
                'error': traceback.format_exc(),
            })
        finally:
            _worker_mark_idle()


class MapProjector:
    """Projects lanelet2 map onto camera images using calibration."""

    def __init__(self, config_path: Optional[str], map_path: str, poses_path: str, timestamp_path: Optional[str], base_dir: str, output_dir: str,
                 type_grouping: str = "default", color_map: Optional[Dict[str, Tuple[int, int, int]]] = None,
                 front_only: bool = False, skip_top_down: bool = False,
                 top_down_only: bool = False,
                 debug_local_submap: bool = False,
                 lat_origin: float | None = None, lon_origin: float | None = None):
        """
        Initialize the projector.

        Args:
            config_path: Optional path to sensor configuration JSON
            map_path: Path to lanelet2 OSM file
            poses_path: Path to PIN-SLAM poses file (TUM format)
            timestamp_path: Optional path to frame timestamps. If unavailable, poses are associated by line number.
            base_dir: Base directory containing camera images
            output_dir: Output directory for projections
            type_grouping: ML converter type grouping ("default", "road_border_merged", "maptr_simple", "m3tr")
            color_map: Optional dict mapping lanelet types to BGR colors.
                      If None, uses ML converter color scheme.
            front_only: If True, only process the front camera image.
            skip_top_down: If True, skip generating matplotlib top-down views.
            top_down_only: If True, only generate top-down views and skip camera projections.
            debug_local_submap: If True, print lanelets and linestrings in the local submap
                               used by the ML converter's initial search region.
            lat_origin: Latitude origin for UTM projection.
            lon_origin: Longitude origin for UTM projection.
        """
        self.config_path = Path(config_path) if config_path else None
        self.map_path = Path(map_path)
        self.poses_path = Path(poses_path)
        self.timestamp_path = Path(timestamp_path) if timestamp_path else None
        if lat_origin is None or lon_origin is None:
            from kitscenes.visualization.map_viz import resolve_map_projection_origin
            lat_origin, lon_origin = resolve_map_projection_origin(
                base_dir,
                map_path=self.map_path,
                lat_origin=lat_origin,
                lon_origin=lon_origin,
            )
        self.lat_origin = lat_origin
        self.lon_origin = lon_origin
        self.max_distance_to_camera = 150
        self.base_dir = Path(base_dir)
        from kitscenes.visualization.output_paths import guard_output_path
        self.output_dir = guard_output_path(Path(output_dir), kind="directory")
        self.front_only = front_only
        self.top_down_only = top_down_only
        self.config = self._load_config(config_path) if config_path and not top_down_only else {}
        self.poses, self.pose_timestamps = self._load_poses(
            poses_path, timestamp_path)
        self.type_grouping = type_grouping
        self.skip_top_down = skip_top_down
        self.debug_local_submap = debug_local_submap

        # Don't load the map here to avoid pickling issues with multiprocessing
        # It will be loaded on-demand in each worker process
        self.ll2_map = None
        self._unknown_subtype_te_ids = None  # Lazily computed after map load

        # Set up color mapping from ML converter
        self.color_map = color_map if color_map is not None else self._create_ml_converter_color_map()
        self.type_to_color_cache = {}  # Cache for consistent random colors
        self._icon_cache = {}  # Cache for loaded icon images

        # Camera directories
        if top_down_only:
            self.camera_names = []
        elif front_only:
            self.camera_names = ['camera_ring_front']
        else:
            self.camera_names = [
                'camera_ring_rear',
                'camera_ring_rear_left',
                'camera_ring_rear_right',
                'camera_ring_front',
                'camera_ring_front_left',
                'camera_ring_front_right',
            ]

        self.output_dir.mkdir(exist_ok=True, parents=True)

        print(f"Loaded {len(self.poses)} poses")
        print(f"Output directory: {self.output_dir}")

    def get_worker_init_kwargs(self) -> Dict:
        """Return spawn-safe constructor kwargs for worker-local projector creation."""
        return {
            'config_path': str(self.config_path) if self.config_path is not None else None,
            'map_path': str(self.map_path),
            'poses_path': str(self.poses_path),
            'timestamp_path': str(self.timestamp_path) if self.timestamp_path is not None else None,
            'base_dir': str(self.base_dir),
            'output_dir': str(self.output_dir),
            'type_grouping': self.type_grouping,
            'color_map': None,
            'front_only': self.front_only,
            'skip_top_down': self.skip_top_down,
            'top_down_only': self.top_down_only,
            'debug_local_submap': self.debug_local_submap,
            'lat_origin': self.lat_origin,
            'lon_origin': self.lon_origin,
        }

    def _touch_worker(self, phase: Optional[int] = None, frame_idx: Optional[int] = None):
        """Emit a worker heartbeat if running inside a supervised worker process."""
        _worker_update_status(frame_idx=frame_idx, phase=phase)

    def _create_default_color_map(self) -> Dict[str, Tuple[int, int, int]]:
        """Create default color map for common lanelet types (BGR format)."""
        return {}

    def _create_ml_converter_color_map(self) -> Dict[LineStringType, Tuple[int, int, int]]:
        """Create color map from ML converter utilities (convert matplotlib colors to BGR)."""
        color_map = {}

        # Convert matplotlib color names to BGR tuples for cv2
        import matplotlib.colors as mcolors

        # All LineStringTypes that have colors defined
        line_types = [
            LineStringType.RoadBorder, LineStringType.Dashed, LineStringType.Solid,
            LineStringType.SolidSolid, LineStringType.SolidDashed, LineStringType.DashedSolid,
            LineStringType.Virtual, LineStringType.Centerline, LineStringType.BikeCenterline,
            LineStringType.CurbstoneHigh, LineStringType.CurbstoneLow, LineStringType.Fence,
            LineStringType.Building, LineStringType.Wall, LineStringType.GuardRail,
            LineStringType.DrivableArea, LineStringType.ZebraCrossing
        ]

        for ls_type in line_types:
            mpl_color = ls_type_to_color(ls_type)
            # Convert matplotlib color name to RGB, then to BGR for cv2
            rgb = mcolors.to_rgb(mpl_color)
            bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
            color_map[ls_type] = bgr

        return color_map

    def _get_color_for_type(self, lanelet_type) -> Tuple[int, int, int]:
        """
        Get consistent color for a lanelet type.
        Uses ML converter color map.

        Args:
            lanelet_type: LineStringType enum or string for special types

        Returns:
            BGR color tuple
        """
        # Handle special string types (traffic elements)
        if isinstance(lanelet_type, str):
            if lanelet_type == 'stop_line':
                return (0, 0, 255)  # Red in BGR
            elif lanelet_type == 'arrow':
                return (0, 255, 255)  # Yellow in BGR
            else:
                return (128, 128, 128)  # Gray for unknown

        # Check if type is in predefined color map
        if lanelet_type in self.color_map:
            return self.color_map[lanelet_type]

        # Default gray for unknown types
        return (128, 128, 128)

    def _pose_to_gps(self, pose_matrix: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Extract GPS coordinates and heading from a pose matrix.

        Args:
            pose_matrix: T_world_to_reference (4x4)

        Returns:
            (latitude, longitude, altitude, heading_degrees)
        """
        from lanelet2.core import BasicPoint3d
        from lanelet2.io import Origin
        from lanelet2.projection import UtmProjector

        T_reference_to_world = np.linalg.inv(pose_matrix)
        position = T_reference_to_world[:3, 3]

        # Convert local UTM coordinates back to GPS
        projector = UtmProjector(Origin(self.lat_origin, self.lon_origin))
        gps_point = projector.reverse(BasicPoint3d(
            float(position[0]), float(position[1]), float(position[2])))

        # Extract heading from rotation matrix
        rotation_matrix = T_reference_to_world[:3, :3]
        rot = Rotation.from_matrix(rotation_matrix) if hasattr(
            Rotation, 'from_matrix') else Rotation.from_dcm(rotation_matrix)
        euler = rot.as_euler('zyx', degrees=True)
        heading = euler[0] % 360  # Normalize to 0-360

        return gps_point.lat, gps_point.lon, gps_point.ele, heading

    def _write_gps_exif(self, image_path: str, lat: float, lon: float, alt: float, heading: float,
                        timestamp: Optional[float] = None):
        """
        Write GPS EXIF data to a JPEG image file.

        Args:
            image_path: Path to the JPEG image
            lat: Latitude in decimal degrees
            lon: Longitude in decimal degrees
            alt: Altitude in meters
            heading: Heading/bearing in degrees (0-360, true north)
            timestamp: Unix epoch timestamp in seconds (optional)
        """
        def _to_deg_min_sec(decimal_degrees: float):
            """Convert decimal degrees to (degrees, minutes, seconds) as rational tuples."""
            d = abs(decimal_degrees)
            degrees = int(d)
            minutes = int((d - degrees) * 60)
            seconds = int(((d - degrees) * 60 - minutes) * 60 * 10000)
            return ((degrees, 1), (minutes, 1), (seconds, 10000))

        try:
            gps_ifd = {
                piexif.GPSIFD.GPSLatitudeRef: b'N' if lat >= 0 else b'S',
                piexif.GPSIFD.GPSLatitude: _to_deg_min_sec(lat),
                piexif.GPSIFD.GPSLongitudeRef: b'E' if lon >= 0 else b'W',
                piexif.GPSIFD.GPSLongitude: _to_deg_min_sec(lon),
                piexif.GPSIFD.GPSAltitudeRef: 0 if alt >= 0 else 1,
                piexif.GPSIFD.GPSAltitude: (int(abs(alt) * 100), 100),
                piexif.GPSIFD.GPSImgDirectionRef: b'T',
                piexif.GPSIFD.GPSImgDirection: (int(heading * 100), 100),
            }

            zeroth_ifd = {}
            exif_ifd = {}

            # Write timestamp if available
            if timestamp is not None:
                dt = datetime.datetime.utcfromtimestamp(timestamp)
                dt_str = dt.strftime('%Y:%m:%d %H:%M:%S').encode('ascii')
                date_str = dt.strftime('%Y:%m:%d').encode('ascii')

                # GPS timestamp (UTC)
                gps_ifd[piexif.GPSIFD.GPSDateStamp] = date_str
                gps_ifd[piexif.GPSIFD.GPSTimeStamp] = (
                    (dt.hour, 1), (dt.minute, 1), (dt.second, 1))

                # Image DateTime fields
                zeroth_ifd[piexif.ImageIFD.DateTime] = dt_str
                exif_ifd[piexif.ExifIFD.DateTimeOriginal] = dt_str
                exif_ifd[piexif.ExifIFD.DateTimeDigitized] = dt_str

            exif_dict = {"GPS": gps_ifd, "0th": zeroth_ifd,
                         "Exif": exif_ifd, "1st": {}}
            exif_bytes = piexif.dump(exif_dict)
            piexif.insert(exif_bytes, str(image_path))
        except Exception as e:
            print(f"Warning: Failed to write GPS EXIF to {image_path}: {e}")

    def _load_config(self, config_path: str) -> Dict:
        """Load sensor calibration configuration."""
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded calibration from {config_path}")
        return config

    def _filter_invalid_single_point_lanelets(self, lanelet_map):
        """Rebuild the map without lanelets whose left or right bound contains only one point."""
        if lanelet_map is None:
            return lanelet_map, 0, 0

        def _point_key(point):
            try:
                return ('id', int(point.id))
            except Exception:
                pass
            try:
                return ('xyz', float(point.x), float(point.y), float(point.z))
            except Exception:
                pass
            try:
                return ('xy', float(point.x), float(point.y))
            except Exception:
                return ('obj', repr(point))

        def _get_lanelet_bound(lanelet, bound_name: str):
            bound = getattr(lanelet, bound_name)
            return bound() if callable(bound) else bound

        def _unique_point_count(line_string) -> int:
            unique_points = set()
            for point in line_string:
                unique_points.add(_point_key(point))
            return len(unique_points)

        def _add_primitives_if_missing(target_map, layer_name: str, primitives, *,
                                       skip_ids: Optional[set] = None):
            target_layer = getattr(target_map, layer_name)
            skipped_ids = skip_ids or set()
            for primitive in primitives:
                primitive_id = primitive.id
                if primitive_id in skipped_ids or primitive_id in target_layer:
                    continue
                try:
                    target_map.add(primitive)
                except Exception as exc:
                    print(
                        f"Warning: Failed to add {layer_name} primitive {primitive_id}: {exc}")

        valid_lanelets = []
        removed_ids = []
        removed_line_string_ids = set()
        for lanelet in list(lanelet_map.laneletLayer):
            try:
                left_bound = _get_lanelet_bound(lanelet, 'leftBound')
                right_bound = _get_lanelet_bound(lanelet, 'rightBound')
                left_size = _unique_point_count(left_bound)
                right_size = _unique_point_count(right_bound)
            except Exception:
                valid_lanelets.append(lanelet)
                continue

            if left_size <= 1 or right_size <= 1:
                removed_ids.append(lanelet.id)
                try:
                    removed_line_string_ids.add(left_bound.id)
                    removed_line_string_ids.add(right_bound.id)
                except Exception:
                    pass
            else:
                valid_lanelets.append(lanelet)

        print(
            f"Malformed-map filter: {len(removed_ids)} lanelet(s) with <=1 unique-point bounds, "
            f"{len(removed_line_string_ids)} associated boundary line string(s) marked for rebuild skipping")

        if not removed_ids:
            return lanelet_map, 0, 0

        rebuilt_submap = lanelet2.core.createSubmapFromLanelets(valid_lanelets)
        rebuilt_map = rebuilt_submap.laneletMap()

        _add_primitives_if_missing(
            rebuilt_map, 'areaLayer', list(lanelet_map.areaLayer))
        _add_primitives_if_missing(
            rebuilt_map, 'regulatoryElementLayer', list(lanelet_map.regulatoryElementLayer))
        _add_primitives_if_missing(
            rebuilt_map, 'polygonLayer', list(lanelet_map.polygonLayer))
        _add_primitives_if_missing(
            rebuilt_map, 'lineStringLayer', list(lanelet_map.lineStringLayer),
            skip_ids=removed_line_string_ids)
        _add_primitives_if_missing(
            rebuilt_map, 'pointLayer', list(lanelet_map.pointLayer))

        print(
            f"Removed {len(removed_ids)} lanelet(s) with <=1 unique point in a left/right bound")
        print(
            f"Skipped {len(removed_line_string_ids)} associated boundary line string(s) during map rebuild")
        return rebuilt_map, len(removed_ids), len(removed_line_string_ids)

    def _load_map(self, map_path: str):
        """Load lanelet2 map."""
        try:
            self._touch_worker(phase=PHASE_MAP_LOAD)
            map_path_str = str(map_path)
            print(f"Loading map from {map_path_str}")
            # lanelet2.io.load returns a tuple (map, errors)
            from lanelet2.io import loadRobust, Origin
            from lanelet2.projection import UtmProjector

            # Create projector with correct origin for this map
            projector = UtmProjector(Origin(self.lat_origin, self.lon_origin))

            # Use loadRobust to handle map errors gracefully
            result = loadRobust(map_path_str, projector)

            # Check if result is tuple
            if isinstance(result, tuple):
                lanelet_map = result[0]
                errors = result[1] if len(result) > 1 else []
                if errors:
                    print(
                        f"Map loaded with {len(errors)} warnings (using loadRobust)")
            else:
                lanelet_map = result

            lanelet_map, removed_count, removed_line_string_count = self._filter_invalid_single_point_lanelets(
                lanelet_map)

            print(f"Successfully loaded lanelet2 map")
            print(f"  Lanelets: {len(lanelet_map.laneletLayer)}")

            return lanelet_map
        except Exception as e:
            print(f"Error loading map: {e}")
            traceback.print_exc()
            return None

    def _get_unknown_subtype_te_ids(self, lanelet_map) -> set:
        """
        Collect IDs of traffic sign linestrings in the map that have subtype="unknown".

        Args:
            lanelet_map: Lanelet2 map object

        Returns:
            Set of linestring IDs with unknown subtype
        """
        unknown_ids = set()
        if lanelet_map is None:
            return unknown_ids
        for ls in lanelet_map.lineStringLayer:
            attrs = ls.attributes
            ls_type = str(attrs["type"]) if "type" in attrs else ""
            ls_subtype = str(attrs["subtype"]) if "subtype" in attrs else ""
            if ls_type == "traffic_sign" and ls_subtype == "unknown":
                unknown_ids.add(ls.id)
        if unknown_ids:
            print(f"Found {len(unknown_ids)} traffic sign linestrings with subtype='unknown' (will be hidden)")
        return unknown_ids

    def _load_poses(self, poses_path: str, timestamp_path: Optional[str]) -> Dict[int, np.ndarray]:
        """
        Load poses from TUM format file.
        TUM format: timestamp tx ty tz qx qy qz qw


        Returns dict mapping frame index to T_world_to_reference transformation matrix.
        """
        poses = {}
        pose_timestamps = {}

        timestamps = []
        if timestamp_path:
            timestamp_file = Path(timestamp_path)
            if timestamp_file.exists():
                with open(timestamp_file, 'r') as f:
                    timestamps = [float(line.strip()) for line in f if line.strip()
                                  and not line.startswith('#')]
            else:
                print(
                    f"Warning: Timestamp file not found: {timestamp_file}. "
                    "Falling back to pose line-number mapping")

        pose_entries = []

        def _nearest_timestamp_index(sorted_timestamps: np.ndarray, query_timestamp: float) -> int:
            insertion_idx = int(np.searchsorted(sorted_timestamps, query_timestamp, side='left'))
            if insertion_idx <= 0:
                return 0
            if insertion_idx >= len(sorted_timestamps):
                return len(sorted_timestamps) - 1

            prev_idx = insertion_idx - 1
            if abs(sorted_timestamps[insertion_idx] - query_timestamp) < abs(sorted_timestamps[prev_idx] - query_timestamp):
                return insertion_idx
            return prev_idx

        def _median_nearest_error(reference_timestamps: np.ndarray, query_timestamps: np.ndarray) -> float:
            if reference_timestamps.size == 0 or query_timestamps.size == 0:
                return float('inf')
            errors = []
            for query_timestamp in query_timestamps:
                match_idx = _nearest_timestamp_index(reference_timestamps, float(query_timestamp))
                errors.append(abs(reference_timestamps[match_idx] - query_timestamp))
            return float(np.median(errors)) if errors else float('inf')

        with open(poses_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split()
                if len(parts) < 8:
                    continue

                try:
                    # TUM format: timestamp tx ty tz qx qy qz qw
                    position = np.array([float(parts[1]), float(
                        parts[2]), float(parts[3])], dtype=np.float64)
                    # position = np.array([float(parts[1]), float(
                    #     parts[2]), 2.1], dtype=np.float64)
                    quat = np.array([float(parts[4]), float(parts[5]), float(
                        parts[6]), float(parts[7])], dtype=np.float64)

                    timestamp = float(parts[0])
                    pose_entries.append((timestamp, position, quat))
                except (ValueError, IndexError):
                    continue

        if not pose_entries:
            return poses, pose_timestamps

        if not timestamps:
            for frame_idx, (timestamp, position, quat) in enumerate(pose_entries):
                try:
                    rot_obj = Rotation.from_quat(quat)
                    rot = rot_obj.as_matrix() if hasattr(rot_obj, 'as_matrix') else rot_obj.as_dcm()

                    T_reference_to_world = np.eye(4, dtype=np.float64)
                    T_reference_to_world[:3, :3] = rot
                    T_reference_to_world[:3, 3] = position

                    poses[frame_idx] = np.linalg.inv(T_reference_to_world)
                    pose_timestamps[frame_idx] = timestamp
                except (ValueError, IndexError):
                    continue

            print(
                f'Pose/timestamp alignment: sequential pose-line mapping, '
                f'unique mapped frames={len(poses)}/{len(pose_entries)}')
            return poses, pose_timestamps

        reference_timestamps_raw = np.asarray(timestamps, dtype=np.float64)
        pose_timestamp_array = np.asarray([entry[0] for entry in pose_entries], dtype=np.float64)
        pose_timestamp_sample = pose_timestamp_array[:min(len(pose_timestamp_array), 200)]

        candidate_alignments = []
        for scale in (1.0, 1e3, 1e6, 1e9):
            scaled_reference = reference_timestamps_raw / scale
            candidate_alignments.append((f'scale={scale:g},shift=0', scaled_reference))
            shifted_reference = scaled_reference - scaled_reference[0] + pose_timestamp_array[0]
            candidate_alignments.append((f'scale={scale:g},shift_to_pose_start', shifted_reference))

        best_alignment_name = None
        best_reference_timestamps = None
        best_alignment_error = float('inf')
        for alignment_name, candidate_reference in candidate_alignments:
            candidate_error = _median_nearest_error(candidate_reference, pose_timestamp_sample)
            if candidate_error < best_alignment_error:
                best_alignment_error = candidate_error
                best_alignment_name = alignment_name
                best_reference_timestamps = candidate_reference

        if best_reference_timestamps is None:
            best_reference_timestamps = reference_timestamps_raw / 1e9
            best_alignment_name = 'fallback_scale=1e9'

        matched_frame_indices = [
            _nearest_timestamp_index(best_reference_timestamps, pose_timestamp)
            for pose_timestamp in pose_timestamp_array
        ]

        unique_frame_count = len(set(matched_frame_indices))
        if len(pose_entries) > 1 and unique_frame_count <= 1 and len(timestamps) >= len(pose_entries):
            print(
                'Warning: Pose/timestamp matching collapsed onto a single frame index '
                f'using {best_alignment_name}; falling back to sequential mapping')
            matched_frame_indices = list(range(len(pose_entries)))
        elif len(pose_entries) > 1 and unique_frame_count < max(2, len(pose_entries) // 5):
            print(
                'Warning: Pose/timestamp matching produced many duplicate frame indices '
                f'({unique_frame_count}/{len(pose_entries)}) using {best_alignment_name} '
                f'(median error {best_alignment_error:.6f}s)')

        for frame_idx, (timestamp, position, quat) in zip(matched_frame_indices, pose_entries):
            try:
                rot_obj = Rotation.from_quat(quat)
                rot = rot_obj.as_matrix() if hasattr(rot_obj, 'as_matrix') else rot_obj.as_dcm()

                T_reference_to_world = np.eye(4, dtype=np.float64)
                T_reference_to_world[:3, :3] = rot
                T_reference_to_world[:3, 3] = position

                poses[int(frame_idx)] = np.linalg.inv(T_reference_to_world)
                pose_timestamps[int(frame_idx)] = timestamp
            except (ValueError, IndexError):
                continue

        if pose_entries:
            print(
                f'Pose/timestamp alignment: {best_alignment_name}, '
                f'unique mapped frames={len(poses)}/{len(pose_entries)}, '
                f'median nearest error={best_alignment_error:.6f}s')

        return poses, pose_timestamps

    def _find_image(self, camera_name: str, frame_idx: int) -> Optional[Path]:
        """Find image file for given camera and frame index."""
        camera_dir = self.base_dir / camera_name
        if not camera_dir.exists():
            return None

        # Try different naming conventions
        for extension in ['.jpg', '.jpeg', '.png']:
            img_path = camera_dir / f'{frame_idx:010d}{extension}'
            if img_path.exists():
                return img_path

        return None

    def _project_point(self, point_camera: np.ndarray, cam_config: Dict) -> Optional[Tuple[int, int]]:
        """
        Project a 3D point onto camera image.

        Args:
            point_camera: 3D point in camera coordinate frame (numpy array of shape (3,))
            cam_config: Camera configuration dictionary

        Returns:
            (u, v) pixel coordinates or None if outside image
        """

        # Discard points that are further than self.max_distance_to_camera
        distance = np.linalg.norm(point_camera[:3])
        if distance > self.max_distance_to_camera:
            return None

        # If point is behind the camera, discard
        if point_camera[2] < 0.01:
            return None

        # Project using pinhole camera model
        intrinsics = cam_config['intrinsics']
        fx = intrinsics['focal_length']
        fy = intrinsics['focal_length']
        cx = intrinsics['principal_point_u']
        cy = intrinsics['principal_point_v']

        # Perspective projection
        u = fx * point_camera[0] / point_camera[2] + cx
        v = fy * point_camera[1] / point_camera[2] + cy

        return (int(u), int(v))

    def _interpolate_points_at_z_plane(self, points_camera: np.ndarray, z_plane: float = 0.1) -> np.ndarray:
        """
        Insert intersection points where consecutive points cross a z-plane.

        Args:
            points_camera: Array of shape (N, 3) in camera coordinates
            z_plane: Plane depth used for clipping/interpolation

        Returns:
            Array of shape (M, 3) with interpolated crossing points inserted
        """
        if points_camera.shape[0] < 2:
            return points_camera

        start_points = points_camera[:-1]
        end_points = points_camera[1:]
        crossing_mask = ((start_points[:, 2] - z_plane) *
                         (end_points[:, 2] - z_plane) < 0)

        if not np.any(crossing_mask):
            return points_camera

        crossing_indices = np.flatnonzero(crossing_mask)
        crossing_start = start_points[crossing_mask]
        crossing_end = end_points[crossing_mask]
        
        # Prevent division by zero when z-coordinates don't actually cross
        z_deltas = crossing_end[:, 2] - crossing_start[:, 2]
        z_deltas = np.where(np.abs(z_deltas) < 1e-10, 1e-10, z_deltas)
        t_values = (z_plane - crossing_start[:, 2]) / z_deltas
        t_values = np.clip(t_values, 0.0, 1.0)  # Clamp to valid range
        intersection_points = crossing_start + \
            t_values[:, None] * (crossing_end - crossing_start)

        insert_counts = crossing_mask.astype(np.int64)
        prefix_counts = np.concatenate(([0], np.cumsum(insert_counts)))

        result = np.empty(
            (points_camera.shape[0] + intersection_points.shape[0], 3), dtype=np.float64)
        original_indices = np.arange(points_camera.shape[0]) + prefix_counts
        result[original_indices] = points_camera

        inserted_indices = crossing_indices + prefix_counts[crossing_indices] + 1
        result[inserted_indices] = intersection_points
        return result

    def _transform_and_project_element(self, element_points: List[List[float]],
                                       T_reference_to_camera: np.ndarray,
                                       cam_config: Dict) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """
        Transform element points into camera space and project them in a batched path.

        Args:
            element_points: Element points in reference frame
            T_reference_to_camera: 4x4 transform from reference to camera
            cam_config: Camera configuration dictionary

        Returns:
            Tuple of (projected_points, points_camera)
        """
        if not element_points:
            return [], np.empty((0, 3), dtype=np.float64)

        points_reference = np.asarray(element_points, dtype=np.float64)
        if points_reference.ndim != 2 or points_reference.shape[1] != 3:
            points_reference = points_reference.reshape(-1, 3)

        rotation = T_reference_to_camera[:3, :3]
        translation = T_reference_to_camera[:3, 3]
        points_camera = points_reference @ rotation.T + translation
        points_camera = self._interpolate_points_at_z_plane(points_camera)

        if points_camera.shape[0] == 0:
            return [], points_camera

        distances = np.linalg.norm(points_camera, axis=1)
        z_values = points_camera[:, 2]
        finite_mask = np.isfinite(points_camera).all(axis=1)
        valid_mask = finite_mask & (distances <= self.max_distance_to_camera) & (z_values >= 0.1)

        if not np.any(valid_mask):
            return [], points_camera

        valid_points = points_camera[valid_mask]

        intrinsics = cam_config['intrinsics']
        fx = intrinsics['focal_length']
        fy = intrinsics['focal_length']
        cx = intrinsics['principal_point_u']
        cy = intrinsics['principal_point_v']

        u_values = fx * valid_points[:, 0] / valid_points[:, 2] + cx
        v_values = fy * valid_points[:, 1] / valid_points[:, 2] + cy
        projected = np.column_stack((u_values, v_values))
        projected_finite_mask = np.isfinite(projected).all(axis=1)
        projected = projected[projected_finite_mask]
        valid_points = valid_points[projected_finite_mask]

        if projected.shape[0] == 0:
            return [], points_camera

        projected = np.rint(projected).astype(np.int64, copy=False)
        projected_points = [tuple(map(int, point)) for point in projected]

        return projected_points, valid_points

    def _get_reference_to_camera_matrix(self, camera_name: str) -> np.ndarray:
        """
        Get transformation matrix from reference frame to camera.

        The ML converter returns points in the reference (vehicle/ego) frame,
        so we only need to transform from reference to camera.

        Args:
            camera_name: Name of camera

        Returns:
            Transformation matrix from reference to camera (4x4)
        """

        if camera_name not in self.config:
            print("Camera " + camera_name + " not found in calib!")
            return np.eye(4)

        # Get camera calibration: T_camera_to_reference (from camera to reference)
        T_camera_to_reference = np.array(
            self.config[camera_name]['T_to_reference'], dtype=np.float64)

        # Invert to get T_reference_to_camera
        T_reference_to_camera = np.linalg.inv(T_camera_to_reference)

        return T_reference_to_camera

    def _get_line_style_params(self, ls_type: LineStringType) -> Tuple[int, int, int, int, float]:
        """
        Get line style parameters for a given LineStringType.

        Args:
            ls_type: LineStringType enum

        Returns:
            Tuple of (dash_length, gap_length, fg_thickness, bg_thickness, alpha)
            - dash_length: 0 for solid, >0 for dashed/dotted
            - gap_length: gap between dashes
            - fg_thickness: foreground line thickness
            - bg_thickness: background line thickness (0 if no background)
            - alpha: transparency (0.0 to 1.0, where 1.0 is fully opaque)
        """
        # Match matplotlib rendering from plot_map_data
        if ls_type == LineStringType.Dashed:
            return (40, 40, 2, 7, 0.9)  # Dashed with gray background
        elif ls_type in [LineStringType.Solid, LineStringType.SolidSolid]:
            return (0, 0, 2, 7, 0.9)  # Solid with gray background
        elif ls_type in [LineStringType.SolidDashed, LineStringType.DashedSolid]:
            return (0, 0, 5, 10, 0.9)  # Solid with gray background
        elif ls_type == LineStringType.Centerline:
            return (10, 10, 2, 0, 0.7)  # Dashed, no background
        elif ls_type == LineStringType.BikeCenterline:
            return (5, 5, 2, 0, 0.7)  # Dotted (shorter dashes), no background
        elif ls_type == LineStringType.Virtual:
            return (0, 0, 2, 0, 0.5)  # Solid, no background, semi-transparent
        elif ls_type == LineStringType.ZebraCrossing:
            return (0, 0, 5, 0, 0.9)  # Thick solid
        else:
            # Road borders and other elements
            return (0, 0, 5, 0, 0.9)  # Thick solid, no background

    def _get_adaptive_dash_pattern(self, dash_length: int, gap_length: int,
                                   thickness: int,
                                   points_camera: Optional[np.ndarray] = None,
                                   representative_distance: Optional[float] = None) -> Tuple[int, int]:
        """Scale dash/gap lengths based on line distance to the ego/camera.

        Nearer lines get longer dash/gap lengths, farther lines get shorter ones,
        but both are clamped to minimum visible sizes.
        """
        if dash_length <= 0 or gap_length < 0:
            return dash_length, gap_length

        min_dash_length = max(6, int(np.ceil(thickness * 2.0)))
        min_gap_length = max(4, int(np.ceil(thickness * 1.5)))

        if representative_distance is not None and np.isfinite(representative_distance):
            reference_distance = 25.0
            scale = np.clip(reference_distance / max(float(representative_distance), 1.0), 0.45, 1.75)
            adaptive_dash = max(min_dash_length, int(round(dash_length * scale)))
            adaptive_gap = max(min_gap_length, int(round(gap_length * scale)))
            return adaptive_dash, adaptive_gap

        if points_camera is None or points_camera.size == 0:
            return max(dash_length, min_dash_length), max(gap_length, min_gap_length)

        finite_mask = np.isfinite(points_camera).all(axis=1)
        if points_camera.ndim != 2 or points_camera.shape[1] != 3:
            return max(dash_length, min_dash_length), max(gap_length, min_gap_length)

        valid_points = points_camera[finite_mask]
        if valid_points.shape[0] == 0:
            return max(dash_length, min_dash_length), max(gap_length, min_gap_length)

        distances = np.linalg.norm(valid_points, axis=1)
        distances = distances[np.isfinite(distances)]
        if distances.size == 0:
            return max(dash_length, min_dash_length), max(gap_length, min_gap_length)

        representative_distance = float(np.median(distances))
        return self._get_adaptive_dash_pattern(
            dash_length, gap_length, thickness,
            representative_distance=representative_distance)

    def _draw_dashed_line(self, img: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int],
                          color: Tuple[int, int, int], thickness: int, dash_length: int, gap_length: int, current_dist: int = 0, start_dist: int = 0) -> Tuple[int, int]:

        """
        Draw a dashed line on image.

        Args:
            img: Image to draw on
            pt1: Start point (x, y)
            pt2: End point (x, y)
            color: BGR color tuple
            thickness: Line thickness in pixels
            dash_length: Length of each dash in pixels
            gap_length: Length of gap between dashes in pixels
        """
        pt1 = (int(pt1[0]), int(pt1[1]))
        pt2 = (int(pt2[0]), int(pt2[1]))

        dist = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
        if dist < 1:
            return current_dist, start_dist

        # Unit vector along the line
        dx = (pt2[0] - pt1[0]) / dist
        dy = (pt2[1] - pt1[1]) / dist

        # Draw dashes
        current_dist = current_dist - start_dist
        pattern_length = dash_length + gap_length
        min_visible_dash_length = min(float(dash_length), max(15.0, thickness * 1.5))
        min_visible_gap_length = min(float(gap_length), max(15.0, float(thickness)))
        last_draw_end = None

        def _draw_interval(interval_start: float, interval_end: float):
            nonlocal last_draw_end
            draw_start = max(0.0, min(interval_start, dist))
            draw_end = max(0.0, min(interval_end, dist))

            if last_draw_end is not None:
                draw_start = max(draw_start, last_draw_end + min_visible_gap_length)

            if draw_end - draw_start < 1e-6:
                return False
            start_x = int(pt1[0] + dx * draw_start)
            start_y = int(pt1[1] + dy * draw_start)
            end_x = int(pt1[0] + dx * draw_end)
            end_y = int(pt1[1] + dy * draw_end)
            cv2.line(img, (start_x, start_y), (end_x, end_y),
                     color, thickness, cv2.LINE_AA)
            last_draw_end = draw_end
            return True

        # Prevent infinite loop if pattern_length is 0
        if pattern_length < 0.01:
            return current_dist + start_dist, start_dist

        pending_small_dash = None

        while current_dist < dist:
            if current_dist < pattern_length and current_dist > gap_length:
                dash_start = 0.0
                dash_end = current_dist - gap_length
            else:
                dash_start = current_dist
                dash_end = min(current_dist + dash_length, dist)

            visible_dash_length = dash_end - dash_start

            if visible_dash_length >= min_visible_dash_length:
                pending_small_dash = None
                _draw_interval(dash_start, dash_end)
            else:
                if pending_small_dash is None:
                    pending_small_dash = (dash_start, dash_end)
                else:
                    pending_small_dash = (pending_small_dash[0], dash_end)

                if pending_small_dash[1] - pending_small_dash[0] >= min_visible_dash_length:
                    _draw_interval(pending_small_dash[0], pending_small_dash[1])
                    pending_small_dash = None

            current_dist += pattern_length
        
        current_dist = current_dist + start_dist
        end_dist = dist + start_dist

        return current_dist, end_dist

    def _draw_dashed_polyline(self, img: np.ndarray, points: List[Tuple[int, int]],
                              color: Tuple[int, int, int], thickness: int,
                              dash_length: int, gap_length: int,
                              is_closed: bool = False,
                              points_camera: Optional[np.ndarray] = None):
        """Draw a dashed polyline with dash/gap lengths adapted per dash."""
        if len(points) < 2:
            return

        has_camera_points = (
            points_camera is not None and
            isinstance(points_camera, np.ndarray) and
            points_camera.ndim == 2 and
            points_camera.shape[1] == 3 and
            points_camera.shape[0] == len(points)
        )

        filtered_points = [np.array(points[0], dtype=np.float64)]
        filtered_camera_points = [np.array(points_camera[0], dtype=np.float64)] if has_camera_points else None
        for idx, point in enumerate(points[1:], start=1):
            point_array = np.array(point, dtype=np.float64)
            if np.linalg.norm(point_array - filtered_points[-1]) > 1e-6:
                filtered_points.append(point_array)
                if has_camera_points:
                    filtered_camera_points.append(np.array(points_camera[idx], dtype=np.float64))

        if len(filtered_points) < 2:
            return

        if is_closed:
            if np.linalg.norm(filtered_points[0] - filtered_points[-1]) > 1e-6:
                filtered_points.append(filtered_points[0].copy())
                if filtered_camera_points is not None:
                    filtered_camera_points.append(filtered_camera_points[0].copy())

        if len(filtered_points) < 2:
            return

        polyline = np.vstack(filtered_points)
        segment_vectors = polyline[1:] - polyline[:-1]
        segment_lengths = np.linalg.norm(segment_vectors, axis=1)
        if segment_lengths.size == 0:
            return

        cumulative_lengths = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        total_length = float(cumulative_lengths[-1])
        if total_length < 1e-6:
            return

        last_draw_end = None
        pending_small_dash = None

        vertex_camera_distances = None
        if filtered_camera_points is not None:
            camera_points_array = np.vstack(filtered_camera_points)
            vertex_camera_distances = np.linalg.norm(camera_points_array, axis=1)
            if not np.isfinite(vertex_camera_distances).all():
                vertex_camera_distances = None

        def _point_at(distance_along: float) -> np.ndarray:
            clamped_distance = min(max(distance_along, 0.0), total_length)
            seg_idx = int(np.searchsorted(cumulative_lengths, clamped_distance, side='right') - 1)
            seg_idx = max(0, min(seg_idx, len(segment_lengths) - 1))
            seg_length = segment_lengths[seg_idx]
            if seg_length < 1e-6:
                return polyline[seg_idx].copy()
            distance_in_segment = clamped_distance - cumulative_lengths[seg_idx]
            interpolation = distance_in_segment / seg_length
            return polyline[seg_idx] + interpolation * segment_vectors[seg_idx]

        def _camera_distance_at(distance_along: float) -> Optional[float]:
            if vertex_camera_distances is None:
                return None

            clamped_distance = min(max(distance_along, 0.0), total_length)
            seg_idx = int(np.searchsorted(cumulative_lengths, clamped_distance, side='right') - 1)
            seg_idx = max(0, min(seg_idx, len(segment_lengths) - 1))
            seg_length = segment_lengths[seg_idx]
            if seg_length < 1e-6:
                return float(vertex_camera_distances[seg_idx])

            distance_in_segment = clamped_distance - cumulative_lengths[seg_idx]
            interpolation = distance_in_segment / seg_length
            start_distance = float(vertex_camera_distances[seg_idx])
            end_distance = float(vertex_camera_distances[seg_idx + 1])
            return start_distance + interpolation * (end_distance - start_distance)

        def _draw_interval(interval_start: float, interval_end: float,
                           min_visible_gap_length: float):
            nonlocal last_draw_end

            draw_start = max(0.0, min(interval_start, total_length))
            draw_end = max(0.0, min(interval_end, total_length))

            if last_draw_end is not None:
                draw_start = max(draw_start, last_draw_end + min_visible_gap_length)

            if draw_end - draw_start < 1e-6:
                return False

            start_seg_idx = int(np.searchsorted(cumulative_lengths, draw_start, side='right') - 1)
            end_seg_idx = int(np.searchsorted(cumulative_lengths, draw_end, side='right') - 1)
            start_seg_idx = max(0, min(start_seg_idx, len(segment_lengths) - 1))
            end_seg_idx = max(0, min(end_seg_idx, len(segment_lengths) - 1))

            draw_points = [_point_at(draw_start)]
            for vertex_idx in range(start_seg_idx + 1, end_seg_idx + 1):
                vertex_distance = cumulative_lengths[vertex_idx]
                if draw_start < vertex_distance < draw_end:
                    draw_points.append(polyline[vertex_idx])
            draw_points.append(_point_at(draw_end))

            raster_points = []
            for draw_point in draw_points:
                point_tuple = (int(round(draw_point[0])), int(round(draw_point[1])))
                if not raster_points or point_tuple != raster_points[-1]:
                    raster_points.append(point_tuple)

            if len(raster_points) < 2:
                return False

            points_array = np.array(raster_points, dtype=np.int32)
            cv2.polylines(img, [points_array], False, color, thickness, cv2.LINE_AA)
            last_draw_end = draw_end
            return True

        dash_start = 0.0
        while dash_start < total_length:
            local_distance = _camera_distance_at(dash_start)
            local_dash_length, local_gap_length = self._get_adaptive_dash_pattern(
                dash_length, gap_length, thickness,
                representative_distance=local_distance)
            local_min_visible_dash_length = min(float(local_dash_length), max(5.0, thickness * 1.5))
            local_min_visible_gap_length = min(float(local_gap_length), max(4.0, float(thickness)))

            dash_end = min(dash_start + local_dash_length, total_length)
            visible_dash_length = dash_end - dash_start

            if visible_dash_length >= local_min_visible_dash_length:
                pending_small_dash = None
                _draw_interval(dash_start, dash_end, local_min_visible_gap_length)
            else:
                if pending_small_dash is None:
                    pending_small_dash = (dash_start, dash_end)
                else:
                    pending_small_dash = (pending_small_dash[0], dash_end)

                if pending_small_dash[1] - pending_small_dash[0] >= local_min_visible_dash_length:
                    _draw_interval(
                        pending_small_dash[0], pending_small_dash[1],
                        local_min_visible_gap_length)
                    pending_small_dash = None

            dash_start += local_dash_length + local_gap_length



    def _draw_styled_polyline(self, img: np.ndarray, points: List[Tuple[int, int]],
                              ls_type: LineStringType, color: Tuple[int, int, int],
                              is_closed: bool = False,
                              points_camera: Optional[np.ndarray] = None):
        """
        Draw polyline with appropriate style (solid/dashed/dotted, with/without background).

        Args:
            img: Image to draw on
            points: List of (x, y) pixel coordinates
            ls_type: LineStringType enum
            color: BGR color tuple for foreground
            is_closed: Whether the polyline should be drawn as a closed loop
            points_camera: Polyline points in camera coordinates for distance-adaptive dash sizing
        """
        normalized_points = [(int(point[0]), int(point[1])) for point in points]

        if len(normalized_points) < 2:
            if len(normalized_points) == 1:
                cv2.circle(img, normalized_points[0], 3, color, -1)
            return

        segment_points = normalized_points + [normalized_points[0]] if is_closed else normalized_points

        dash_length, gap_length, fg_thickness, bg_thickness, alpha = self._get_line_style_params(
            ls_type)

        # Draw background if needed (gray thick line)
        if bg_thickness > 0:
            gray = (128, 128, 128)
            if dash_length > 0:
                self._draw_dashed_polyline(
                    img, normalized_points, gray, bg_thickness,
                    dash_length, gap_length, is_closed=is_closed,
                    points_camera=points_camera)
            else:
                # Solid background
                points_array = np.array(normalized_points, dtype=np.int32)
                cv2.polylines(img, [points_array], is_closed,
                              gray, bg_thickness, cv2.LINE_AA)

        # Draw foreground (with alpha blending if needed)
        if alpha < 1.0:
            # Compute tight ROI to avoid copying the entire image
            pts_arr = np.array(normalized_points, dtype=np.int32)
            pad = max(fg_thickness, bg_thickness) + 4
            roi_x1 = max(0, int(pts_arr[:, 0].min()) - pad)
            roi_y1 = max(0, int(pts_arr[:, 1].min()) - pad)
            roi_x2 = min(img.shape[1], int(pts_arr[:, 0].max()) + pad)
            roi_y2 = min(img.shape[0], int(pts_arr[:, 1].max()) + pad)

            if roi_x2 > roi_x1 and roi_y2 > roi_y1:
                # Copy only the ROI for overlay (contiguous for cv2 compatibility)
                overlay_roi = np.ascontiguousarray(
                    img[roi_y1:roi_y2, roi_x1:roi_x2].copy())
                orig_roi = np.ascontiguousarray(
                    img[roi_y1:roi_y2, roi_x1:roi_x2].copy())

                # Offset points to ROI-local coordinates
                offset_points = [(p[0] - roi_x1, p[1] - roi_y1)
                                 for p in normalized_points]
                offset_segment_points = offset_points + [offset_points[0]] if is_closed else offset_points

                if dash_length > 0:
                    self._draw_dashed_polyline(
                        overlay_roi, offset_points, color, fg_thickness,
                        dash_length, gap_length, is_closed=is_closed,
                        points_camera=points_camera)
                else:
                    # Solid foreground
                    points_array = np.array(offset_points, dtype=np.int32)
                    cv2.polylines(overlay_roi, [points_array], is_closed,
                                  color, fg_thickness, cv2.LINE_AA)
                # Blend overlay ROI back
                blended = cv2.addWeighted(
                    overlay_roi, alpha, orig_roi, 1 - alpha, 0)
                img[roi_y1:roi_y2, roi_x1:roi_x2] = blended
        else:
            # Fully opaque - draw directly
            if dash_length > 0:
                self._draw_dashed_polyline(
                    img, normalized_points, color, fg_thickness,
                    dash_length, gap_length, is_closed=is_closed,
                    points_camera=points_camera)
            else:
                # Solid foreground
                points_array = np.array(normalized_points, dtype=np.int32)
                cv2.polylines(img, [points_array], is_closed,
                              color, fg_thickness, cv2.LINE_AA)

    def _load_icon_image(self, icon_path: str, target_width: int = None, target_height: int = None) -> Optional[np.ndarray]:
        """
        Load icon from file (PNG or SVG) and optionally resize.
        Results are cached by (icon_path, target_width, target_height).

        Args:
            icon_path: Path to icon file
            target_width: Target width in pixels (optional)
            target_height: Target height in pixels (optional)

        Returns:
            Icon as RGBA numpy array or None if loading fails
        """
        cache_key = (icon_path, target_width, target_height)
        if cache_key in self._icon_cache:
            return self._icon_cache[cache_key]

        try:
            if icon_path.endswith('.svg'):
                # Convert SVG to PNG in memory
                png_data = cairosvg.svg2png(
                    url=icon_path, output_width=256, output_height=256)
                icon_img = Image.open(io.BytesIO(png_data))
            else:
                # Load PNG directly
                icon_img = Image.open(icon_path)

            # Convert to RGBA
            if icon_img.mode != 'RGBA':
                icon_img = icon_img.convert('RGBA')

            # Resize if target dimensions provided
            if target_width is not None or target_height is not None:
                orig_w, orig_h = icon_img.size
                aspect_ratio = orig_w / orig_h

                if target_width and target_height:
                    new_size = (target_width, target_height)
                elif target_width:
                    new_size = (target_width, int(target_width / aspect_ratio))
                else:
                    new_size = (int(target_height * aspect_ratio),
                                target_height)

                # Use LANCZOS resampling (compatible with older Pillow versions)
                try:
                    icon_img = icon_img.resize(
                        new_size, Image.Resampling.LANCZOS)
                except AttributeError:
                    icon_img = icon_img.resize(new_size, Image.LANCZOS)

            # Convert to numpy array and then to BGR for OpenCV
            icon_array = np.array(icon_img)
            # PIL uses RGB, OpenCV uses BGR - convert RGB to BGR for color channels
            if icon_array.shape[2] == 4:  # RGBA
                icon_bgr = cv2.cvtColor(
                    icon_array[:, :, :3], cv2.COLOR_RGB2BGR)
                # Recombine with alpha
                icon_array = np.dstack([icon_bgr, icon_array[:, :, 3]])
            self._icon_cache[cache_key] = icon_array
            return icon_array
        except Exception as e:
            print(f"Error loading icon {icon_path}: {e}")
            self._icon_cache[cache_key] = None
            return None

    def _draw_perspective_icon(self, img: np.ndarray, icon_rgba: np.ndarray,
                               quad_points_2d: List[Tuple[int, int]], alpha: float = 0.5):
        """
        Draw icon with perspective transform onto image.

        Args:
            img: Image to draw on (BGR)
            icon_rgba: Icon image (RGBA numpy array)
            quad_points_2d: 4 corner points in image space [(x,y), ...] defining where icon should appear
            alpha: Transparency (0-1)
        """
        if icon_rgba is None or len(quad_points_2d) != 4:
            return

        try:
            icon_h, icon_w = icon_rgba.shape[:2]

            # Source points (icon corners)
            src_points = np.array([
                [0, 0],
                [icon_w - 1, 0],
                [icon_w - 1, icon_h - 1],
                [0, icon_h - 1]
            ], dtype=np.float32)

            # Destination points (in image space)
            dst_points = np.array(quad_points_2d, dtype=np.float32)

            # Compute tight ROI bounding box from destination quad (avoid full-image ops)
            roi_x1 = max(0, int(np.floor(dst_points[:, 0].min())) - 2)
            roi_y1 = max(0, int(np.floor(dst_points[:, 1].min())) - 2)
            roi_x2 = min(img.shape[1], int(
                np.ceil(dst_points[:, 0].max())) + 2)
            roi_y2 = min(img.shape[0], int(
                np.ceil(dst_points[:, 1].max())) + 2)

            if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
                return

            # Offset destination points into ROI-local coordinates
            roi_offset = np.array([roi_x1, roi_y1], dtype=np.float32)
            dst_roi = dst_points - roi_offset
            roi_w = roi_x2 - roi_x1
            roi_h = roi_y2 - roi_y1

            # Compute perspective transform into ROI-sized buffer
            transform_matrix = cv2.getPerspectiveTransform(
                src_points, dst_roi)

            # Warp icon into small ROI buffer only
            warped_roi = cv2.warpPerspective(icon_rgba, transform_matrix,
                                             (roi_w, roi_h),
                                             flags=cv2.INTER_LINEAR)

            # Vectorized alpha blend over ROI only (not the full image)
            alpha_channel = warped_roi[:, :, 3:4] / 255.0 * alpha
            img_roi = img[roi_y1:roi_y2, roi_x1:roi_x2]
            img[roi_y1:roi_y2, roi_x1:roi_x2] = (
                alpha_channel * warped_roi[:, :, :3] +
                (1 - alpha_channel) * img_roi
            ).astype(np.uint8)
        except Exception as e:
            print(f"Error drawing perspective icon: {e}")

    def _draw_simple_icon(self, img: np.ndarray, icon_rgba: np.ndarray,
                          center: Tuple[int, int], alpha: float = 0.4):
        """
        Draw icon at a specific location without perspective transform.

        Args:
            img: Image to draw on (BGR)
            icon_rgba: Icon image (RGBA numpy array)
            center: Center position (x, y) in image coordinates
            alpha: Transparency (0-1)
        """
        if icon_rgba is None:
            return

        try:
            icon_h, icon_w = icon_rgba.shape[:2]

            # Calculate desired icon position in image
            icon_left = center[0] - icon_w // 2
            icon_top = center[1] - icon_h // 2
            icon_right = icon_left + icon_w
            icon_bottom = icon_top + icon_h

            # Clip to image bounds
            x1 = max(0, icon_left)
            y1 = max(0, icon_top)
            x2 = min(img.shape[1], icon_right)
            y2 = min(img.shape[0], icon_bottom)

            # Check if there's any visible region
            if x2 <= x1 or y2 <= y1:
                return

            # Calculate corresponding icon region
            icon_x1 = x1 - icon_left
            icon_y1 = y1 - icon_top
            icon_x2 = icon_x1 + (x2 - x1)
            icon_y2 = icon_y1 + (y2 - y1)

            # Extract regions (ensure they're the same size)
            icon_region = icon_rgba[icon_y1:icon_y2, icon_x1:icon_x2]
            img_region = img[y1:y2, x1:x2]

            # Verify shapes match
            if icon_region.shape[:2] != img_region.shape[:2]:
                return

            # Extract alpha channel and apply overall alpha
            alpha_channel = icon_region[:, :, 3:4] / 255.0 * alpha

            # Blend
            img[y1:y2, x1:x2] = (alpha_channel * icon_region[:, :, :3] +
                                 (1 - alpha_channel) * img_region).astype(np.uint8)
        except Exception as e:
            print(f"Error drawing simple icon: {e}")

    def _is_vertical_traffic_element(self, te_type: TEType) -> bool:
        """
        Check if traffic element is vertical (sign) or horizontal (road marking).

        Args:
            te_type: Traffic element type

        Returns:
            True if vertical element (traffic sign, traffic light)
        """
        # Traffic signs and traffic lights are vertical
        vertical_types = [
            TEType.TLCar, TEType.TLBike, TEType.TLPedestrian, TEType.TLMisc,
            TEType.TSStop, TEType.TSYield, TEType.TSSpeedLimit, TEType.TSNoEntry,
            TEType.TSRightOfWay, TEType.TSPriorityRoad, TEType.TSRoundabout,
            TEType.TSTurnRight, TEType.TSTurnLeft, TEType.TSGoStraight,
            TEType.TSGoStraightOrRight, TEType.TSGoStraightOrLeft, TEType.TSTurnLeftOrRight,
            TEType.TSPassRight, TEType.TSPassLeft, TEType.TSPedestrianCrossing,
            TEType.TSMisc
        ]
        return te_type in vertical_types

    def _get_type_groupings(self):
        """
        Get type grouping objects on-demand (recreated in each worker process).

        Returns:
            Tuple of (ls_type_grouping, te_type_grouping)
        """
        from lanelet2.ml_converter import getDefaultTETypeGrouping
        ls_type_grouping = TYPE_GROUPINGS.get(
            self.type_grouping, TYPE_GROUPINGS["default"])
        te_type_grouping = getDefaultTETypeGrouping()
        return ls_type_grouping, te_type_grouping

    def _get_centerline_types(self) -> set:
        """Return line-string types that should use the reduced ML-converter extent."""
        return {LineStringType.Centerline, LineStringType.BikeCenterline}

    def _get_polygonal_linestring_types(self) -> set:
        """Return line-string types that represent polygon perimeters."""
        return {LineStringType.ZebraCrossing, LineStringType.PedestrianCrossing}

    def _normalize_polygonal_linestring_points(self, points_3d: List[List[float]]) -> List[List[float]]:
        """Normalize polygon-like linestring points into a non-self-intersecting outer ring.

        This is intended for crosswalk-style geometries where the ML-converter may
        occasionally emit a self-intersecting perimeter or even a resampled bow-tie
        polyline. In that case we rebuild the shape from the XY convex hull, which
        recovers the expected outer rectangle/convex boundary from the sampled points.
        """
        if len(points_3d) < 3:
            return points_3d

        points_array = np.asarray(points_3d, dtype=np.float64)
        if points_array.ndim != 2 or points_array.shape[1] != 3:
            return points_3d

        if np.linalg.norm(points_array[0] - points_array[-1]) < 1e-6:
            points_array = points_array[:-1]

        if points_array.shape[0] < 3:
            return points_array.tolist()

        unique_xy = np.unique(np.round(points_array[:, :2], decimals=9), axis=0)
        if unique_xy.shape[0] < 3:
            return points_array.tolist()

        try:
            hull_xy = cv2.convexHull(unique_xy.astype(np.float32))[:, 0, :]
        except Exception:
            hull_xy = None

        if hull_xy is not None and hull_xy.shape[0] >= 3:
            hull_points = []
            for hull_point_xy in hull_xy:
                nearest_idx = int(np.argmin(np.sum(
                    (points_array[:, :2] - hull_point_xy) ** 2, axis=1)))
                hull_points.append(points_array[nearest_idx])

            ordered_points = np.asarray(hull_points, dtype=np.float64)
            start_idx = int(np.argmin(np.sum(
                (ordered_points[:, :2] - points_array[0, :2]) ** 2, axis=1)))
            if start_idx > 0:
                ordered_points = np.roll(ordered_points, -start_idx, axis=0)
            return ordered_points.tolist()

        centroid_xy = points_array[:, :2].mean(axis=0)
        offsets_xy = points_array[:, :2] - centroid_xy
        if not np.isfinite(offsets_xy).all():
            return points_array.tolist()

        angles = np.arctan2(offsets_xy[:, 1], offsets_xy[:, 0])
        radial_dist_sq = np.sum(offsets_xy * offsets_xy, axis=1)
        sort_order = np.lexsort((radial_dist_sq, angles))
        ordered_points = points_array[sort_order]

        start_idx = int(np.argmin(np.sum(
            (ordered_points[:, :2] - points_array[0, :2]) ** 2, axis=1)))
        if start_idx > 0:
            ordered_points = np.roll(ordered_points, -start_idx, axis=0)

        return ordered_points.tolist()

    def _format_linestring_points(self, line_string) -> str:
        """Format a Lanelet2 linestring's points for debug logging."""
        point_strings = []
        for point in line_string:
            try:
                point_strings.append(
                    f"({float(point.x):.3f}, {float(point.y):.3f}, {float(point.z):.3f})")
            except Exception:
                point_strings.append(f"point_id={getattr(point, 'id', '?')}")
        return '[' + ', '.join(point_strings) + ']'

    def _get_lanelet_bound(self, lanelet, bound_name: str):
        """Return a lanelet bound for bindings that expose bounds as properties or methods."""
        bound = getattr(lanelet, bound_name)
        return bound() if callable(bound) else bound

    def _get_local_submap_debug_path(self) -> Path:
        """Return the path of the overwritten local-submap debug dump."""
        return self.output_dir / 'local_submap_debug.txt'

    def _debug_print_local_submap(self, center_x: float, center_y: float,
                                  frame_idx: Optional[int] = None):
        """Write the initial ML-converter local submap contents for debugging."""
        if not self.debug_local_submap or self.ll2_map is None:
            return

        try:
            from lanelet2.core import BasicPoint2d, BoundingBox2d
        except Exception as exc:
            debug_path = self._get_local_submap_debug_path()
            debug_path.write_text(
                f"Debug local submap: failed to import Lanelet2 geometry types: {exc}\n",
                encoding='utf-8')
            return

        extent_long = float(self.max_distance_to_camera)
        extent_lat = float(self.max_distance_to_camera / 2)
        max_extent = float(np.sqrt(extent_long * extent_long + extent_lat * extent_lat))
        margin = 1.1 * max_extent
        search_region = BoundingBox2d(
            BasicPoint2d(center_x - margin, center_y - margin),
            BasicPoint2d(center_x + margin, center_y + margin),
        )

        lanelets = list(self.ll2_map.laneletLayer.search(search_region))
        line_strings = list(self.ll2_map.lineStringLayer.search(search_region))

        frame_label = f" frame={frame_idx}" if frame_idx is not None else ""
        lines = [
            f"Local submap debug{frame_label}: center=({center_x:.3f}, {center_y:.3f}), "
            f"extent_long={extent_long:.3f}, extent_lat={extent_lat:.3f}, "
            f"search_margin={margin:.3f}, lanelets={len(lanelets)}, linestrings={len(line_strings)}"
        ]

        for lanelet in lanelets:
            try:
                left_bound = self._get_lanelet_bound(lanelet, 'leftBound')
                right_bound = self._get_lanelet_bound(lanelet, 'rightBound')
                lines.append(
                    f"  lanelet {lanelet.id}: left_bound={left_bound.id} {self._format_linestring_points(left_bound)}"
                )
                lines.append(
                    f"  lanelet {lanelet.id}: right_bound={right_bound.id} {self._format_linestring_points(right_bound)}"
                )
            except Exception as exc:
                lines.append(f"  lanelet {lanelet.id}: failed to print bounds: {exc}")

        lanelet_bound_ids = set()
        for lanelet in lanelets:
            try:
                lanelet_bound_ids.add(self._get_lanelet_bound(lanelet, 'leftBound').id)
                lanelet_bound_ids.add(self._get_lanelet_bound(lanelet, 'rightBound').id)
            except Exception:
                pass

        for line_string in line_strings:
            try:
                ownership = 'lanelet_bound' if line_string.id in lanelet_bound_ids else 'standalone'
                lines.append(
                    f"  linestring {line_string.id} ({ownership}): {self._format_linestring_points(line_string)}"
                )
            except Exception as exc:
                lines.append(
                    f"  linestring {getattr(line_string, 'id', '?')}: failed to print: {exc}"
                )

        debug_path = self._get_local_submap_debug_path()
        debug_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    def get_map_data_for_pose(self, pose_matrix: np.ndarray, frame_idx: Optional[int] = None,
                              extent_long: Optional[float] = None,
                              extent_lat: Optional[float] = None):
        """
        Extract MapData using ML converter for a given pose.

        Args:
            pose_matrix: 4x4 transformation matrix (T_world_to_reference)

        Returns:
            MapData object from ML converter
        """
        # Load map on-demand if not already loaded (for multiprocessing)
        if self.ll2_map is None:
            self.ll2_map = self._load_map(self.map_path)

        self._touch_worker(phase=PHASE_MAP_QUERY)

        # Extract position and orientation from pose matrix
        # The pose matrix is T_world_to_reference, we need T_reference_to_world
        T_reference_to_world = np.linalg.inv(pose_matrix)

        position = T_reference_to_world[:3, 3]
        rotation_matrix = T_reference_to_world[:3, :3]

        self._debug_print_local_submap(position[0], position[1], frame_idx=frame_idx)

        # Convert rotation matrix to Euler angles (ZYX convention)
        from scipy.spatial.transform import Rotation
        # Use from_matrix() if available (scipy >= 1.4), otherwise from_dcm()
        if hasattr(Rotation, 'from_matrix'):
            rot = Rotation.from_matrix(rotation_matrix)
        else:
            rot = Rotation.from_dcm(rotation_matrix)
        euler_zyx = rot.as_euler('zyx', degrees=False)
        yaw, pitch, roll = euler_zyx[0], euler_zyx[1], euler_zyx[2]

        if extent_long is None:
            extent_long = float(self.max_distance_to_camera)
        else:
            extent_long = float(extent_long)

        if extent_lat is None:
            extent_lat = float(self.max_distance_to_camera / 2)
        else:
            extent_lat = float(extent_lat)

        # Get map data using ML converter
        mData = get_map_data(
            self.ll2_map,
            position[0], position[1], position[2],
            yaw, pitch, roll,
            type_grouping=self.type_grouping,
            n_points_lanes=60,
            extent_long=extent_long,
            extent_lat=extent_lat
        )

        return mData

    def extract_polylines_from_map_data(self, mData, include_line_types: Optional[set] = None,
                                        exclude_line_types: Optional[set] = None) -> list:
        """
        Extract polylines from MapData for projection.
        Returns list of tuples: (polyline_points_3d, element_type, element_category, metadata)
        where:
        - polyline_points_3d: [[x,y,z], ...]
        - element_type: LineStringType or TEType enum
        - element_category: 'linestring' or 'traffic_element'
        - metadata: dict with additional info (e.g., icon_path for TEs)
        """
        if isinstance(mData, str):  # Error case
            return []

        self._touch_worker(phase=PHASE_ELEMENT_EXTRACT)

        elements = []

        # Get tensor data
        tfData = mData.getTensorInstanceData(
            pointsIn2d=False, ignoreBuffer=True)

        # Get type groupings on-demand (avoids pickling issues)
        ls_type_grouping, te_type_grouping = self._get_type_groupings()
        include_line_types = set(include_line_types) if include_line_types is not None else None
        exclude_line_types = set(exclude_line_types) if exclude_line_types is not None else set()

        # Extract line types from grouping (use representative types)
        line_types = [group.representative for group in ls_type_grouping]

        for ls_type in line_types:
            if include_line_types is not None and ls_type not in include_line_types:
                continue
            if ls_type in exclude_line_types:
                continue
            lines = tfData.compoundLineStringsOfType(ls_type)
            for line in lines:
                # line is a numpy array of shape (N, 2) or (N, 3)
                points_3d = []
                for point in line:
                    if point.shape[0] == 2:
                        points_3d.append(
                            [float(point[0]), float(point[1]), 0.0])
                    else:
                        points_3d.append(
                            [float(point[0]), float(point[1]), float(point[2])])

                if len(points_3d) > 0:
                    elements.append((points_3d, ls_type, 'linestring', {}))

        # Extract traffic element types from grouping (use representative types)
        te_types = [group.representative for group in te_type_grouping]

        # Lazily compute unknown-subtype TE IDs (once per worker process)
        if self._unknown_subtype_te_ids is None:
            self._unknown_subtype_te_ids = self._get_unknown_subtype_te_ids(self.ll2_map)

        for te_type in te_types:
            # Use mData-level API to get instances with their lanelet2 IDs
            te_instances_with_ids = mData.teInstancesOfType(te_type)
            for te_id, te_instance in te_instances_with_ids.items():
                # Skip traffic signs with subtype="unknown" in the map
                if te_id in self._unknown_subtype_te_ids:
                    continue

                # Get point data from the instance
                point_matrices = te_instance.pointMatrices(False)  # pointsIn2d=False
                points_3d = []
                for matrix in point_matrices:
                    for row_idx in range(matrix.shape[0]):
                        point = matrix[row_idx]
                        if matrix.shape[1] == 2:
                            points_3d.append(
                                [float(point[0]), float(point[1]), 0.0])
                        else:
                            points_3d.append(
                                [float(point[0]), float(point[1]), float(point[2])])

                if len(points_3d) > 0:
                    # Get icon path for this traffic element
                    icon_path = te_type_to_icon_path(te_type)
                    metadata = {'icon_path': icon_path, 'te_type': te_type}
                    elements.append(
                        (points_3d, te_type, 'traffic_element', metadata))

        return elements

    def generate_top_down_view(self, mData, frame_idx: int) -> Optional[Path]:
        """
        Generate and save top-down matplotlib view for a frame.

        Args:
            mData: MapData object from ML converter
            frame_idx: Frame index

        Returns:
            Path to saved top-down view image
        """
        if isinstance(mData, str):  # Error case
            return None

        try:
            self._touch_worker(phase=PHASE_TOP_DOWN, frame_idx=frame_idx)
            # Create figure for top-down view
            plt.figure(figsize=(10, 20))
            from kitscenes.visualization.map_viz import default_car_img_path
            _car_img = default_car_img_path()
            # Lazily compute unknown-subtype TE IDs
            if self._unknown_subtype_te_ids is None:
                self._unknown_subtype_te_ids = self._get_unknown_subtype_te_ids(self.ll2_map)
            plot_map_data(mData, car_img_path=_car_img, show_traffic_elements=True,
                          show_bike_lanes=True, show_stats=False, show_drivable_area=True,
                          orientation='vertical', show_legend=True,
                          exclude_te_ids=self._unknown_subtype_te_ids,
                          max_arrow_length_fraction=0.3)

            # Save to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            plt.close()
            buf.seek(0)

            # Convert to cv2 image
            pil_img = Image.open(buf)
            cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            # Save top-down view
            output_file = self.output_dir / \
                'top_down' / f'{frame_idx:010d}.jpg'
            output_file.parent.mkdir(exist_ok=True, parents=True)
            self._touch_worker(phase=PHASE_IMAGE_WRITE, frame_idx=frame_idx)
            cv2.imwrite(str(output_file), cv2_img)

            # Geotag with pose GPS coordinates
            if frame_idx in self.poses:
                lat, lon, alt, heading = self._pose_to_gps(
                    self.poses[frame_idx])
                ts = self.pose_timestamps.get(frame_idx)
                self._write_gps_exif(str(output_file), lat,
                                     lon, alt, heading, timestamp=ts)

            return output_file
        except Exception as e:
            print(f"Error generating top-down view for frame {frame_idx}: {e}")
            return None

    def project_frame(self, frame_idx: int, mData_precomputed=None) -> Dict[str, Path]:
        """
        Project map onto all camera images for a given frame and generate top-down view.

        Args:
            frame_idx: Frame index (pose index)
            mData_precomputed: Pre-computed MapData object (optional, for multiprocessing)

        Returns:
            Dictionary mapping camera name to output image path (includes 'top_down')
        """
        if frame_idx not in self.poses:
            return {}

        # Get T_world_to_reference from poses
        T_world_to_reference = self.poses[frame_idx]
        results = {}
        self._touch_worker(phase=PHASE_MAP_QUERY, frame_idx=frame_idx)

        # Compute GPS coordinates for geotagging
        lat, lon, alt, heading = self._pose_to_gps(T_world_to_reference)

        # Get MapData using ML converter
        if mData_precomputed is None:
            mData = self.get_map_data_for_pose(T_world_to_reference, frame_idx=frame_idx)
        else:
            mData = mData_precomputed

        if isinstance(mData, str):  # Error case
            mData = None

        # Generate top-down view (optional — matplotlib is very slow)
        if not self.skip_top_down and mData is not None:
            top_down_path = self.generate_top_down_view(mData, frame_idx)
            if top_down_path:
                results['top_down'] = top_down_path

        if self.top_down_only:
            return results

        # Extract elements from MapData for projection.
        # Run a second ML-converter query with reduced extents for centerlines only.
        centerline_types = self._get_centerline_types()
        elements = []
        if mData is not None:
            elements.extend(
                self.extract_polylines_from_map_data(
                    mData,
                    exclude_line_types=centerline_types,
                )
            )

            centerline_map_data = self.get_map_data_for_pose(
                T_world_to_reference,
                frame_idx=frame_idx,
                extent_long=self.max_distance_to_camera / 2.0,
                extent_lat=self.max_distance_to_camera / 4.0,
            )
            if not isinstance(centerline_map_data, str) and centerline_map_data is not None:
                elements.extend(
                    self.extract_polylines_from_map_data(
                        centerline_map_data,
                        include_line_types=centerline_types,
                    )
                )

        polygonal_linestring_types = self._get_polygonal_linestring_types()

        for camera_name in self.camera_names:
            self._touch_worker(phase=PHASE_CAMERA_FIND_IMAGE, frame_idx=frame_idx)
            # Find image
            img_path = self._find_image(camera_name, frame_idx)
            if not img_path:
                continue

            # Load image
            self._touch_worker(phase=PHASE_CAMERA_READ_IMAGE, frame_idx=frame_idx)
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # Get transformation from reference to this camera
            # (MapData points are already in reference frame)
            self._touch_worker(phase=PHASE_CAMERA_PREPARE, frame_idx=frame_idx)
            T_reference_to_camera = self._get_reference_to_camera_matrix(
                camera_name)

            # Get camera config
            if camera_name  not in self.config:
                continue

            cam_config = self.config[camera_name]

            # Project and draw map elements
            self._touch_worker(phase=PHASE_CAMERA_DRAW, frame_idx=frame_idx)
            elements_drawn = 0

            for element_points, element_type, element_category, metadata in elements:
                # print(element_type)
                # print(elements_drawn)

                is_polygonal_linestring = (
                    element_category == 'linestring' and
                    element_type in polygonal_linestring_types
                )
                element_points_for_projection = (
                    self._normalize_polygonal_linestring_points(element_points)
                    if is_polygonal_linestring else element_points
                )

                projected_points, element_points_camera = self._transform_and_project_element(
                    element_points_for_projection, T_reference_to_camera, cam_config)

                # Draw based on element category
                if element_category == 'linestring':
                    # Draw styled polyline
                    color = self._get_color_for_type(element_type)
                    self._draw_styled_polyline(
                        img, projected_points, element_type, color,
                        is_closed=is_polygonal_linestring,
                        points_camera=element_points_camera)
                    if len(projected_points) > 0:
                        elements_drawn += 1

                elif element_category == 'traffic_element':
                    te_type = metadata.get('te_type')
                    icon_path = metadata.get('icon_path')

                    # import pdb
                    # pdb.set_trace()

                    # Handle stop lines - draw thick red line
                    if te_type == TEType.StopLine:
                        if len(projected_points) >= 2:
                            red_color = (0, 0, 255)
                            points_array = np.array(
                                projected_points, dtype=np.int32)
                            cv2.polylines(
                                img, [points_array], False, red_color, 5, cv2.LINE_AA)
                            elements_drawn += 1

                    # Handle road markings (arrows, symbols) - draw yellow line + perspective icon
                    elif not self._is_vertical_traffic_element(te_type):

                        # Draw perspective-transformed icon on top
                        if len(projected_points) >= 2:
                            # Calculate icon dimensions from linestring
                            # Use start and end points for orientation and size
                            start_2d = np.array(
                                projected_points[0], dtype=np.float32)
                            end_2d = np.array(
                                projected_points[-1], dtype=np.float32)

                            # Calculate line length in pixels (for long side of icon)
                            line_length = np.linalg.norm(end_2d - start_2d)

                            if line_length > 5 and icon_path:  # Only draw if visible and icon available
                                # Load icon
                                icon_rgba = self._load_icon_image(icon_path)
                                if icon_rgba is not None:
                                    icon_h, icon_w = icon_rgba.shape[:2]
                                    aspect_ratio = icon_w / icon_h

                                    # Rotate icon clockwise by 90 degrees to match line orientation
                                    icon_rgba = np.rot90(icon_rgba, k=1)
                                    # Update dimensions after rotation
                                    icon_h, icon_w = icon_rgba.shape[:2]
                                    aspect_ratio = icon_w / icon_h
                                    # Now treat as landscape
                                    width_pixels = max(
                                        16, min(64, 0.5*line_length))
                                    height_pixels = width_pixels * aspect_ratio

                                    # Calculate perpendicular direction for width
                                    line_dir = (
                                        end_2d - start_2d) / line_length
                                    perp_dir = np.array(
                                        [-line_dir[1], line_dir[0]])

                                    # Calculate 4 corners of the icon quad
                                    center = (start_2d + end_2d) / 2
                                    half_length_vec = perp_dir * \
                                        (width_pixels / 2)
                                    half_width_vec = line_dir * \
                                        (height_pixels / 2)

                                    quad_points = [
                                        tuple((center - half_length_vec +
                                              half_width_vec).astype(int)),
                                        tuple((center + half_length_vec +
                                              half_width_vec).astype(int)),
                                        tuple((center + half_length_vec -
                                               half_width_vec).astype(int)),
                                        tuple((center - half_length_vec -
                                               half_width_vec).astype(int)),
                                    ]

                                    # Resize icon to reasonable resolution
                                    icon_resized = self._load_icon_image(
                                        icon_path,
                                        target_width=min(
                                            256, int(width_pixels * 2)),
                                        target_height=min(
                                            256, int(height_pixels * 2))
                                    )

                                    self._draw_perspective_icon(
                                        img, icon_resized, quad_points, alpha=0.4)
                                    elements_drawn += 1

                        # Draw yellow line for the linestring
                        if len(projected_points) >= 2:
                            yellow_color = (0, 255, 255)  # Yellow in BGR
                            points_array = np.array(
                                projected_points, dtype=np.int32)
                            cv2.polylines(
                                img, [points_array], False, yellow_color, 3, cv2.LINE_AA)

                    # Handle vertical elements (signs) - simple placement
                    elif self._is_vertical_traffic_element(te_type):

                        # Draw icon slightly above the projected polyline box
                        if len(projected_points) >= 2 and icon_path:
                            pts_arr = np.array(projected_points, dtype=np.int32)

                            # Calculate distance-based icon size
                            centroid_3d = np.mean(element_points_camera, axis=0)
                            distance = np.linalg.norm(centroid_3d)
                            reference_distance = 30.0
                            base_size = 64
                            scaled_height = int(
                                base_size * min(2.0, reference_distance / max(1.0, distance)))
                            scaled_height = max(16, min(64, scaled_height))

                            # Place icon center to the right of the box
                            right_x = int(pts_arr[:, 0].max())
                            center_y = int(pts_arr[:, 1].mean())
                            icon_center = (right_x + scaled_height // 2 + 4, center_y)

                            icon_rgba = self._load_icon_image(
                                icon_path, target_height=scaled_height)
                            self._draw_simple_icon(
                                img, icon_rgba, icon_center, alpha=0.4)
                            elements_drawn += 1

                        # Draw yellow line for the linestring
                        if len(projected_points) >= 2:
                            yellow_color = (0, 255, 255)  # Yellow in BGR
                            points_array = np.array(projected_points, dtype=np.int32)                           
                            cv2.polylines(
                                img, [points_array], True, yellow_color, 3, cv2.LINE_AA)

            # Save result
            output_file = self.output_dir / \
                f'{camera_name}' / f'{frame_idx:010d}.jpg'
            output_file.parent.mkdir(exist_ok=True, parents=True)
            self._touch_worker(phase=PHASE_IMAGE_WRITE, frame_idx=frame_idx)
            cv2.imwrite(str(output_file), img)

            # Geotag with pose GPS coordinates
            ts = self.pose_timestamps.get(frame_idx)
            self._write_gps_exif(str(output_file), lat, lon,
                                 alt, heading, timestamp=ts)

            results[camera_name] = output_file

        return results

    def create_grid_image(self, frame_idx: int, resize_factor: float = 0.5) -> Optional[np.ndarray]:
        """
        Create a 2x3 grid of projected camera images for a given frame (no top-down view).

        Layout:
          Top row:    front_left,  front,  front_right
          Bottom row: rear_left,   rear,   rear_right

        Args:
            frame_idx: Frame index
            resize_factor: Resize factor for individual camera images

        Returns:
            Grid image as numpy array (BGR), or None if no images found
        """
        top_row_cameras = ['camera_ring_front_left',
                           'camera_ring_front', 'camera_ring_front_right']
        bottom_row_cameras = ['camera_ring_rear_left',
                              'camera_ring_rear', 'camera_ring_rear_right']

        def _load_camera_img(camera_name: str) -> np.ndarray:
            img_path = self.output_dir / camera_name / f'{frame_idx:010d}.jpg'
            if img_path.exists():
                img = cv2.imread(str(img_path))
                if img is not None:
                    if resize_factor != 1.0:
                        new_size = (int(img.shape[1] * resize_factor),
                                    int(img.shape[0] * resize_factor))
                        img = cv2.resize(img, new_size)
                    return img
            # Placeholder for missing images
            placeholder = np.zeros(
                (int(2272 * resize_factor), int(3504 * resize_factor), 3), dtype=np.uint8)
            cv2.putText(placeholder, f'MISSING: {camera_name}',
                        (int(placeholder.shape[1] // 2 -
                         300), placeholder.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (128, 128, 128), 3)
            return placeholder

        top_images = [_load_camera_img(cam) for cam in top_row_cameras]
        bottom_images = [_load_camera_img(cam) for cam in bottom_row_cameras]

        top_concat = cv2.hconcat(top_images)
        bottom_concat = cv2.hconcat(bottom_images)
        grid = cv2.vconcat([top_concat, bottom_concat])

        # Add frame number overlay
        cv2.putText(grid, f'Frame {frame_idx:010d}',
                    (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4, cv2.LINE_AA)

        return grid

    def generate_all_grid_images(self, resize_factor: float = 0.5, max_frames: Optional[int] = None,
                                 frame_step: int = 1, max_workers: int = 8, verbose: bool = True):
        """
        Generate 2x3 camera grid images for all frames from already-projected images.

        Args:
            resize_factor: Resize factor for individual camera images
            max_frames: Maximum number of frames to process (None = all)
            frame_step: Process every nth frame
            max_workers: Number of parallel threads for loading
            verbose: Print progress information
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm

        num_frames = min(
            len(self.poses), max_frames) if max_frames else len(self.poses)
        frame_indices = sorted(self.poses.keys())[:num_frames][::frame_step]

        grid_dir = self.output_dir / 'grid'
        grid_dir.mkdir(exist_ok=True, parents=True)

        if verbose:
            print(
                f"\nGenerating grid images for {len(frame_indices)} frames...")
            print(f"  Resize factor: {resize_factor}")
            print(f"  Output: {grid_dir}")

        written = 0

        def _process(fidx):
            grid_img = self.create_grid_image(fidx, resize_factor)
            if grid_img is not None:
                out_path = grid_dir / f'{fidx:010d}.jpg'
                cv2.imwrite(str(out_path), grid_img)
                # Write GPS EXIF data if pose is available
                if fidx in self.poses:
                    lat, lon, alt, heading = self._pose_to_gps(
                        self.poses[fidx])
                    ts = self.pose_timestamps.get(fidx)
                    self._write_gps_exif(
                        str(out_path), lat, lon, alt, heading, timestamp=ts)
                return True
            return False

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(
                _process, fidx): fidx for fidx in frame_indices}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Grid images",
                               disable=not verbose):
                if future.result():
                    written += 1

        if verbose:
            print(f"Grid images written: {written}/{len(frame_indices)}")
            print(f"Saved to: {grid_dir}")

    def _project_frame_wrapper(self, frame_idx: int) -> Tuple[int, Dict[str, Path]]:
        """
        Wrapper for project_frame that returns frame_idx with results.
        Needed for multiprocessing to track which frames were processed.
        """
        results = self.project_frame(frame_idx)
        return frame_idx, results

    def project_all_frames(self, max_frames: Optional[int] = None, verbose: bool = True,
                           num_processes: Optional[int] = None, use_multiprocessing: bool = True,
                           frame_step: int = 1, generate_grid: bool = True,
                           grid_resize_factor: float = 0.5):
        """
        Project map for all frames with optional multiprocessing.

        Args:
            max_frames: Maximum number of frames to process (None = all)
            verbose: Print progress information
            num_processes: Number of parallel processes (None = cpu_count - 1)
            use_multiprocessing: If True, use multiprocessing; if False, run single-threaded
            frame_step: Process every nth frame (1 = process all frames, 2 = every other frame, etc.)
            generate_grid: If True, generate 2x3 camera grid images after projection
            grid_resize_factor: Resize factor for camera images in the grid
        """
        num_frames = min(
            len(self.poses), max_frames) if max_frames else len(self.poses)

        # Generate list of frame indices to process based on frame_step
        frame_indices = list(range(0, num_frames, frame_step))

        if verbose:
            print(
                f"\nProcessing {len(frame_indices)} frames (every {frame_step} frame(s) out of {num_frames} total)...")
            print(f"Using type grouping: {self.type_grouping}")

        frames_with_projections = 0
        frames_with_top_down = 0

        if use_multiprocessing:
            import signal

            # Determine number of processes
            if num_processes is None:
                num_processes = max(1, mp.cpu_count() - 1)

            if not frame_indices:
                num_processes = 0
            else:
                num_processes = max(1, min(num_processes, len(frame_indices)))

            reserved_replacement_slots = max(4, 4*num_processes)
            total_worker_slots = max(1, num_processes + reserved_replacement_slots)

            if verbose:
                print(
                    f"Using {num_processes} supervised parallel process(es) "
                    f"with spawn start method "
                    f"(+{reserved_replacement_slots} replacement slot(s))")

            ctx = mp.get_context('spawn')
            slots_frame = ctx.Array('l', total_worker_slots, lock=False)
            slots_pid = ctx.Array('l', total_worker_slots, lock=False)
            slots_heartbeat = ctx.Array('d', total_worker_slots, lock=False)
            slots_phase = ctx.Array('i', total_worker_slots, lock=False)
            for i in range(total_worker_slots):
                slots_frame[i] = -1
                slots_pid[i] = 0
                slots_heartbeat[i] = time.time()
                slots_phase[i] = PHASE_IDLE

            task_queue = ctx.Queue(maxsize=max(2, num_processes * 2 if num_processes else 2))
            result_queue = ctx.Queue()
            projector_kwargs = self.get_worker_init_kwargs()
            workers = [None] * total_worker_slots
            worker_init_failures = [0] * total_worker_slots
            frames_remaining = deque(frame_indices)
            inflight_frames = set()
            completed_frames = set()
            timed_out = 0
            failed_frames = 0
            cpu_samples = {}
            phase_windows = {}
            quarantined_slots = set()
            parallel_abort_reason = None
            warned_reduced_parallelism = False
            report_every = max(1, len(frame_indices) // 10) if frame_indices else 1

            def _get_phase_elapsed(slot: int, pid: int, frame_idx: int, phase: int, now_ts: float) -> float:
                previous = phase_windows.get(slot)
                if previous is None or previous[:3] != (pid, frame_idx, phase):
                    phase_windows[slot] = (pid, frame_idx, phase, now_ts)
                    return 0.0
                return now_ts - previous[3]

            def _spawn_worker(slot: int):
                if slot in quarantined_slots or workers[slot] is not None:
                    return False
                proc = ctx.Process(
                    target=_worker_main,
                    args=(slot, projector_kwargs, task_queue, result_queue,
                          slots_frame, slots_pid, slots_heartbeat, slots_phase),
                )
                proc.start()
                workers[slot] = proc
                slots_pid[slot] = proc.pid or 0
                slots_frame[slot] = -1
                slots_phase[slot] = PHASE_STARTUP
                slots_heartbeat[slot] = time.time()
                cpu_samples.pop(slot, None)
                phase_windows.pop(slot, None)
                return True

            def _active_worker_count() -> int:
                count = 0
                for slot, proc in enumerate(workers):
                    if slot in quarantined_slots or proc is None:
                        continue
                    if proc.is_alive():
                        count += 1
                return count

            def _next_available_slot() -> Optional[int]:
                for slot, proc in enumerate(workers):
                    if slot in quarantined_slots:
                        continue
                    if proc is None:
                        return slot
                return None

            def _ensure_worker_capacity() -> int:
                nonlocal warned_reduced_parallelism
                if not frames_remaining:
                    warned_reduced_parallelism = False
                    return _active_worker_count()

                active_workers = _active_worker_count()
                while active_workers < num_processes:
                    next_slot = _next_available_slot()
                    if next_slot is None:
                        if not warned_reduced_parallelism:
                            print(
                                f"  WARNING: No replacement worker slots available; "
                                f"continuing with {active_workers}/{num_processes} active worker(s)",
                                flush=True)
                            warned_reduced_parallelism = True
                        break
                    if not _spawn_worker(next_slot):
                        break
                    active_workers += 1

                if active_workers >= num_processes:
                    warned_reduced_parallelism = False
                return active_workers

            def _wait_for_worker_exit(proc, timeout_sec: float) -> bool:
                deadline = time.time() + timeout_sec
                while True:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        break
                    try:
                        proc.join(timeout=min(0.2, max(0.01, remaining)))  # Ensure positive timeout
                    except Exception:
                        pass
                    if not proc.is_alive():
                        return True
                return not proc.is_alive()

            def _terminate_worker(slot: int, reset_slot: bool = True):
                proc = workers[slot]
                pid = slots_pid[slot] or (proc.pid if proc is not None else 0)
                exited = True
                final_state = None
                if proc is not None:
                    if proc.is_alive() and pid > 0:
                        for sig, wait_time in ((signal.SIGTERM, 3.0), (signal.SIGKILL, 12.0)):
                            try:
                                os.kill(pid, sig)
                            except (ProcessLookupError, OSError):
                                exited = True
                                break
                            exited = _wait_for_worker_exit(proc, wait_time)
                            if exited:
                                break
                    else:
                        exited = _wait_for_worker_exit(proc, 1.0)

                    if not exited:
                        proc_info = _read_proc_cpu_seconds(pid)
                        final_state = proc_info[1] if proc_info is not None else '?'

                if exited:
                    workers[slot] = None
                if reset_slot:
                    slots_frame[slot] = -1
                    slots_pid[slot] = 0
                    slots_phase[slot] = PHASE_IDLE
                    slots_heartbeat[slot] = time.time()
                cpu_samples.pop(slot, None)
                phase_windows.pop(slot, None)
                return exited, final_state

            def _quarantine_worker_slot(slot: int, reason: str):
                quarantined_slots.add(slot)
                worker_init_failures[slot] = 0
                print(f"  WARNING: Quarantining worker slot {slot}: {reason}", flush=True)
                active_workers = _ensure_worker_capacity()
                if active_workers == 0 and frames_remaining:
                    _abort_parallel_processing(
                        "No healthy worker slots remain; continuing remaining frames in single-threaded mode")

            def _abort_parallel_processing(reason: str):
                nonlocal parallel_abort_reason
                if parallel_abort_reason is None:
                    parallel_abort_reason = reason
                    print(f"  WARNING: {reason}", flush=True)

            def _enqueue_next_frame() -> bool:
                if not frames_remaining:
                    return False
                frame_to_queue = frames_remaining.popleft()
                try:
                    task_queue.put_nowait(frame_to_queue)
                except queue.Full:
                    frames_remaining.appendleft(frame_to_queue)
                    return False
                inflight_frames.add(frame_to_queue)
                return True

            def _fill_task_queue():
                while frames_remaining:
                    if not _enqueue_next_frame():
                        break

            try:
                _ensure_worker_capacity()

                _fill_task_queue()

                completed = 0
                while completed < len(frame_indices) and parallel_abort_reason is None:
                    try:
                        message = result_queue.get(timeout=0.5)
                    except queue.Empty:
                        message = None

                    if message is not None:
                        kind = message.get('kind')
                        if kind == 'worker_init_failed':
                            slot = int(message['slot'])
                            worker_init_failures[slot] += 1
                            print(
                                f"  WARNING: Worker slot {slot} failed to initialize:\n"
                                f"{message['error']}",
                                flush=True)
                            _terminate_worker(slot)
                            if worker_init_failures[slot] >= 3:
                                _quarantine_worker_slot(
                                    slot,
                                    "repeated initialization failures; allocating a replacement slot")
                                if parallel_abort_reason is not None:
                                    break
                                _fill_task_queue()
                                continue
                            if completed < len(frame_indices):
                                _spawn_worker(slot)
                                _fill_task_queue()
                            continue

                        if kind == 'result':
                            slot = int(message['slot'])
                            worker_init_failures[slot] = 0
                            frame_idx = int(message['frame_idx'])
                            if frame_idx in completed_frames:
                                continue
                            completed_frames.add(frame_idx)
                            inflight_frames.discard(frame_idx)
                            completed += 1

                            error = message.get('error')
                            results = message.get('results') or {}
                            if error:
                                failed_frames += 1
                                print(
                                    f"  WARNING: Frame {frame_idx} failed:\n{error}",
                                    flush=True)
                            else:
                                camera_results = {
                                    k: v for k, v in results.items()
                                    if k != 'top_down'}
                                if camera_results:
                                    frames_with_projections += 1
                                if 'top_down' in results:
                                    frames_with_top_down += 1

                            _fill_task_queue()

                            if verbose and completed % report_every == 0:
                                print(
                                    f"  Completed {completed}/{len(frame_indices)} frames",
                                    flush=True)

                    now = time.time()
                    for slot, proc in enumerate(workers):
                        if slot in quarantined_slots:
                            continue
                        if proc is None:
                            continue

                        active_frame = int(slots_frame[slot])
                        pid = int(slots_pid[slot] or 0)
                        phase = int(slots_phase[slot])
                        heartbeat_age = now - float(slots_heartbeat[slot])
                        phase_elapsed = _get_phase_elapsed(slot, pid, active_frame, phase, now)

                        if not proc.is_alive():
                            exitcode = proc.exitcode
                            if active_frame >= 0 and active_frame not in completed_frames:
                                completed_frames.add(active_frame)
                                inflight_frames.discard(active_frame)
                                completed += 1
                                failed_frames += 1
                                print(
                                    f"  WARNING: Worker PID {pid} exited while processing "
                                    f"frame {active_frame} (exitcode={exitcode})",
                                    flush=True)
                            exited, state_text = _terminate_worker(slot)
                            if not exited:
                                _quarantine_worker_slot(
                                    slot,
                                    f"PID {pid} could not be reaped after exit handling (state={state_text})")
                                if parallel_abort_reason is not None:
                                    break
                                _fill_task_queue()
                                continue
                            if completed < len(frame_indices):
                                _ensure_worker_capacity()
                                _fill_task_queue()
                            continue

                        if phase == PHASE_STARTUP and heartbeat_age > PHASE_TIMEOUTS_SEC[PHASE_STARTUP]:
                            hard_timeout = PHASE_HARD_TIMEOUTS_SEC[PHASE_STARTUP]
                            if phase_elapsed > hard_timeout:
                                print(
                                    f"  WARNING: Worker PID {pid} exceeded hard startup deadline "
                                    f"({phase_elapsed:.1f}s) — restarting worker",
                                    flush=True)
                            else:
                                print(
                                    f"  WARNING: Worker PID {pid} appears stuck during startup "
                                    f"for {heartbeat_age:.1f}s — restarting worker",
                                    flush=True)
                            exited, state_text = _terminate_worker(slot)
                            if not exited:
                                _quarantine_worker_slot(
                                    slot,
                                    f"PID {pid} could not be killed during startup timeout (state={state_text})")
                                if parallel_abort_reason is not None:
                                    break
                                _fill_task_queue()
                                continue
                            if completed < len(frame_indices):
                                _ensure_worker_capacity()
                                _fill_task_queue()
                            continue

                        if active_frame < 0 or active_frame in completed_frames:
                            if phase == PHASE_IDLE:
                                worker_init_failures[slot] = 0
                            cpu_samples.pop(slot, None)
                            phase_windows.pop(slot, None)
                            continue

                        phase_timeout = PHASE_TIMEOUTS_SEC.get(phase, 180.0)
                        hard_phase_timeout = PHASE_HARD_TIMEOUTS_SEC.get(
                            phase, max(phase_timeout * 4.0, phase_timeout + 120.0))
                        if heartbeat_age <= phase_timeout:
                            cpu_info = _read_proc_cpu_seconds(pid)
                            if cpu_info is not None:
                                cpu_samples[slot] = (now, pid, cpu_info[0])
                            continue

                        if phase_elapsed > hard_phase_timeout:
                            proc_info = _read_proc_cpu_seconds(pid)
                            proc_state = proc_info[1] if proc_info is not None else '?'
                            timed_out += 1
                            failed_frames += 1
                            completed_frames.add(active_frame)
                            inflight_frames.discard(active_frame)
                            completed += 1
                            phase_name = PHASE_NAMES.get(phase, f'phase_{phase}')
                            print(
                                f"  WARNING: Frame {active_frame} exceeded hard deadline in {phase_name} "
                                f"for {phase_elapsed:.1f}s (pid={pid}, state={proc_state}) — restarting worker",
                                flush=True)
                            exited, final_state = _terminate_worker(slot)
                            if not exited:
                                _quarantine_worker_slot(
                                    slot,
                                    f"PID {pid} for frame {active_frame} could not be killed after hard deadline "
                                    f"in {phase_name} (state={final_state or proc_state})")
                                if parallel_abort_reason is not None:
                                    break
                                _fill_task_queue()
                                continue
                            if completed < len(frame_indices):
                                _ensure_worker_capacity()
                                _fill_task_queue()
                            continue

                        cpu_info = _read_proc_cpu_seconds(pid)
                        still_progressing = False
                        if cpu_info is not None:
                            cpu_seconds, proc_state = cpu_info
                            prev_sample = cpu_samples.get(slot)
                            cpu_samples[slot] = (now, pid, cpu_seconds)
                            if prev_sample is not None and prev_sample[1] == pid:
                                elapsed_wall = max(now - prev_sample[0], 1e-6)
                                cpu_rate = (cpu_seconds - prev_sample[2]) / elapsed_wall
                                still_progressing = cpu_rate > 0.05
                            else:
                                proc_state = cpu_info[1]
                        else:
                            proc_state = '?'

                        if still_progressing:
                            continue

                        timed_out += 1
                        failed_frames += 1
                        completed_frames.add(active_frame)
                        inflight_frames.discard(active_frame)
                        completed += 1
                        phase_name = PHASE_NAMES.get(phase, f'phase_{phase}')
                        print(
                            f"  WARNING: Frame {active_frame} appears stuck in {phase_name} "
                            f"for {heartbeat_age:.1f}s (pid={pid}, state={proc_state}) — "
                            f"restarting worker",
                            flush=True)
                        exited, final_state = _terminate_worker(slot)
                        if not exited:
                            _quarantine_worker_slot(
                                slot,
                                f"PID {pid} for frame {active_frame} could not be killed after timeout "
                                f"in {phase_name} (state={final_state or proc_state})")
                            if parallel_abort_reason is not None:
                                break
                            _fill_task_queue()
                            continue
                        if completed < len(frame_indices):
                            _ensure_worker_capacity()
                            _fill_task_queue()

                    if parallel_abort_reason is not None:
                        break

                    _ensure_worker_capacity()
                    _fill_task_queue()

                if verbose and timed_out:
                    print(
                        f"  {timed_out} frame(s) were skipped due to "
                        f"phase watchdog timeouts")
            finally:
                print(f"Finished reprojection, cleaning up workers...")
                for _ in range(len(workers)):
                    try:
                        task_queue.put_nowait(None)
                    except Exception:
                        break
                for slot in range(len(workers)):
                    _terminate_worker(slot)
                try:
                    task_queue.close()
                    result_queue.close()
                except Exception:
                    pass

            if parallel_abort_reason is not None:
                remaining_frame_indices = [
                    frame_idx for frame_idx in frame_indices
                    if frame_idx not in completed_frames
                ]
                if remaining_frame_indices:
                    print(
                        f"Continuing {len(remaining_frame_indices)} remaining frame(s) in single-threaded mode...",
                        flush=True)
                    fallback_report_every = max(1, len(remaining_frame_indices) // 10)
                    for fallback_index, frame_idx in enumerate(remaining_frame_indices, start=1):
                        results = self.project_frame(frame_idx)

                        if results:
                            camera_results = {k: v for k, v in results.items() if k != 'top_down'}
                            if camera_results:
                                frames_with_projections += 1
                            if 'top_down' in results:
                                frames_with_top_down += 1

                        completed_frames.add(frame_idx)

                        if verbose and fallback_index % fallback_report_every == 0:
                            print(
                                f"  Single-thread fallback completed {fallback_index}/{len(remaining_frame_indices)} frames",
                                flush=True)
        else:
            # Single-threaded mode
            if verbose:
                print(f"Running in single-threaded mode")

            for frame_idx in frame_indices:
                results = self.project_frame(frame_idx)

                if results:
                    # Count camera projections (exclude top_down)
                    camera_results = {k: v for k,
                                      v in results.items() if k != 'top_down'}
                    if camera_results:
                        frames_with_projections += 1
                    if 'top_down' in results:
                        frames_with_top_down += 1

                if verbose and (frame_indices.index(frame_idx) + 1) % max(1, len(frame_indices) // 10) == 0:
                    print(
                        f"  Completed {frame_indices.index(frame_idx) + 1}/{len(frame_indices)} frames")

        if verbose:
            print(f"\nCompleted!")
            print(
                f"Frames with camera projections: {frames_with_projections}/{len(frame_indices)}")
            print(
                f"Frames with top-down views: {frames_with_top_down}/{len(frame_indices)}")
            print(f"Projections saved to: {self.output_dir}")

        # Generate grid images from projected camera images
        if generate_grid:
            self.generate_all_grid_images(
                resize_factor=grid_resize_factor, max_frames=max_frames,
                frame_step=frame_step, verbose=verbose)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Project lanelet2 map onto camera images.')

    # Required / path arguments
    parser.add_argument('--base-dir', type=str, required=True,
                        help='Base directory containing camera images (sequence directory).')
    parser.add_argument('--map-path', type=str, required=True,
                        help='Path to lanelet2 OSM map file.')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for projections (required unless '
                             '$KITSCENES_VIZ_OUTPUT is set; must not be inside '
                             '$KITSCENES_ROOT).')
    parser.add_argument('--config-path', type=str, default=None,
                        help='Path to sensor calibration JSON. '
                             'Default: <base-dir>/calibration/calib.json')
    parser.add_argument('--poses-path', type=str, default=None,
                        help='Path to poses file in TUM format. '
                             'Default: <base-dir>/poses_kiss_slam_tum.txt')
    parser.add_argument('--timestamp-path', type=str, default=None,
                        help='Path to reference timestamp file. '
                             'Default: <base-dir>/timestamp.reference.txt')

    # UTM projection origin
    parser.add_argument('--lat-origin', type=float, default=None,
                        help='UTM latitude origin override (default: read maps/origin.json).')
    parser.add_argument('--lon-origin', type=float, default=None,
                        help='UTM longitude origin override (default: read maps/origin.json).')

    # Processing options
    parser.add_argument('--front-only', action='store_true',
                        help='Only process the front camera image.')
    parser.add_argument('--frame-step', type=int, default=5,
                        help='Process every nth frame (default: 5).')
    parser.add_argument('--skip-top-down', action='store_true',
                        help='Skip generating matplotlib top-down views (much faster).')
    parser.add_argument('--top-down-only', action='store_true',
                        help='Only generate top-down images and skip all camera projection work.')
    parser.add_argument('--debug-local-submap', action='store_true',
                        help='Print all lanelets and linestrings in the ML-converter local submap search region.')
    parser.add_argument('--no-generate-grid', dest='generate_grid', action='store_false', default=True,
                        help='Disable generating 2x3 camera grid images after projection (enabled by default).')
    parser.add_argument('--grid-only', action='store_true',
                        help='Only generate grid images from existing projected camera images (skip projection).')
    parser.add_argument('--grid-resize-factor', type=float, default=0.5,
                        help='Resize factor for camera images in the grid (default: 0.5).')
    parser.add_argument('--num-processes', type=int, default=16,
                        help='Number of parallel processes (default: 8).')

    args = parser.parse_args()

    if args.grid_only and args.top_down_only:
        parser.error('--grid-only cannot be combined with --top-down-only.')
    if args.skip_top_down and args.top_down_only:
        parser.error('--skip-top-down cannot be combined with --top-down-only.')

    # Resolve paths with sensible defaults relative to base_dir
    base_dir = Path(args.base_dir)
    config_path = Path(args.config_path) if args.config_path else base_dir / 'calibration' / 'calib.json'
    poses_path = Path(args.poses_path) if args.poses_path else base_dir / 'poses.txt'
    timestamp_path = Path(args.timestamp_path) if args.timestamp_path else base_dir / 'timestamp.reference.txt'
    map_path = Path(args.map_path)
    from kitscenes.visualization.map_viz import resolve_map_projection_origin
    from kitscenes.visualization.output_paths import resolve_viz_output_dir
    try:
        output_dir = resolve_viz_output_dir(args.output_dir)
    except ValueError as exc:
        print(f"Error: {exc}")
        return
    lat_origin, lon_origin = resolve_map_projection_origin(
        base_dir,
        map_path=map_path,
        lat_origin=args.lat_origin,
        lon_origin=args.lon_origin,
    )

    print("=" * 70)
    print("Lanelet2 Map Projection onto Camera Images")
    print("=" * 70)

    # In --grid-only mode we only need poses; timestamps are optional for geotagging.
    if args.grid_only:
        required_files = [(poses_path, 'Poses')]
    elif args.top_down_only:
        required_files = [(map_path, 'Map'), (poses_path, 'Poses')]
    else:
        required_files = [(config_path, 'Config'), (map_path, 'Map'),
                          (poses_path, 'Poses')]

    optional_files = []
    if timestamp_path is not None:
        optional_files.append((timestamp_path, 'Timestamps'))

    # Verify files exist
    all_exist = True
    for path, name in required_files:
        status = "✓" if path.exists() else "✗"
        print(f"{status} {name}: {path}")
        if not path.exists():
            all_exist = False

    if not all_exist:
        print("\nError: One or more required files not found!")
        return

    for path, name in optional_files:
        status = "✓" if path.exists() else "✗"
        name_suffix = '' if path.exists() else ' (optional)'
        print(f"{status} {name}{name_suffix}: {path}")

    print("\n" + "=" * 70)

    generate_grid = args.generate_grid
    if args.top_down_only and generate_grid:
        print("Top-down-only mode: disabling camera grid generation.")
        generate_grid = False

    # Create projector and run
    projector = MapProjector(
        None if args.top_down_only else str(config_path), str(map_path), str(poses_path), str(timestamp_path),
        str(base_dir), str(output_dir),
        lat_origin=lat_origin, lon_origin=lon_origin,
        front_only=args.front_only, skip_top_down=args.skip_top_down,
        top_down_only=args.top_down_only,
        debug_local_submap=args.debug_local_submap)

    if args.grid_only:
        print("Grid-only mode: skipping projection, generating grid images from existing outputs.")
        projector.generate_all_grid_images(
            resize_factor=args.grid_resize_factor,
            frame_step=args.frame_step, verbose=True)
    else:
        projector.project_all_frames(
            frame_step=args.frame_step, num_processes=args.num_processes,
            generate_grid=generate_grid, grid_resize_factor=args.grid_resize_factor)


if __name__ == '__main__':
    main()
