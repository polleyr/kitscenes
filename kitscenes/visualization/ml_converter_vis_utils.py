"""
Lanelet2 ML Converter Visualization Utilities

This module provides functions for visualizing Lanelet2 map data with traffic elements,
generating labels from pose trajectories, and configuring type groupings.
"""

import numpy as np
import time
import os
import cv2
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp
import matplotlib.pyplot as plt
from PIL import Image
import cairosvg
import io
from typing import Optional
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection

import lanelet2 as ll2
from lanelet2.core import BasicPoint3d
from lanelet2.ml_converter import (
    MapDataInterface,
    LineStringType,
    TEType,
    getDefaultLineStringTypeGrouping,
    getRoadBorderMergedGrouping,
    getMapTRDefaultSimpleGrouping,
    getM3TRDefaultGrouping,
    getDefaultTETypeGrouping
)


# Type grouping options
TYPE_GROUPINGS = {
    "default": getDefaultLineStringTypeGrouping(),
    "road_border_merged": getRoadBorderMergedGrouping(),
    "maptr_simple": getMapTRDefaultSimpleGrouping(),
    "m3tr": getM3TRDefaultGrouping()
}


ARROW_SYMBOL_TE_TYPES = {
    TEType.ArrowGoStraight,
    TEType.ArrowTurnLeft,
    TEType.ArrowTurnRight,
    TEType.ArrowGoStraightOrLeft,
    TEType.ArrowGoStraightOrRight,
    TEType.ArrowTurnLeftOrRight,
}


def _normalize_polygonal_linestring_points(points: np.ndarray) -> np.ndarray:
    """Normalize polygon-like linestring points into a non-self-intersecting outer ring."""
    points_array = np.asarray(points, dtype=np.float64)
    if points_array.ndim != 2 or points_array.shape[0] < 3 or points_array.shape[1] < 2:
        return points_array

    points_xy = points_array[:, :2]
    if np.linalg.norm(points_xy[0] - points_xy[-1]) < 1e-6:
        points_xy = points_xy[:-1]

    if points_xy.shape[0] < 3:
        return points_xy

    unique_xy = np.unique(np.round(points_xy, decimals=9), axis=0)
    if unique_xy.shape[0] < 3:
        return points_xy

    try:
        hull_xy = cv2.convexHull(unique_xy.astype(np.float32))[:, 0, :]
    except Exception:
        hull_xy = None

    if hull_xy is None or hull_xy.shape[0] < 3:
        return points_xy

    hull_points = []
    for hull_point_xy in hull_xy:
        nearest_idx = int(np.argmin(np.sum((points_xy - hull_point_xy) ** 2, axis=1)))
        hull_points.append(points_xy[nearest_idx])

    ordered_points = np.asarray(hull_points, dtype=np.float64)
    start_idx = int(np.argmin(np.sum((ordered_points - points_xy[0]) ** 2, axis=1)))
    if start_idx > 0:
        ordered_points = np.roll(ordered_points, -start_idx, axis=0)

    if np.linalg.norm(ordered_points[0] - ordered_points[-1]) >= 1e-6:
        ordered_points = np.vstack([ordered_points, ordered_points[0]])

    return ordered_points


def _get_closest_polyline_point_at_least_distance(polyline: np.ndarray,
                                                  source_point: np.ndarray,
                                                  min_distance: float,
                                                  max_distance: Optional[float] = None) -> np.ndarray:
    """Return the first ordered polyline point inside a distance band, else the closest point.

    The search walks the centerline in order and returns the first point whose distance
    to source_point lies in [$min_distance$, $max_distance$]. If no such point exists,
    the closest point on the polyline is returned.
    """
    polyline_array = np.asarray(polyline, dtype=np.float64)
    source = np.asarray(source_point, dtype=np.float64)
    if polyline_array.ndim != 2 or polyline_array.shape[0] == 0:
        return source
    if polyline_array.shape[0] == 1:
        return polyline_array[0]

    segment_vectors = polyline_array[1:] - polyline_array[:-1]

    def interpolate_point(segment_idx: int, t: float) -> np.ndarray:
        return polyline_array[segment_idx] + t * segment_vectors[segment_idx]

    def project_onto_polyline():
        best_distance_sq = np.inf
        best_point = polyline_array[0].copy()

        for segment_idx, (start, direction) in enumerate(zip(polyline_array[:-1], segment_vectors)):
            segment_len_sq = float(np.dot(direction, direction))
            if segment_len_sq <= 1e-12:
                candidate_t = 0.0
                candidate = start
            else:
                candidate_t = float(np.clip(np.dot(source - start, direction) / segment_len_sq, 0.0, 1.0))
                candidate = start + candidate_t * direction

            distance_sq = float(np.sum((candidate - source) ** 2))
            if distance_sq < best_distance_sq:
                best_distance_sq = distance_sq
                best_point = candidate.copy()

        return best_point

    def get_leq_intervals(offset: np.ndarray, direction: np.ndarray, radius_sq: float):
        segment_len_sq = float(np.dot(direction, direction))
        distance_sq_at_start = float(np.dot(offset, offset))
        if segment_len_sq <= 1e-12:
            return [(0.0, 1.0)] if distance_sq_at_start <= radius_sq + 1e-9 else []

        a = segment_len_sq
        b = 2.0 * float(np.dot(offset, direction))
        c = distance_sq_at_start - radius_sq
        discriminant = b * b - 4.0 * a * c
        if discriminant < -1e-12:
            return []

        if abs(discriminant) <= 1e-12:
            t_touch = float(np.clip(-b / (2.0 * a), 0.0, 1.0))
            return [(t_touch, t_touch)]

        sqrt_disc = float(np.sqrt(discriminant))
        low = max(0.0, min((-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a)))
        high = min(1.0, max((-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a)))
        if high < 0.0 or low > 1.0 or low > high:
            return []
        return [(low, high)]

    def complement_intervals(intervals):
        if not intervals:
            return [(0.0, 1.0)]
        result = []
        cursor = 0.0
        for start_t, end_t in sorted(intervals):
            clipped_start = max(0.0, min(1.0, start_t))
            clipped_end = max(0.0, min(1.0, end_t))
            if clipped_start > cursor + 1e-9:
                result.append((cursor, clipped_start))
            cursor = max(cursor, clipped_end)
        if cursor < 1.0 - 1e-9:
            result.append((cursor, 1.0))
        return result

    def intersect_intervals(intervals_a, intervals_b):
        intersections = []
        for start_a, end_a in intervals_a:
            for start_b, end_b in intervals_b:
                start_t = max(start_a, start_b)
                end_t = min(end_a, end_b)
                if start_t <= end_t + 1e-9:
                    intersections.append((start_t, end_t))
        return intersections

    radius_sq = float(min_distance) ** 2
    max_radius_sq = float(max_distance) ** 2 if max_distance is not None else None

    for segment_idx in range(len(segment_vectors)):
        start = polyline_array[segment_idx]
        direction = segment_vectors[segment_idx]
        offset = start - source
        min_excluded_intervals = get_leq_intervals(offset, direction, radius_sq)
        candidate_intervals = complement_intervals(min_excluded_intervals)

        if max_radius_sq is not None:
            max_allowed_intervals = get_leq_intervals(offset, direction, max_radius_sq)
            candidate_intervals = intersect_intervals(candidate_intervals, max_allowed_intervals)

        if not candidate_intervals:
            continue

        for interval_start, interval_end in candidate_intervals:
            candidate_t = max(0.0, interval_start)
            if candidate_t <= interval_end + 1e-9:
                return interpolate_point(segment_idx, candidate_t)

    return project_onto_polyline()


def ls_type_to_color(type):
    """Map LineStringType to matplotlib color"""
    color_map = {
        LineStringType.RoadBorder: "green",
        LineStringType.Dashed: "blue",
        LineStringType.Solid: "blue",
        LineStringType.SolidSolid: "darkblue",
        LineStringType.SolidDashed: "royalblue",
        LineStringType.DashedSolid: "cornflowerblue",
        LineStringType.Virtual: "dimgrey",
        LineStringType.Centerline: "darkred",
        LineStringType.BikeCenterline: "orange",
        LineStringType.Unknown: "darkgray",
        LineStringType.CurbstoneHigh: "darkgreen",
        LineStringType.CurbstoneLow: "limegreen",
        LineStringType.Fence: "brown",
        LineStringType.Building: "sandybrown",
        LineStringType.Wall: "peru",
        LineStringType.DrivableArea: "lightblue",
        LineStringType.Divider: "gray",
        LineStringType.BikeMarkingDashed: "darkorange",
        LineStringType.BikeMarkingSolid: "orangered",
        LineStringType.GuardRail: "saddlebrown",
        LineStringType.PedestrianCrossing: "yellow",
        LineStringType.ZebraCrossing: "gold"
    }
    return color_map.get(type, "purple")


def default_te_icon_dir() -> str:
    """Return the default traffic-element icon directory under ``res/icon_images/``."""
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    return os.path.join(repo_root, "res", "icon_images")


def te_type_to_icon_path(type, base_path=None):
    """Map TEType to icon file path (PNG or SVG)"""
    if base_path is None:
        env_path = os.environ.get("KITSCENES_TE_ICON_PATH")
        if env_path and os.path.isdir(env_path):
            base_path = env_path
        else:
            packaged_dir = default_te_icon_dir()
            if os.path.isdir(packaged_dir):
                base_path = packaged_dir
            else:
                _script_dir = os.path.dirname(os.path.abspath(__file__))
                legacy = os.path.join(
                    _script_dir, '..', '..', '..', 'Lanelet2',
                    'lanelet2_maps', 'josm', 'style_images',
                )
                base_path = legacy
    icon_map = {
        # Traffic Lights
        TEType.TLCar: "traffic_light.png",
        TEType.TLBike: "traffic_light_bikes.svg",
        TEType.TLPedestrian: "traffic_light_pedestrians.svg",
        TEType.TLMisc: "traffic_light_misc.svg",

        # Traffic Signs - Regulatory
        TEType.TSStop: "206.png",                    # Stop sign (StVO 206)
        TEType.TSYield: "205.png",                   # Yield sign (StVO 205)
        TEType.TSNoEntry: "267.png",                 # No entry (StVO 267)
        # Right of way at intersection (StVO 301)
        TEType.TSRightOfWay: "301.png",
        TEType.TSPriorityRoad: "306.png",           # Priority road (StVO 306)
        TEType.TSOneWayStreet: "220-10.png",        # One-way street (StVO 220)
        TEType.TSRoundabout: "215.png",             # Roundabout (StVO 215)
        # Speed limit 30 as default (StVO 274)
        TEType.TSSpeedLimit: "274-30.png",
        # Pedestrian crossing sign (StVO 350)
        TEType.TSPedestrianCrossing: "350-10.png",

        # Traffic Signs - Directional
        TEType.TSTurnRight: "209.png",              # Turn right (StVO 209)
        TEType.TSTurnLeft: "209-10.png",            # Turn left (StVO 209-10)
        TEType.TSGoStraight: "209-30.png",          # Go straight (StVO 209-30)
        # Go straight or right (StVO 214)
        TEType.TSGoStraightOrRight: "214.png",
        # Go straight or left (StVO 214-10)
        TEType.TSGoStraightOrLeft: "214-10.png",
        # Turn left or right (StVO 214-30)
        TEType.TSTurnLeftOrRight: "214-30.png",
        TEType.TSPassRight: "222.png",              # Pass on right (StVO 222)
        # Pass on left (StVO 222-10)
        TEType.TSPassLeft: "222-10.png",

        # Road Markings - Arrows
        TEType.ArrowGoStraight: "pf-g.png",         # Pfeil geradeaus
        TEType.ArrowTurnLeft: "pf-l.png",           # Pfeil links
        TEType.ArrowTurnRight: "pf-r.png",          # Pfeil rechts
        TEType.ArrowGoStraightOrLeft: "pf-gl.png",  # Pfeil geradeaus/links
        TEType.ArrowGoStraightOrRight: "pf-gr.png",  # Pfeil geradeaus/rechts
        TEType.ArrowTurnLeftOrRight: "pf-lr.png",   # Pfeil links/rechts

        # Road Markings - Symbols
        TEType.BikeSymbol: "bike.png",
        TEType.BusSymbol: "bus.png",
        TEType.Symbol30: "30.png",
        TEType.Symbol50: "50.png",
        TEType.Symbol70: "70.png",

        # Misc
        TEType.TSMisc: "traffic_sign.png",
        TEType.Unknown: "traffic_sign.png"
    }

    icon_file = icon_map.get(type, "traffic_sign.png")
    full_path = os.path.join(base_path, icon_file)

    # Check if file exists, otherwise try fallback icons
    if os.path.exists(full_path):
        return full_path

    # Fallback: use generic traffic_light.png for any traffic light type
    tl_types = {TEType.TLCar, TEType.TLBike, TEType.TLPedestrian, TEType.TLMisc}
    if type in tl_types:
        fallback = os.path.join(base_path, "traffic_light.png")
        if os.path.exists(fallback):
            return fallback

    # Fallback: use generic traffic_sign.png for any traffic sign type
    fallback = os.path.join(base_path, "traffic_sign.png")
    if os.path.exists(fallback):
        return fallback

    return None


def get_grouping_description(name):
    """Get human-readable description of type grouping"""
    descriptions = {
        "default": "Each type in its own group (no merging)",
        "road_border_merged": "RoadBorder merged with Fence, CurbstoneHigh, CurbstoneLow",
        "maptr_simple": "RoadBorder merged + all lane dividers merged (except Virtual)",
        "m3tr": "RoadBorder merged + Solid types merged (Solid, SolidSolid, SolidDashed, DashedSolid)"
    }
    return descriptions.get(name, "Unknown grouping")


def get_file_name(frame_no):
    """Generate zero-padded filename for frame number"""
    frame_no = int(frame_no)
    return str(frame_no).zfill(10)


def get_map_data(ll2_map, x, y, z, yaw, pitch, roll, ignore_map_ele=False,
                 type_grouping="default", extent_long=30, extent_lat=15,
                 n_points_lanes=20, n_points_te=20):
    """Extract MapData for a given pose using the new API

    Args:
        ll2_map: Lanelet2 map
        x, y, z: Position coordinates
        yaw, pitch, roll: Orientation angles
        ignore_map_ele: Ignore elevation in map elements
        type_grouping: Type grouping strategy ("default", "road_border_merged", "maptr_simple", "m3tr")
        extent_long: Longitudinal extent of submap extraction
        extent_lat: Lateral extent of submap extraction
        n_points_lanes: Number of points for lane resampling
        n_points_te: Number of points for traffic element resampling

    Returns:
        MapData object or error string
    """
    pos = BasicPoint3d(x, y, z)
    config = MapDataInterface.Configuration()
    config.ignoreMapElevation = ignore_map_ele
    config.submapExtentLongitudinal = extent_long
    config.submapExtentLateral = extent_lat
    config.nPointsLanes = n_points_lanes
    config.nPointsTE = n_points_te
    config.lineStringTypeGrouping = TYPE_GROUPINGS.get(
        type_grouping, getDefaultLineStringTypeGrouping())
    config.teTypeGrouping = getDefaultTETypeGrouping()

    mDataIf = MapDataInterface(ll2_map, config)
    mDataIf.setCurrPosAndExtractSubmap(pos, yaw, pitch, roll)
    try:
        # Updated API: mapData instead of laneData
        mData = mDataIf.mapData(True)
        return mData
    except Exception as e:
        return str(e)


def generate_labels_from_poses(ll2_map_path, poses_file_path, timestamps_file_path,
                               out_path, lat_origin, lon_origin,
                               type_grouping="default", num_processes=8):
    """Generate MapData labels from legacy pose/timestamp files on disk.

    For KITScenes scenes (``poses.txt`` + reference timestamps + ``maps/``),
    prefer :func:`~kitscenes.visualization.map_viz.generate_map_labels_for_scene`.

    Args:
        ll2_map_path: Path to Lanelet2 map file
        poses_file_path: Path to poses file
        timestamps_file_path: Path to timestamps file
        out_path: Output directory for labels
        lat_origin: Latitude origin for projection
        lon_origin: Longitude origin for projection
        type_grouping: Type grouping strategy
        num_processes: Number of parallel processes (not used currently)

    Returns:
        Tuple of (map_data_list, interpolated_translations)
    """
    ll2_projector = ll2.projection.UtmProjector(
        ll2.io.Origin(lat_origin, lon_origin))
    ll2_map = ll2.io.load(ll2_map_path, ll2_projector)

    translations = []
    rotations = []
    pose_timestamps = []
    try:
        with open(poses_file_path, 'r') as file:
            for line in file:
                try:
                    values = np.array(
                        [float(num) for num in line.strip().split(' ') if num != ''])
                    pose_timestamps.append(values[2])
                    rotations.append([values[5], values[4], values[3]])
                    translations.append(values[6:9])
                except ValueError:
                    print(
                        f"Skipping invalid number(s) on line: '{line.strip().split(' ')}'")
    except FileNotFoundError:
        print(f"File '{poses_file_path}' not found.")
        return [], []
    except Exception as e:
        print(f"An error occurred: {e}")
        return [], []

    translations = np.row_stack(translations)
    rotations = Rotation.from_euler(
        'zyx', np.row_stack(rotations), degrees=False)
    pose_timestamps = np.row_stack(pose_timestamps).flatten()

    timestamps = []
    try:
        with open(timestamps_file_path, 'r') as file:
            for line in file:
                try:
                    value = np.array(float(line))
                    timestamps.append(value)
                except ValueError:
                    print(f"Skipping invalid number on line: '{line.strip()}'")
    except FileNotFoundError:
        print(f"File '{timestamps_file_path}' not found.")
        return [], []
    except Exception as e:
        print(f"An error occurred: {e}")
        return [], []

    timestamps = np.row_stack(timestamps).flatten()
    timestamps[timestamps > np.max(pose_timestamps)] = np.max(pose_timestamps)
    timestamps[timestamps < np.min(pose_timestamps)] = np.min(pose_timestamps)

    # Interpolate positions
    cs_x = CubicSpline(
        pose_timestamps, translations[:, 0].flatten(), bc_type='clamped')
    cs_y = CubicSpline(
        pose_timestamps, translations[:, 1].flatten(), bc_type='clamped')
    cs_z = CubicSpline(
        pose_timestamps, translations[:, 2].flatten(), bc_type='clamped')
    translations_ip = np.column_stack([
        cs_x(timestamps.flatten()),
        cs_y(timestamps.flatten()),
        cs_z(timestamps.flatten())
    ])

    # Interpolate orientations
    slerp = Slerp(pose_timestamps, rotations)
    rotations_ip = slerp(timestamps)
    rotations_ip = rotations_ip.as_euler('zyx', degrees=False)

    # Generate map data for all poses
    start_time = time.time()
    m_data_list = [
        get_map_data(
            ll2_map,
            translations_ip[i][0],
            translations_ip[i][1],
            translations_ip[i][2],
            rotations_ip[i][0],
            rotations_ip[i][1],
            rotations_ip[i][2],
            type_grouping=type_grouping
        )
        for i in range(len(timestamps))
    ]
    print(
        f"Generated {len(m_data_list)} map data samples in {time.time() - start_time:.2f} seconds")

    return m_data_list, translations_ip


def _setup_coordinate_transform(orientation, lim_lat=30, lim_lon=60):
    """Set up coordinate transformation functions based on plot orientation

    Args:
        orientation: 'vertical' or 'horizontal'

    Returns:
        Tuple of (transform_coords, transform_point, xlim_range, ylim_range, car_rotation, legend_loc)
    """
    if orientation == 'vertical':
        # Vertical: swap and negate x-coordinate for 90-degree rotation
        def transform_coords(line):
            return -line[:, 1], line[:, 0]

        def transform_point(point):
            return -point[1], point[0]
        xlim_range = [-lim_lat, lim_lat]
        ylim_range = [-lim_lon, lim_lon]
        car_rotation = -90
        legend_loc = 'lower right'
    elif orientation == 'horizontal':
        # Horizontal: use original coordinates
        def transform_coords(line):
            return line[:, 0], line[:, 1]

        def transform_point(point):
            return point[0], point[1]
        xlim_range = [-lim_lon, lim_lon]
        ylim_range = [-lim_lat, lim_lat]
        car_rotation = 180
        legend_loc = 'upper right'
    else:
        raise RuntimeError(
            "Orientation has to be either vertical or horizontal!")

    return transform_coords, transform_point, xlim_range, ylim_range, car_rotation, legend_loc


def _plot_car_overlay(car_img_path, car_rotation):
    """Plot car image overlay on the map

    Args:
        car_img_path: Path to car image file
        car_rotation: Rotation angle for the car image
    """
    if not car_img_path:
        return

    try:
        from PIL import ImageOps
        scale_factor = 600
        car_img = Image.open(car_img_path)
        # Rotate image based on orientation
        if car_rotation != 0:
            car_img = car_img.rotate(car_rotation, expand=True)
        plt.imshow(
            car_img,
            extent=[
                -car_img.width/scale_factor, car_img.width/scale_factor,
                -car_img.height/scale_factor, car_img.height/scale_factor
            ],
            zorder=10)
    except FileNotFoundError:
        print(f"Warning: Car image not found at {car_img_path}")


def _plot_road_borders(tfData, transform_coords, show_drivable_area, show_other_road_border_types,
                       types_present, stats):
    """Plot road borders and similar elements

    Args:
        tfData: Tensor instance data
        transform_coords: Coordinate transformation function
        show_drivable_area: Whether to show drivable area
        show_other_road_border_types: Whether to show other road border types
        types_present: Set to track which types are present
        stats: Statistics dictionary

    Returns:
        Number of elements plotted
    """
    border_types = [LineStringType.RoadBorder, LineStringType.CurbstoneHigh,
                    LineStringType.CurbstoneLow, LineStringType.Fence,
                    LineStringType.GuardRail, LineStringType.DrivableArea,
                    LineStringType.Building, LineStringType.Wall]

    numel = 0
    for rb_type in border_types:
        if rb_type == LineStringType.DrivableArea and not show_drivable_area:
            continue
        elif rb_type != LineStringType.DrivableArea and not show_other_road_border_types:
            continue
        road_borders = tfData.compoundLineStringsOfType(rb_type)
        if road_borders:
            types_present.add(rb_type)
        for line in road_borders:
            x, y = transform_coords(line)
            plt.plot(x, y, color=ls_type_to_color(
                rb_type), linewidth=7.0, zorder=5)
            numel += 1
            stats["road_borders_or_similar"] += 1

    return numel


def _plot_lane_dividers(tfData, transform_coords, types_present, stats):
    """Plot lane dividers with proper styling

    Args:
        tfData: Tensor instance data
        transform_coords: Coordinate transformation function
        types_present: Set to track which types are present
        stats: Statistics dictionary

    Returns:
        Number of elements plotted
    """
    divider_types = [LineStringType.Dashed, LineStringType.Solid,
                     LineStringType.SolidSolid, LineStringType.SolidDashed,
                     LineStringType.DashedSolid, LineStringType.Virtual]

    numel = 0
    for div_type in divider_types:
        dividers = tfData.compoundLineStringsOfType(div_type)
        if dividers:
            types_present.add(div_type)
        for line in dividers:
            x, y = transform_coords(line)
            # Draw gray background for solid/dashed distinction
            if div_type == LineStringType.Dashed:
                plt.plot(x, y, color='gray', linewidth=12.0,
                         linestyle=(0, (5, 5)), zorder=5)
            elif div_type in [LineStringType.Solid, LineStringType.SolidSolid]:
                plt.plot(x, y, color='gray', linewidth=12.0, zorder=5)

            # Draw colored line on top
            plt.plot(x, y, color=ls_type_to_color(
                div_type), linewidth=3.0, zorder=5)
            numel += 1
            stats["dividers"] += 1

    return numel


def _plot_centerlines(tfData, transform_coords, show_bike_lanes, types_present, stats):
    """Plot vehicle and bike centerlines

    Args:
        tfData: Tensor instance data
        transform_coords: Coordinate transformation function
        show_bike_lanes: Whether to show bike centerlines
        types_present: Set to track which types are present
        stats: Statistics dictionary

    Returns:
        Number of elements plotted
    """
    numel = 0

    # Plot vehicle centerlines
    centerlines = tfData.compoundLineStringsOfType(LineStringType.Centerline)
    if centerlines:
        types_present.add(LineStringType.Centerline)
    for line in centerlines:
        x, y = transform_coords(line)
        plt.plot(x, y, color=ls_type_to_color(LineStringType.Centerline),
                 linestyle='dashed', linewidth=2.0, zorder=4)
        numel += 1
        stats["centerlines"] += 1

    # Plot bike centerlines
    if show_bike_lanes:
        bike_centerlines = tfData.compoundLineStringsOfType(
            LineStringType.BikeCenterline)
        if bike_centerlines:
            types_present.add(LineStringType.BikeCenterline)
        for line in bike_centerlines:
            x, y = transform_coords(line)
            plt.plot(x, y, color=ls_type_to_color(LineStringType.BikeCenterline),
                     linestyle='dotted', linewidth=2.5, zorder=4)
            numel += 1
            stats["bike_centerlines"] += 1

    return numel


def _plot_pedestrian_crossings(tfData, transform_coords, types_present):
    """Plot pedestrian crossings

    Args:
        tfData: Tensor instance data
        transform_coords: Coordinate transformation function
        types_present: Set to track which types are present

    Returns:
        Number of elements plotted
    """
    numel = 0
    for crossing_type in (LineStringType.ZebraCrossing, LineStringType.PedestrianCrossing):
        crossings = tfData.compoundLineStringsOfType(crossing_type)
        if crossings:
            types_present.add(crossing_type)
        for line in crossings:
            normalized_line = _normalize_polygonal_linestring_points(line)
            x, y = transform_coords(normalized_line)
            plt.plot(x, y, color=ls_type_to_color(crossing_type),
                     linewidth=5.0, zorder=6)
            numel += 1

    return numel


def _plot_traffic_element_icon(te_line, te_type, icon_path, icon_size, transform_point, transform_coords):
    """Plot a single traffic element with icon or fallback marker

    Args:
        te_line: Traffic element line data
        te_type: Traffic element type
        icon_path: Path to icon file
        icon_size: Size of icon in plot units
        transform_point: Point transformation function
        transform_coords: Coordinate transformation function
    """
    # plot the line for stop lines
    if te_type == TEType.StopLine:
        x, y = transform_coords(te_line)
        plt.plot(x, y, color='red', linewidth=4.0, zorder=7)
    elif icon_path:
        # Load and display icon
        try:
            centroid = te_line.mean(axis=0)

            # Handle SVG files
            if icon_path.endswith('.svg'):
                # Convert SVG to PNG in memory
                png_data = cairosvg.svg2png(
                    url=icon_path, output_width=128, output_height=128)
                icon_img = Image.open(io.BytesIO(png_data))
            else:
                # Load PNG directly
                icon_img = Image.open(icon_path)

            # Get icon dimensions and maintain aspect ratio
            img_width, img_height = icon_img.size
            aspect_ratio = img_width / img_height

            if aspect_ratio > 1:
                # Wider than tall
                width = icon_size
                height = icon_size / aspect_ratio
            else:
                # Taller than wide
                width = icon_size * aspect_ratio
                height = icon_size

            cx, cy = transform_point(centroid)
            extent = [cx - width/2, cx + width/2, cy - height/2, cy + height/2]
            plt.imshow(icon_img, extent=extent, zorder=8, alpha=0.9)

        except Exception as e:
            # Fallback to marker if icon loading fails
            cx, cy = transform_point(centroid)
            plt.plot(cx, cy, marker='o', color='gray', markersize=6,
                     markeredgecolor='black', markeredgewidth=0.5, zorder=8)
    else:
        # No icon available, use fallback marker
        centroid = te_line.mean(axis=0)
        cx, cy = transform_point(centroid)
        plt.plot(cx, cy, marker='o', color='gray', markersize=6,
                 markeredgecolor='black', markeredgewidth=0.5, zorder=8)


def _plot_traffic_elements(tfData, transform_point, transform_coords, icon_size, types_present, stats,
                           mData=None, exclude_te_ids=None):
    """Plot traffic elements with icons

    Args:
        tfData: Tensor instance data
        transform_point: Point transformation function
        transform_coords: Coordinate transformation function
        icon_size: Size of traffic element icons
        types_present: Set to track which types are present
        stats: Statistics dictionary
        mData: MapData object (optional, needed for ID-based filtering)
        exclude_te_ids: Set of lanelet2 linestring IDs to exclude (optional)
    """
    # Check for stop lines
    stop_line_instances = tfData.teInstancesOfType(TEType.StopLine)
    if stop_line_instances:
        types_present.add(TEType.StopLine)

    # All traffic element types
    te_types = [
        TEType.StopLine, TEType.TLCar, TEType.TLBike, TEType.TLPedestrian, TEType.TLMisc,
        TEType.ArrowGoStraight, TEType.ArrowTurnLeft, TEType.ArrowTurnRight,
        TEType.ArrowGoStraightOrLeft, TEType.ArrowGoStraightOrRight, TEType.ArrowTurnLeftOrRight,
        TEType.TSStop, TEType.TSYield, TEType.TSSpeedLimit, TEType.TSNoEntry,
        TEType.TSRightOfWay, TEType.TSPriorityRoad, TEType.TSRoundabout,
        TEType.TSTurnRight, TEType.TSTurnLeft, TEType.TSGoStraight,
        TEType.TSGoStraightOrRight, TEType.TSGoStraightOrLeft, TEType.TSTurnLeftOrRight,
        TEType.TSPassRight, TEType.TSPassLeft, TEType.TSPedestrianCrossing,
        TEType.BikeSymbol, TEType.BusSymbol,
        TEType.Symbol30, TEType.Symbol50, TEType.Symbol70,
        TEType.TSMisc, TEType.Unknown
    ]

    for te_type in te_types:
        if exclude_te_ids and mData is not None:
            # Use mData-level API to get instances with IDs for filtering
            te_instances_with_ids = mData.teInstancesOfType(te_type)
            te_instances = []
            for te_id, te_instance in te_instances_with_ids.items():
                if te_id not in exclude_te_ids:
                    # Get 2D point data from the instance
                    matrices = te_instance.pointMatrices(True)  # pointsIn2d=True
                    for mat in matrices:
                        te_instances.append(mat)
        else:
            te_instances = tfData.teInstancesOfType(te_type)
        icon_path = te_type_to_icon_path(te_type)

        for te_line in te_instances:
            _plot_traffic_element_icon(te_line, te_type, icon_path, icon_size,
                                       transform_point, transform_coords)
            stats["traffic_elements"] += 1


def _plot_te_edges(tfData, transform_point, show_te_edges, show_te_to_centerline_edges,
                   xlim_range=None, ylim_range=None, max_arrow_length_fraction=None):
    """Plot edges between traffic elements and centerlines

    Args:
        tfData: Tensor instance data
        transform_point: Point transformation function
        show_te_edges: Whether to show TE to TE edges
        show_te_to_centerline_edges: Whether to show TE to centerline edges
        xlim_range: X-axis limits for bounds checking
        ylim_range: Y-axis limits for bounds checking
        max_arrow_length_fraction: Skip arrows longer than this fraction of the
            larger figure span. If None, draw arrows regardless of length.
    """
    # Helper function to check if point is within bounds
    def is_within_bounds(x, y):
        if xlim_range is None or ylim_range is None:
            return True
        return (xlim_range[0] <= x <= xlim_range[1] and
                ylim_range[0] <= y <= ylim_range[1])

    max_arrow_length = None
    if (max_arrow_length_fraction is not None and xlim_range is not None and
            ylim_range is not None):
        figure_span = max(xlim_range[1] - xlim_range[0], ylim_range[1] - ylim_range[0])
        if figure_span > 0:
            max_arrow_length = float(max_arrow_length_fraction) * float(figure_span)

    def should_draw_arrow(source_xy, target_xy):
        if not (is_within_bounds(source_xy[0], source_xy[1]) and
                is_within_bounds(target_xy[0], target_xy[1])):
            return False
        if max_arrow_length is None:
            return True
        arrow_length = float(np.hypot(target_xy[0] - source_xy[0], target_xy[1] - source_xy[1]))
        return arrow_length <= max_arrow_length

    # Get centerlines for TE to centerline edges
    centerlines = tfData.compoundLineStringsOfType(LineStringType.Centerline)

    # Plot TE to TE edges (e.g., traffic light to stop line)
    if show_te_edges:
        te_to_te_edges = tfData.teToTEIndexEdges
        for edge in te_to_te_edges:
            source_type, source_idx, target_type, target_idx = edge

            if source_type in ARROW_SYMBOL_TE_TYPES or target_type in ARROW_SYMBOL_TE_TYPES:
                continue

            source_matrices = tfData.teInstancesOfType(source_type)
            target_matrices = tfData.teInstancesOfType(target_type)

            if source_idx < len(source_matrices) and target_idx < len(target_matrices):
                source_mat = source_matrices[source_idx]
                target_mat = target_matrices[target_idx]

                # Use center point of each TE for the arrow
                source_centroid = source_mat.mean(axis=0)
                target_centroid = target_mat.mean(axis=0)

                sx, sy = transform_point(source_centroid)
                tx, ty = transform_point(target_centroid)

                # Only plot arrow if it stays within bounds and is not overly long
                if should_draw_arrow((sx, sy), (tx, ty)):
                    # Plot arrow from source to target
                    plt.annotate('', xy=(tx, ty), xytext=(sx, sy),
                                 arrowprops=dict(arrowstyle='->', color='darkgray',
                                                 lw=3, alpha=0.7), zorder=1)

    # Plot TE to centerline edges (e.g., stop line to lanelet, arrows to lanelet)
    if show_te_to_centerline_edges:
        te_to_cl_edges = tfData.teToCenterlineIndexEdges
        for edge in te_to_cl_edges:
            source_type, source_idx, centerline_idx = edge

            if source_type in ARROW_SYMBOL_TE_TYPES:
                continue

            source_matrices = tfData.teInstancesOfType(source_type)

            if source_idx < len(source_matrices) and centerline_idx < len(centerlines):
                source_mat = source_matrices[source_idx]
                centerline_mat = centerlines[centerline_idx]

                # Skip if centerline is empty (no points)
                if len(centerline_mat) == 0:
                    continue

                # Use center point of TE
                source_centroid = source_mat.mean(axis=0)
                sx, sy = transform_point(source_centroid)

                target_point = _get_closest_polyline_point_at_least_distance(
                    centerline_mat, source_centroid, min_distance=3.0, max_distance=5.0)
                tx, ty = transform_point(target_point)

                # Only plot arrow if it stays within bounds and is not overly long
                if should_draw_arrow((sx, sy), (tx, ty)):
                    # Plot arrow from TE to centerline
                    plt.annotate('', xy=(tx, ty), xytext=(sx, sy),
                                 arrowprops=dict(arrowstyle='->', color='darkgray',
                                                 lw=2, alpha=0.7), zorder=1)


def _create_legend(types_present, legend_loc):
    """Create and display legend for map elements

    Args:
        types_present: Set of types present in the plot
        legend_loc: Location for the legend
    """
    legend_handles = []
    legend_labels = []

    # Define legend entries in display order
    legend_entries = [
        (LineStringType.RoadBorder, "Road Border", '-', 3),
        (LineStringType.CurbstoneHigh, "Curbstone High", '-', 3),
        (LineStringType.CurbstoneLow, "Curbstone Low", '-', 3),
        (LineStringType.Fence, "Fence", '-', 3),
        (LineStringType.Building, "Fence", '-', 3),
        (LineStringType.Wall, "Fence", '-', 3),
        (LineStringType.GuardRail, "Guard Rail", '-', 3),
        (LineStringType.DrivableArea, "Drivable Area", '-', 3),
        (LineStringType.Dashed, "Dashed", '-', 2),
        (LineStringType.Solid, "Solid", '-', 2),
        (LineStringType.SolidSolid, "Solid-Solid", '-', 2),
        (LineStringType.SolidDashed, "Solid-Dashed", '-', 2),
        (LineStringType.DashedSolid, "Dashed-Solid", '-', 2),
        (LineStringType.Virtual, "Virtual", '-', 2),
        (LineStringType.Centerline, "Centerline", '--', 2),
        (LineStringType.BikeCenterline, "Bike Centerline", ':', 2),
        (LineStringType.PedestrianCrossing, "Pedestrian Crossing", '-', 3),
        (LineStringType.ZebraCrossing, "Zebra Crossing", '-', 3),
        (TEType.StopLine, "Stop Line", '-', 3),
    ]

    # Define divider types that need gray background
    divider_types_with_bg = [
        LineStringType.Dashed, LineStringType.Solid,
        LineStringType.SolidSolid, LineStringType.SolidDashed,
        LineStringType.DashedSolid
    ]

    for el_type, label, linestyle, linewidth in legend_entries:
        if el_type in types_present:
            # Handle stop lines specially (string marker instead of LineStringType)
            if el_type == TEType.StopLine:
                color = 'red'
            else:
                color = ls_type_to_color(el_type)

            # Use Line2D for divider types to mirror actual plot rendering
            if el_type in divider_types_with_bg:
                # Determine linestyle for gray background
                if el_type == LineStringType.Dashed:
                    bg_linestyle = (0, (1, 1))
                else:
                    bg_linestyle = '-'

                legend_handles.append(
                    (Line2D([0], [0], color='gray', linewidth=8.0, linestyle=bg_linestyle),
                     Line2D([0], [0], color=color, linestyle=linestyle, linewidth=linewidth))
                )
            else:
                # Use standard Line2D for non-divider types
                legend_handles.append(
                    Line2D([0], [0], color=color,
                           linestyle=linestyle, linewidth=linewidth)
                )

            legend_labels.append(label)

    if legend_handles:
        plt.legend(legend_handles, legend_labels, loc=legend_loc, fontsize=15,
                   framealpha=0.9, edgecolor='black').set_zorder(100)


def _configure_plot_appearance(xlim_range, ylim_range):
    """Configure plot appearance settings

    Args:
        xlim_range: X-axis limits
        ylim_range: Y-axis limits
    """
    plt.gca().set_aspect('equal')
    plt.xlim(xlim_range)
    plt.ylim(ylim_range)
    plt.gca().set(yticklabels=[], xticklabels=[], ylabel=None)
    plt.gca().tick_params(left=False, bottom=False)
    for side in ['top', 'right', 'bottom', 'left']:
        plt.gca().spines[side].set_visible(False)


def plot_map_data(mData, car_img_path=None, show_traffic_elements=True,
                  show_bike_lanes=True, show_stats=True, show_drivable_area=False,
                  show_other_road_border_types=True, icon_size=3.0, show_legend=True,
                  orientation='vertical', show_te_edges=True, show_te_to_centerline_edges=True,
                  lim_lat=30, lim_lon=60, exclude_te_ids=None,
                  max_arrow_length_fraction=0.15):
    """Plot MapData using the new tensor-based API with actual traffic sign icons

    Args:
        mData: MapData object
        car_img_path: Path to car image overlay (optional)
        show_traffic_elements: Whether to show traffic elements
        show_bike_lanes: Whether to show bike centerlines
        show_stats: Whether to print statistics
        show_drivable_area: Whether to show the drivable area (this can overlap with other road border types)
        show_other_road_border_types: Whether to show other road border types
        icon_size: Size of traffic element icons in plot units (controls max dimension)
        show_legend: Whether to show legend for line types
        orientation: Plot orientation - 'vertical' or 'horizontal' (default: 'vertical')
        show_te_edges: Whether to show traffic element to traffic element edges (e.g., traffic light to stop line)
        show_te_to_centerline_edges: Whether to show traffic element to centerline edges (e.g., stop line to lanelet)
        max_arrow_length_fraction: Skip grey edge arrows longer than this fraction of the larger figure span.
    """
    # Get tensor data (all as numpy arrays)
    tfData = mData.getTensorInstanceData(pointsIn2d=True, ignoreBuffer=True)

    # Set up coordinate transformation based on orientation
    transform_coords, transform_point, xlim_range, ylim_range, car_rotation, legend_loc = \
        _setup_coordinate_transform(
            orientation, lim_lat=lim_lat, lim_lon=lim_lon)

    # Display car image if provided
    _plot_car_overlay(car_img_path, car_rotation)

    # Initialize statistics and tracking
    stats = {
        "road_borders_or_similar": 0,
        "dividers": 0,
        "centerlines": 0,
        "bike_centerlines": 0,
        "traffic_elements": 0
    }
    types_present = set()

    # Plot all map elements
    numel = 0
    numel += _plot_road_borders(tfData, transform_coords, show_drivable_area,
                                show_other_road_border_types, types_present, stats)
    numel += _plot_lane_dividers(tfData,
                                 transform_coords, types_present, stats)
    numel += _plot_centerlines(tfData, transform_coords,
                               show_bike_lanes, types_present, stats)
    numel += _plot_pedestrian_crossings(tfData,
                                        transform_coords, types_present)

    # Plot traffic elements with icons
    if show_traffic_elements:
        _plot_traffic_elements(tfData, transform_point, transform_coords, icon_size,
                               types_present, stats, mData=mData, exclude_te_ids=exclude_te_ids)

    # Plot traffic element edges
    if show_te_edges or show_te_to_centerline_edges:
        _plot_te_edges(tfData, transform_point, show_te_edges,
                       show_te_to_centerline_edges, xlim_range, ylim_range,
                       max_arrow_length_fraction=max_arrow_length_fraction)

    # Create legend
    if show_legend:
        _create_legend(types_present, legend_loc)

    # Configure plot appearance
    _configure_plot_appearance(xlim_range, ylim_range)

    # Print statistics
    if show_stats:
        print(f"Total labels: {numel}")
        print(f"Road borders (or similar): {stats['road_borders_or_similar']}")
        print(f"Lane dividers: {stats['dividers']}")
        print(f"Vehicle centerlines: {stats['centerlines']}")
        print(f"Bike centerlines: {stats['bike_centerlines']}")
        print(f"Traffic elements: {stats['traffic_elements']}")

    return stats
