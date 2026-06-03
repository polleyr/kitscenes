"""HD map API for the kitscenes dataset, backed by Lanelet2.

Provides :class:`SceneMap` for loading Lanelet2 ``.osm`` maps, building
routing graphs, and performing spatial queries (lane segments in ROI,
drivable area, crosswalks, image projection).

Lanelet2 is an **optional** dependency.  Importing this module without
lanelet2 installed will raise :class:`ImportError` only when
:class:`SceneMap` is instantiated, not at module import time.

Example::

    from kitscenes.map_api import load_scene_map

    scene_map = load_scene_map(scene_path)  # reads maps/map.osm + maps/origin.json
    if scene_map is not None:
        lanelets_nearby = scene_map.get_lanelets_in_roi(center=np.array([0, 0]), radius=50.0)
"""

from __future__ import annotations

import dataclasses
import logging
import warnings
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy Lanelet2 import — keeps the module importable without lanelet2
# ---------------------------------------------------------------------------

_LANELET2_AVAILABLE: bool = False

try:
    import lanelet2  # type: ignore[import-untyped]
    from lanelet2.core import (  # type: ignore[import-untyped]
        BasicPoint2d,
        BoundingBox2d,
        GPSPoint,
        LaneletMap,
    )
    from lanelet2.io import Origin, loadRobust  # type: ignore[import-untyped]
    from lanelet2.projection import UtmProjector  # type: ignore[import-untyped]

    _LANELET2_AVAILABLE = True
except ImportError:
    pass


def _require_lanelet2() -> None:
    """Raise a helpful error if lanelet2 is not installed."""
    if not _LANELET2_AVAILABLE:
        raise ImportError(
            "lanelet2 is required for kitscenes.map_api but is not installed. "
            "Install a pre-built wheel from res/ml_converter_wheels/ or build "
            "from source.  See https://github.com/fzi-forschungszentrum-informatik/Lanelet2"
        )


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class LaneSegment:
    """A single lane segment extracted from the HD map.

    Attributes:
        lanelet_id: Lanelet2 primitive ID.
        left_boundary: (N, 2) XY polyline of the left lane boundary.
        right_boundary: (M, 2) XY polyline of the right lane boundary.
        centerline: (K, 2) XY polyline of the computed centerline.
        speed_limit_mps: Speed limit in m/s, or ``None`` if unknown.
    """

    lanelet_id: int
    left_boundary: NDArray[np.float64]
    right_boundary: NDArray[np.float64]
    centerline: NDArray[np.float64]
    speed_limit_mps: Optional[float]


# ---------------------------------------------------------------------------
# SceneMap
# ---------------------------------------------------------------------------


class SceneMap:
    """HD map for a single scene, backed by Lanelet2.

    The map is loaded lazily: the ``.osm`` file is parsed on first access to
    :attr:`lanelet_map` (or any method that needs it).

    Args:
        map_path: Path to the Lanelet2 ``.osm`` file.
        origin_lat: WGS-84 latitude of the UTM projection origin.
        origin_lon: WGS-84 longitude of the UTM projection origin.

    Raises:
        ImportError: If lanelet2 is not installed.
        FileNotFoundError: If *map_path* does not exist.
    """

    def __init__(
        self,
        map_path: str | Path,
        origin_lat: float,
        origin_lon: float,
    ) -> None:
        _require_lanelet2()

        self._map_path = Path(map_path)
        if not self._map_path.exists():
            raise FileNotFoundError(f"Map file not found: {self._map_path}")

        self._origin_lat = origin_lat
        self._origin_lon = origin_lon

    # -- representation -----------------------------------------------------

    def __repr__(self) -> str:
        map_state = f"lanelets={len(self.lanelet_map.laneletLayer)}" if "lanelet_map" in self.__dict__ else "not loaded"
        return (
            f"SceneMap(path={self._map_path.name!r}, "
            f"map={map_state}, "
            f"origin=({self._origin_lat:.4f}, {self._origin_lon:.4f}))"
        )

    # -- lazy properties ----------------------------------------------------

    @cached_property
    def projector(self) -> "UtmProjector":
        """UTM projector anchored at the configured origin."""
        return UtmProjector(Origin(self._origin_lat, self._origin_lon))

    @cached_property
    def utm_origin(self) -> NDArray[np.float64]:
        """Absolute UTM coordinates of the map origin.

        The Lanelet2 ``UtmProjector`` maps the origin (lat, lon) to local
        ``(0, 0)``.  This property returns the **absolute** UTM easting /
        northing of that origin so that callers can shift between the
        map-local frame and absolute UTM 32N (used by ego poses).

        Returns:
            (2,) float64 array ``[easting, northing]``.
        """
        import pyproj  # type: ignore[import-untyped]

        transformer = pyproj.Transformer.from_crs(
            "EPSG:4326", "EPSG:32632", always_xy=True,
        )
        easting, northing = transformer.transform(
            self._origin_lon, self._origin_lat,
        )
        return np.array([easting, northing], dtype=np.float64)

    @cached_property
    def lanelet_map(self) -> "LaneletMap":
        """The full Lanelet2 map loaded from the ``.osm`` file."""
        ll2_map, errors = loadRobust(str(self._map_path), self.projector)
        if errors:
            warnings.warn(
                f"Lanelet2 map loaded with {len(errors)} warnings "
                f"from {self._map_path.name}",
                stacklevel=2,
            )
            for err in errors[:5]:
                logger.debug("Lanelet2 load warning: %s", err)
        logger.info(
            "Loaded Lanelet2 map: %d lanelets, %d areas, %d points",
            len(ll2_map.laneletLayer),
            len(ll2_map.areaLayer),
            len(ll2_map.pointLayer),
        )
        return ll2_map

    @cached_property
    def traffic_rules(self) -> "lanelet2.traffic_rules.TrafficRules":
        """Default traffic rules (Germany, vehicle participant)."""
        return lanelet2.traffic_rules.create(
            lanelet2.traffic_rules.Locations.Germany,
            lanelet2.traffic_rules.Participants.Vehicle,
        )

    @cached_property
    def routing_graph(self) -> "lanelet2.routing.RoutingGraph":
        """Routing graph built from the map and default traffic rules."""
        graph = lanelet2.routing.RoutingGraph(self.lanelet_map, self.traffic_rules)
        logger.info("Built routing graph")
        return graph

    # -- coordinate helpers -------------------------------------------------

    def to_gps(self, point: NDArray[np.float64]) -> tuple[float, float, float]:
        """Convert a local (x, y, z) point to (lat, lon, alt).

        Args:
            point: (3,) array in the map's local UTM frame.

        Returns:
            Tuple of (latitude, longitude, altitude).
        """
        from lanelet2.core import BasicPoint3d  # type: ignore[import-untyped]

        gps = self.projector.reverse(BasicPoint3d(float(point[0]), float(point[1]), float(point[2])))
        return gps.lat, gps.lon, gps.alt

    def from_gps(self, lat: float, lon: float, alt: float = 0.0) -> NDArray[np.float64]:
        """Convert GPS (lat, lon, alt) to the map's local UTM frame.

        Returns:
            (3,) float64 array [x, y, z].
        """
        pt = self.projector.forward(GPSPoint(lat, lon, alt))
        return np.array([pt.x, pt.y, pt.z], dtype=np.float64)

    # -- spatial queries ----------------------------------------------------

    def get_lanelets_in_roi(
        self,
        center: NDArray[np.float64],
        radius: float,
    ) -> list["lanelet2.core.ConstLanelet"]:
        """Return all lanelets whose bounding box intersects a circle.

        Uses Lanelet2's spatial index via :meth:`findWithin2d` for efficiency.

        Args:
            center: (2,) or (3,) query center in local UTM coordinates.
            radius: Search radius in metres.

        Returns:
            List of Lanelet2 ``ConstLanelet`` primitives within *radius* of
            *center*.
        """
        query_pt = BasicPoint2d(float(center[0]), float(center[1]))
        results = lanelet2.geometry.findWithin2d(
            self.lanelet_map.laneletLayer, query_pt, radius
        )
        # findWithin2d returns list of (distance, lanelet) tuples
        return [lanelet for _dist, lanelet in results]

    def get_lane_segments_in_roi(
        self,
        center: NDArray[np.float64],
        radius: float,
    ) -> list[LaneSegment]:
        """Return lane segments near a query point as numpy polylines.

        Convenience wrapper that converts Lanelet2 primitives into
        :class:`LaneSegment` dataclasses with numpy arrays.

        Args:
            center: (2,) or (3,) query center in local UTM coordinates.
            radius: Search radius in metres.

        Returns:
            List of :class:`LaneSegment` instances.
        """
        lanelets = self.get_lanelets_in_roi(center, radius)
        segments: list[LaneSegment] = []
        for llt in lanelets:
            speed_limit = self._get_speed_limit(llt)
            segments.append(
                LaneSegment(
                    lanelet_id=llt.id,
                    left_boundary=_linestring_to_array(llt.leftBound),
                    right_boundary=_linestring_to_array(llt.rightBound),
                    centerline=_linestring_to_array(llt.centerline),
                    speed_limit_mps=speed_limit,
                )
            )
        return segments

    def get_nearest_lanelet(
        self,
        point: NDArray[np.float64],
    ) -> tuple[float, "lanelet2.core.ConstLanelet"]:
        """Find the lanelet closest to a query point.

        Args:
            point: (2,) or (3,) query point.

        Returns:
            Tuple of (distance, lanelet).

        Raises:
            ValueError: If the map contains no lanelets.
        """
        query_pt = BasicPoint2d(float(point[0]), float(point[1]))
        results = lanelet2.geometry.findNearest(
            self.lanelet_map.laneletLayer, query_pt, 1
        )
        if not results:
            raise ValueError("Map contains no lanelets")
        return results[0]  # (distance, lanelet)

    def get_drivable_area(self) -> list[NDArray[np.float64]]:
        """Return drivable-area polygons from passable Lanelet2 areas.

        Uses ``areaLayer`` participant rules (``traffic_rules.canPass(area)``),
        not lanelet footprints or ``drivable_space_border`` linestrings.
        """
        polygons: list[NDArray[np.float64]] = []
        for area in self.lanelet_map.areaLayer:
            if self.traffic_rules.canPass(area):
                poly = area.outerBound()
                pts = np.array([[p.x, p.y] for p in poly], dtype=np.float64)
                if len(pts) > 0:
                    polygons.append(pts)
        return polygons

    def get_drivable_space_borders(self) -> list[NDArray[np.float64]]:
        """Return ``drivable_space_border`` linestrings (open polylines, not polygons)."""
        borders: list[NDArray[np.float64]] = []
        for ls in self.lanelet_map.lineStringLayer:
            if "drivable_space_border" in ls.attributes:
                arr = _linestring_to_array(ls)
                if len(arr) > 0:
                    borders.append(arr)
        return borders

    def get_crosswalks(self) -> list[NDArray[np.float64]]:
        """Return crosswalk polygons as numpy arrays.

        Crosswalks are identified by the Lanelet2 ``subtype`` attribute
        ``"crosswalk"`` on lanelets.

        Returns:
            List of (N, 2) float64 arrays, one per crosswalk polygon.
        """
        crosswalks: list[NDArray[np.float64]] = []
        for llt in self.lanelet_map.laneletLayer:
            subtype = llt.attributes["subtype"] if "subtype" in llt.attributes else ""
            if subtype == "crosswalk":
                poly = llt.polygon2d()
                pts = np.array(
                    [[p.x, p.y] for p in poly], dtype=np.float64
                )
                if len(pts) > 0:
                    crosswalks.append(pts)
        return crosswalks

    def get_stop_lines(self) -> list[NDArray[np.float64]]:
        """Return stop-line polylines as numpy arrays.

        Stop lines are extracted from :class:`RightOfWay` and
        :class:`AllWayStop` regulatory elements.

        Returns:
            List of (N, 2) float64 arrays, one per stop line.
        """
        stop_lines: list[NDArray[np.float64]] = []
        for reg_elem in self.lanelet_map.regulatoryElementLayer:
            attrs = reg_elem.attributes
            # RightOfWay and AllWayStop elements may have a stop_line role
            for role in ("ref_line", "stop_line"):
                params = reg_elem.parameters
                if role in params:
                    for ls in params[role]:
                        arr = _linestring_to_array(ls)
                        if len(arr) > 0:
                            stop_lines.append(arr)
        return stop_lines

    # -- routing convenience ------------------------------------------------

    def get_route(
        self,
        from_lanelet: "lanelet2.core.ConstLanelet",
        to_lanelet: "lanelet2.core.ConstLanelet",
    ) -> Optional["lanelet2.routing.Route"]:
        """Compute a route between two lanelets.

        Args:
            from_lanelet: Start lanelet.
            to_lanelet: Goal lanelet.

        Returns:
            A :class:`Route` object, or ``None`` if no route exists.
        """
        return self.routing_graph.getRoute(from_lanelet, to_lanelet)

    def get_shortest_path(
        self,
        from_lanelet: "lanelet2.core.ConstLanelet",
        to_lanelet: "lanelet2.core.ConstLanelet",
    ) -> Optional["lanelet2.routing.LaneletPath"]:
        """Compute the shortest path between two lanelets.

        Returns:
            A :class:`LaneletPath`, or ``None`` if unreachable.
        """
        return self.routing_graph.shortestPath(from_lanelet, to_lanelet)

    def get_following_lanelets(
        self,
        lanelet: "lanelet2.core.ConstLanelet",
    ) -> list["lanelet2.core.ConstLanelet"]:
        """Return lanelets that directly follow the given one."""
        return list(self.routing_graph.following(lanelet))

    def get_adjacent_lanelets(
        self,
        lanelet: "lanelet2.core.ConstLanelet",
    ) -> tuple[
        Optional["lanelet2.core.ConstLanelet"],
        Optional["lanelet2.core.ConstLanelet"],
    ]:
        """Return (left, right) adjacent lanelets for lane change.

        Returns:
            Tuple of (left_lanelet, right_lanelet). Either may be ``None``.
        """
        left = self.routing_graph.left(lanelet)
        right = self.routing_graph.right(lanelet)
        return left, right

    # -- private helpers ----------------------------------------------------

    def _get_speed_limit(
        self, lanelet: "lanelet2.core.ConstLanelet"
    ) -> Optional[float]:
        """Extract speed limit in m/s from a lanelet's regulatory elements."""
        try:
            speed = self.traffic_rules.speedLimit(lanelet)
            if speed.speedLimit.value() > 0:
                return float(speed.speedLimit.value())
        except Exception:
            pass
        return None


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _linestring_to_array(
    linestring: "lanelet2.core.ConstLineString2d | lanelet2.core.ConstLineString3d",
) -> NDArray[np.float64]:
    """Convert a Lanelet2 linestring to a numpy (N, 2) XY array."""
    pts = [[p.x, p.y] for p in linestring]
    if not pts:
        return np.empty((0, 2), dtype=np.float64)
    return np.array(pts, dtype=np.float64)


def _linestring_to_array_3d(
    linestring: "lanelet2.core.ConstLineString3d",
) -> NDArray[np.float64]:
    """Convert a Lanelet2 3D linestring to a numpy (N, 3) array."""
    pts = [[p.x, p.y, p.z] for p in linestring]
    if not pts:
        return np.empty((0, 3), dtype=np.float64)
    return np.array(pts, dtype=np.float64)


def load_scene_map(
    scene_path: Path,
) -> Optional[SceneMap]:
    """Attempt to load a :class:`SceneMap` from a scene directory.

    Looks for ``maps/map.osm`` and reads the UTM origin from
    ``maps/origin.json``. Returns ``None`` if either file is missing or
    lanelet2 is not installed.

    Args:
        scene_path: Path to the scene directory.

    Returns:
        A :class:`SceneMap` instance, or ``None``.
    """
    if not _LANELET2_AVAILABLE:
        logger.debug("lanelet2 not available — skipping map loading")
        return None

    map_path = scene_path / "maps" / "map.osm"
    origin_path = scene_path / "maps" / "origin.json"

    if not map_path.exists():
        return None
    
    if origin_path.exists():
        import json
        data = json.loads(origin_path.read_text())
        origin_lat, origin_lon = data["latitude"], data["longitude"]
    else:
        logger.warning("Map origin file not found for %s, skipping map.", scene_path.name)
        return None
    
    return SceneMap(map_path, origin_lat=origin_lat, origin_lon=origin_lon)