"""Tests for kitscenes.map_api — SceneMap and helpers.

Tests are split into two groups:

1. **Unit tests** (always run): Use ``unittest.mock`` to patch lanelet2 so that
   the test suite passes even on machines where lanelet2 is not installed
   (e.g. macOS / ARM where only x86 wheels exist).

2. **Integration tests** (skip if lanelet2 unavailable): Exercise real Lanelet2
   loading against a small synthetic ``.osm`` file.  Marked with
   ``@pytest.mark.skipif(not HAS_LANELET2, ...)``.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np
import pytest

from kitscenes.map_api import (
    LaneSegment,
    SceneMap,
    _linestring_to_array,
    _linestring_to_array_3d,
    load_scene_map,
)

# ---------------------------------------------------------------------------
# Detect whether lanelet2 is actually importable
# ---------------------------------------------------------------------------

try:
    import lanelet2  # type: ignore[import-untyped]

    HAS_LANELET2 = True
except ImportError:
    HAS_LANELET2 = False

needs_lanelet2 = pytest.mark.skipif(
    not HAS_LANELET2, reason="lanelet2 not installed"
)


# ===================================================================
# Section 1 — Pure-Python unit tests (no lanelet2 required)
# ===================================================================


class TestLaneSegment:
    """Test the LaneSegment frozen dataclass."""

    def test_construction(self) -> None:
        left = np.array([[0, 0], [10, 0]], dtype=np.float64)
        right = np.array([[0, 3], [10, 3]], dtype=np.float64)
        center = np.array([[0, 1.5], [10, 1.5]], dtype=np.float64)
        seg = LaneSegment(
            lanelet_id=42,
            left_boundary=left,
            right_boundary=right,
            centerline=center,
            speed_limit_mps=13.89,
        )
        assert seg.lanelet_id == 42
        assert seg.speed_limit_mps == pytest.approx(13.89)
        np.testing.assert_array_equal(seg.left_boundary, left)
        np.testing.assert_array_equal(seg.right_boundary, right)
        np.testing.assert_array_equal(seg.centerline, center)

    def test_frozen(self) -> None:
        seg = LaneSegment(
            lanelet_id=1,
            left_boundary=np.zeros((2, 2)),
            right_boundary=np.zeros((2, 2)),
            centerline=np.zeros((2, 2)),
            speed_limit_mps=None,
        )
        with pytest.raises(AttributeError):
            seg.lanelet_id = 99  # type: ignore[misc]


class TestLinestringToArray:
    """Test the module-level linestring→numpy conversion helpers."""

    def test_2d_conversion(self) -> None:
        # Simulate a lanelet2 linestring as a list of point-like objects
        FakePoint = SimpleNamespace
        ls = [FakePoint(x=1.0, y=2.0), FakePoint(x=3.0, y=4.0)]
        arr = _linestring_to_array(ls)  # type: ignore[arg-type]
        assert arr.shape == (2, 2)
        assert arr.dtype == np.float64
        np.testing.assert_array_equal(arr, [[1, 2], [3, 4]])

    def test_3d_conversion(self) -> None:
        FakePoint = SimpleNamespace
        ls = [FakePoint(x=1.0, y=2.0, z=3.0), FakePoint(x=4.0, y=5.0, z=6.0)]
        arr = _linestring_to_array_3d(ls)  # type: ignore[arg-type]
        assert arr.shape == (2, 3)
        np.testing.assert_array_equal(arr, [[1, 2, 3], [4, 5, 6]])

    def test_empty_linestring(self) -> None:
        arr = _linestring_to_array([])  # type: ignore[arg-type]
        assert arr.shape == (0, 2)

        arr3 = _linestring_to_array_3d([])  # type: ignore[arg-type]
        assert arr3.shape == (0, 3)



class TestLoadSceneMap:
    """Test the ``load_scene_map`` convenience function."""

    def test_returns_none_when_no_lanelet2(self, tmp_path: Path) -> None:
        (tmp_path / "map.osm").write_text("<osm/>")
        with mock.patch("kitscenes.map_api._LANELET2_AVAILABLE", False):
            result = load_scene_map(tmp_path)
        assert result is None

    def test_returns_none_when_no_file(self, tmp_path: Path) -> None:
        if not HAS_LANELET2:
            with mock.patch("kitscenes.map_api._LANELET2_AVAILABLE", True):
                result = load_scene_map(tmp_path)
        else:
            result = load_scene_map(tmp_path)
        assert result is None



# ===================================================================
# Section 2 — Integration tests (require lanelet2)
# ===================================================================

# Minimal valid Lanelet2 OSM content for testing
_MINIMAL_OSM = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <osm version="0.6">
      <!-- Two points forming a linestring -->
      <node id="1" lat="49.01440" lon="8.41720" />
      <node id="2" lat="49.01450" lon="8.41720" />
      <node id="3" lat="49.01440" lon="8.41725" />
      <node id="4" lat="49.01450" lon="8.41725" />

      <!-- Left boundary -->
      <way id="10">
        <nd ref="1" />
        <nd ref="2" />
        <tag k="type" v="line_thin" />
        <tag k="subtype" v="solid" />
      </way>

      <!-- Right boundary -->
      <way id="11">
        <nd ref="3" />
        <nd ref="4" />
        <tag k="type" v="line_thin" />
        <tag k="subtype" v="solid" />
      </way>

      <!-- A lanelet -->
      <relation id="100">
        <member type="way" ref="10" role="left" />
        <member type="way" ref="11" role="right" />
        <tag k="type" v="lanelet" />
        <tag k="subtype" v="road" />
        <tag k="location" v="urban" />
      </relation>
    </osm>
""")

# OSM with a crosswalk lanelet
_CROSSWALK_OSM = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <osm version="0.6">
      <node id="1" lat="49.01440" lon="8.41720" />
      <node id="2" lat="49.01450" lon="8.41720" />
      <node id="3" lat="49.01440" lon="8.41725" />
      <node id="4" lat="49.01450" lon="8.41725" />

      <way id="10">
        <nd ref="1" />
        <nd ref="2" />
      </way>

      <way id="11">
        <nd ref="3" />
        <nd ref="4" />
      </way>

      <relation id="100">
        <member type="way" ref="10" role="left" />
        <member type="way" ref="11" role="right" />
        <tag k="type" v="lanelet" />
        <tag k="subtype" v="crosswalk" />
        <tag k="location" v="urban" />
      </relation>
    </osm>
""")


@needs_lanelet2
class TestSceneMapIntegration:
    """Integration tests that load real Lanelet2 maps."""

    @pytest.fixture()
    def osm_path(self, tmp_path: Path) -> Path:
        p = tmp_path / "map.osm"
        p.write_text(_MINIMAL_OSM)
        return p

    @pytest.fixture()
    def crosswalk_osm_path(self, tmp_path: Path) -> Path:
        p = tmp_path / "crosswalk_map.osm"
        p.write_text(_CROSSWALK_OSM)
        return p

    @pytest.fixture()
    def scene_map(self, osm_path: Path) -> SceneMap:
        return SceneMap(osm_path, origin_lat=49.01439, origin_lon=8.41722)

    def test_load_map(self, scene_map: SceneMap) -> None:
        ll2_map = scene_map.lanelet_map
        assert len(ll2_map.laneletLayer) >= 1

    def test_repr(self, scene_map: SceneMap) -> None:
        r = repr(scene_map)
        assert "SceneMap" in r
        assert "map=" in r

    def test_routing_graph(self, scene_map: SceneMap) -> None:
        graph = scene_map.routing_graph
        assert graph is not None

    def test_traffic_rules(self, scene_map: SceneMap) -> None:
        rules = scene_map.traffic_rules
        assert rules is not None

    def test_get_lanelets_in_roi(self, scene_map: SceneMap) -> None:
        # Query near the map origin — the single lanelet should be found
        results = scene_map.get_lanelets_in_roi(
            center=np.array([0.0, 0.0]), radius=500.0
        )
        assert len(results) >= 1

    def test_get_lane_segments_in_roi(self, scene_map: SceneMap) -> None:
        segments = scene_map.get_lane_segments_in_roi(
            center=np.array([0.0, 0.0]), radius=500.0
        )
        assert len(segments) >= 1
        seg = segments[0]
        assert isinstance(seg, LaneSegment)
        assert seg.left_boundary.ndim == 2
        assert seg.right_boundary.ndim == 2
        assert seg.centerline.ndim == 2

    def test_get_nearest_lanelet(self, scene_map: SceneMap) -> None:
        dist, llt = scene_map.get_nearest_lanelet(np.array([0.0, 0.0]))
        assert dist >= 0
        assert llt.id > 0

    def test_get_drivable_area(self, scene_map: SceneMap) -> None:
        polygons = scene_map.get_drivable_area()
        assert isinstance(polygons, list)
        for poly in polygons:
            assert poly.ndim == 2
            assert poly.shape[1] == 2

    def test_get_crosswalks_empty(self, scene_map: SceneMap) -> None:
        # The minimal OSM has subtype="road", not crosswalk
        crosswalks = scene_map.get_crosswalks()
        assert len(crosswalks) == 0

    def test_get_crosswalks_found(self, crosswalk_osm_path: Path) -> None:
        smap = SceneMap(crosswalk_osm_path, origin_lat=49.01439, origin_lon=8.41722)
        crosswalks = smap.get_crosswalks()
        assert len(crosswalks) >= 1

    def test_to_gps_roundtrip(self, scene_map: SceneMap) -> None:
        # Convert origin GPS to local, should be near (0, 0, 0)
        local = scene_map.from_gps(49.01439, 8.41722, 0.0)
        assert abs(local[0]) < 1.0
        assert abs(local[1]) < 1.0

        # Round-trip
        lat, lon, alt = scene_map.to_gps(local)
        assert abs(lat - 49.01439) < 1e-4
        assert abs(lon - 8.41722) < 1e-4

    def test_projector_cached(self, scene_map: SceneMap) -> None:
        p1 = scene_map.projector
        p2 = scene_map.projector
        assert p1 is p2

    def test_lanelet_map_cached(self, scene_map: SceneMap) -> None:
        m1 = scene_map.lanelet_map
        m2 = scene_map.lanelet_map
        assert m1 is m2


