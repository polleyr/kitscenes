"""Tests for kitscenes.visualization — scene_viz rendering functions.

All tests use synthetic data and run with matplotlib's non-interactive Agg
backend, so no display server is needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be set before pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pytest

from kitscenes.schema import (
    EgoPose,
    Scene,
    SceneMetadata,
)
from kitscenes.sensors import SensorDataLoader
from kitscenes.visualization.scene_viz import (
    _draw_heading_arrow,
    _SURROUND_LAYOUT,
    render_scene_animation,
    render_scene_bev,
    render_surround_view,
)

# ---------------------------------------------------------------------------
# Helpers — build synthetic scene data
# ---------------------------------------------------------------------------

_START_NS = 1700000000_000000000
_STEP_NS = 100_000_000  # 100 ms


def _make_gnss_ego_poses(n: int = 20) -> tuple[EgoPose, ...]:
    """Create *n* ego poses moving along the x-axis."""
    poses = []
    for i in range(n):
        poses.append(EgoPose(
            timestamp_ns=_START_NS + i * _STEP_NS,
            translation=np.array([float(i), 0.0, 0.0], dtype=np.float64),
            rotation=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),  # identity quat
        ))
    return tuple(poses)


def _make_scene(n_frames: int = 20) -> Scene:
    """Build a minimal synthetic scene."""
    ego = _make_gnss_ego_poses(n_frames)
    timestamps = np.array(
        [_START_NS + i * _STEP_NS for i in range(n_frames)], dtype=np.int64
    )
    return Scene(
        scene_id="synthetic_001",
        scene_path=Path("/tmp/fake_scene"),
        timestamps_ns=timestamps,
        _ego_loader=lambda: ego,
        _map_loader=lambda: None,
        metadata=SceneMetadata(
            num_frames=n_frames,
            duration_s=n_frames * 0.1,
            sensor_names=("lidar_top",),
        ),
    )


def _write_timestamp_file(path: Path, count: int) -> None:
    lines = [str(_START_NS + i * _STEP_NS) for i in range(count)]
    path.write_text("\n".join(lines) + "\n")


def _make_synthetic_scene_dir(tmp_path: Path, n_frames: int = 5) -> Path:
    """Create a scene directory with ring camera images for surround-view tests."""
    scene_dir = tmp_path / "scene_001"
    scene_dir.mkdir()

    # Reference timestamps
    _write_timestamp_file(scene_dir / "timestamp.reference.txt", n_frames)

    # Calibration with one camera
    calib = {
        "camera_ring_front_pinhole": {
            "T_to_reference": np.eye(4).tolist(),
            "intrinsics": {
                "focal_length": 1000.0,
                "principal_point_u": 50.0,
                "principal_point_v": 40.0,
            },
        },
    }
    (scene_dir / "calibration").mkdir()
    (scene_dir / "calibration" / "calib.json").write_text(json.dumps(calib))

    # Create small JPEG images for ring cameras
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("Pillow not installed — cannot create test images")

    for cam_name in ("camera_ring_front",):
        cam_dir = scene_dir / cam_name
        cam_dir.mkdir()
        _write_timestamp_file(
            scene_dir / f"timestamp.{cam_name}.txt", n_frames
        )
        for i in range(n_frames):
            img = Image.fromarray(
                np.random.randint(0, 255, (80, 100, 3), dtype=np.uint8)
            )
            img.save(cam_dir / f"{i:010d}.jpg")

    return scene_dir


# ===================================================================
# Tests
# ===================================================================


class TestRenderSceneBev:
    """Test render_scene_bev with synthetic data."""

    def test_full_trajectory(self) -> None:
        scene = _make_scene()
        fig, ax = render_scene_bev(scene, timestep=None)
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_single_timestep(self) -> None:
        scene = _make_scene()
        fig, ax = render_scene_bev(scene, timestep=5)
        assert fig is not None
        plt.close(fig)

    def test_timestep_out_of_range(self) -> None:
        scene = _make_scene(n_frames=10)
        with pytest.raises(IndexError, match="out of range"):
            render_scene_bev(scene, timestep=100)

    def test_save_to_file(self, tmp_path: Path) -> None:
        scene = _make_scene()
        out = tmp_path / "bev.png"
        fig, ax = render_scene_bev(scene, timestep=0, out_path=out)
        assert out.exists()
        assert out.stat().st_size > 0
        plt.close(fig)

    def test_existing_axes(self) -> None:
        scene = _make_scene()
        fig, ax = plt.subplots()
        fig2, ax2 = render_scene_bev(scene, ax=ax)
        assert ax2 is ax
        plt.close(fig)

    def test_no_ego(self) -> None:
        scene = _make_scene()
        fig, ax = render_scene_bev(scene, show_ego=False)
        assert fig is not None
        plt.close(fig)

    def test_empty_scene(self) -> None:
        """A scene with no ego poses should not crash."""
        scene = Scene(
            scene_id="empty",
            scene_path=Path("/tmp/empty"),
            timestamps_ns=np.array([], dtype=np.int64),
            _map_loader=lambda: None,
            metadata=SceneMetadata(num_frames=0, duration_s=0.0, sensor_names=()),
        )
        fig, ax = render_scene_bev(scene)
        assert fig is not None
        plt.close(fig)


class TestRenderSceneAnimation:
    """Test render_scene_animation."""

    def test_animation_gif(self, tmp_path: Path) -> None:
        """Create a short GIF animation (pillow writer, no ffmpeg needed)."""
        scene = _make_scene(n_frames=5)
        out = tmp_path / "anim.gif"
        render_scene_animation(scene, out_path=out, fps=2)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_empty_scene_no_crash(self, tmp_path: Path) -> None:
        scene = Scene(
            scene_id="empty",
            scene_path=Path("/tmp/empty"),
            timestamps_ns=np.array([], dtype=np.int64),
            _map_loader=lambda: None,
            metadata=SceneMetadata(num_frames=0, duration_s=0.0, sensor_names=()),
        )
        out = tmp_path / "anim.gif"
        render_scene_animation(scene, out_path=out)
        # Should just log a warning, not crash
        assert not out.exists()


class TestRenderSurroundView:
    """Test render_surround_view with synthetic scene data."""

    def test_basic(self, tmp_path: Path) -> None:
        scene_dir = _make_synthetic_scene_dir(tmp_path, n_frames=3)
        loader = SensorDataLoader(scene_dir)
        fig, axes = render_surround_view(loader, frame_idx=0)
        assert fig is not None
        assert axes.shape[0] == 2  # 2 rows
        assert axes.shape[1] == 3  # 3 cols
        plt.close(fig)

    def test_missing_cameras_no_crash(self, tmp_path: Path) -> None:
        """Cameras not present on disk show 'N/A' instead of crashing."""
        scene_dir = _make_synthetic_scene_dir(tmp_path, n_frames=2)
        loader = SensorDataLoader(scene_dir)
        # Default layout has 6 ring cameras, only 1 exists
        fig, axes = render_surround_view(loader, frame_idx=0)
        assert fig is not None
        plt.close(fig)

    def test_save_to_file(self, tmp_path: Path) -> None:
        scene_dir = _make_synthetic_scene_dir(tmp_path, n_frames=2)
        loader = SensorDataLoader(scene_dir)
        out = tmp_path / "surround.png"
        fig, axes = render_surround_view(
            loader, frame_idx=0, out_path=out
        )
        assert out.exists()
        plt.close(fig)

    def test_custom_layout(self, tmp_path: Path) -> None:
        scene_dir = _make_synthetic_scene_dir(tmp_path, n_frames=2)
        loader = SensorDataLoader(scene_dir)
        layout = [["camera_ring_front", "", ""]]
        fig, axes = render_surround_view(
            loader, frame_idx=0, camera_names=layout
        )
        assert axes.shape == (1, 3)
        plt.close(fig)


class TestDrawHelpers:
    """Test internal drawing helper functions."""

    def test_draw_heading_arrow(self) -> None:
        fig, ax = plt.subplots()
        pose = EgoPose(
            timestamp_ns=0,
            translation=np.array([0.0, 0.0, 0.0]),
            rotation=np.array([0.0, 0.0, 0.0, 1.0]),  # facing +x
        )
        _draw_heading_arrow(ax, pose, color="red")
        plt.close(fig)


class TestPlotMapInteractive:
    """Test plot_map_interactive (requires plotly)."""

    def test_import_error_without_plotly(self) -> None:
        """Should raise ImportError with a useful message when plotly absent."""
        import unittest.mock as mock

        # Even if plotly is installed, patch the import to fail
        with mock.patch.dict("sys.modules", {"plotly": None, "plotly.graph_objects": None}):
            from kitscenes.visualization import scene_viz
            # Force re-import attempt
            import importlib
            importlib.reload(scene_viz)
            try:
                from kitscenes.visualization.scene_viz import plot_map_interactive

                # We need a SceneMap mock to call it
                fake_map = mock.Mock()
                with pytest.raises((ImportError, ModuleNotFoundError)):
                    plot_map_interactive(fake_map, center=np.array([0.0, 0.0]))
            except ImportError:
                pass  # Expected if plotly is not installed at all
            finally:
                importlib.reload(scene_viz)
