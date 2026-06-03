"""Tests for HD map visualization modules (video generation, imports)."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from kitscenes.constants import KITSCENES_ROOT_ENV_VAR, KITSCENES_VIZ_OUTPUT_ENV_VAR
from kitscenes.visualization.output_paths import (
    guard_output_path,
    resolve_viz_output_dir,
    resolve_viz_output_file,
)
from kitscenes.visualization.video_generation import (
    create_composite_frame,
    generate_map_projection_video,
    get_frame_indices,
)


def _write_test_projection_dir(tmp_path: Path, n_frames: int = 3) -> Path:
    visdir = tmp_path / "projections"
    grid_dir = visdir / "grid"
    top_down_dir = visdir / "top_down"
    grid_dir.mkdir(parents=True)
    top_down_dir.mkdir(parents=True)

    for i in range(n_frames):
        grid = np.full((120, 180, 3), i * 40, dtype=np.uint8)
        top = np.full((200, 100, 3), 255 - i * 30, dtype=np.uint8)
        cv2.imwrite(str(grid_dir / f"{i:010d}.jpg"), grid)
        cv2.imwrite(str(top_down_dir / f"{i:010d}.jpg"), top)
    return visdir


class TestOutputPaths:
    def test_resolve_from_explicit_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(KITSCENES_VIZ_OUTPUT_ENV_VAR, raising=False)
        out = tmp_path / "viz_out"
        assert resolve_viz_output_dir(out) == out.resolve()

    def test_resolve_from_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        env_out = tmp_path / "from_env"
        monkeypatch.setenv(KITSCENES_VIZ_OUTPUT_ENV_VAR, str(env_out))
        assert resolve_viz_output_dir(None) == env_out.resolve()

    def test_missing_output_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(KITSCENES_VIZ_OUTPUT_ENV_VAR, raising=False)
        with pytest.raises(ValueError, match="KITSCENES_VIZ_OUTPUT"):
            resolve_viz_output_dir(None)

    def test_rejects_output_inside_dataset_root(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dataset_root = tmp_path / "dataset"
        dataset_root.mkdir()
        inside = dataset_root / "scene" / "map_projections"
        monkeypatch.setenv(KITSCENES_ROOT_ENV_VAR, str(dataset_root))
        with pytest.raises(ValueError, match="Refusing to write"):
            guard_output_path(inside, kind="directory")

    def test_resolve_video_file_from_env_dir(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(KITSCENES_VIZ_OUTPUT_ENV_VAR, str(tmp_path / "renders"))
        path = resolve_viz_output_file(None, default_name="demo.mp4")
        assert path == (tmp_path / "renders" / "demo.mp4").resolve()


    def test_resolve_origin_from_scene_maps(self, tmp_path: Path) -> None:
        from kitscenes.visualization.map_viz import resolve_map_projection_origin

        scene = tmp_path / "scene"
        (scene / "maps").mkdir(parents=True)
        (scene / "maps" / "origin.json").write_text(
            '{"latitude": 49.01, "longitude": 8.42}\n',
            encoding="utf-8",
        )
        lat, lon = resolve_map_projection_origin(scene)
        assert lat == pytest.approx(49.01)
        assert lon == pytest.approx(8.42)


    def test_default_te_icon_dir(self) -> None:
        from kitscenes.visualization.ml_converter_vis_utils import default_te_icon_dir

        icon_dir = Path(default_te_icon_dir())
        assert icon_dir.name == "icon_images"
        assert icon_dir.parent.name == "res"
        assert icon_dir.is_dir()


class TestVideoGeneration:
    def test_get_frame_indices(self, tmp_path: Path) -> None:
        visdir = _write_test_projection_dir(tmp_path, n_frames=4)
        indices = get_frame_indices(visdir)
        assert indices == [0, 1, 2, 3]

    def test_create_composite_frame(self, tmp_path: Path) -> None:
        visdir = _write_test_projection_dir(tmp_path, n_frames=1)
        composite = create_composite_frame(visdir, 0)
        assert composite is not None
        assert composite.shape[0] == 120
        assert composite.shape[1] > 180

    def test_generate_map_projection_video(self, tmp_path: Path) -> None:
        visdir = _write_test_projection_dir(tmp_path, n_frames=3)
        out = tmp_path / "out.mp4"
        written = generate_map_projection_video(
            visdir,
            out,
            fps=2,
            max_workers=1,
            batch_size=2,
            verbose=False,
        )
        assert written == out
        assert out.is_file()
        assert out.stat().st_size > 0


class TestMapVizImports:
    def test_core_visualization_import_without_lanelet2(self) -> None:
        import kitscenes.visualization as viz

        assert callable(viz.render_lidar_bev)
        assert callable(viz.render_scene_bev)

    def test_cli_help(self) -> None:
        from kitscenes.visualization.__main__ import main

        with pytest.raises(SystemExit) as exc:
            main(["--help"])
        assert exc.value.code == 0
