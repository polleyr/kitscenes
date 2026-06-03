from __future__ import annotations

import functools
import logging
import os
from pathlib import Path
from typing import Iterator

from kitscenes.constants import (
    ALL_SENSOR_NAMES,
    KITSCENES_ROOT_ENV_VAR
)
from kitscenes.map_api import load_scene_map
from kitscenes.sensors import SensorDataLoader
from kitscenes.schema import Scene, SceneMetadata, EgoPose
from kitscenes.poses import load_ego_poses

_SPLIT_DIR = (
    Path(__file__).parent / "split" / "generated_splits" / "default_geo_split_v1_0"
)

_SPLIT_FILES: dict[str, str] = {
    "train": "train.txt",
    "val": "validation.txt",
    "test": "test.txt",
    "test_e2e": "test-e2e.txt",
    "overlap_train_val": "overlap_train_val.txt",
}

_KNOWN_SPLITS: frozenset[str] = frozenset(_SPLIT_FILES)


_DATA_SUBDIR = "data"

logger = logging.getLogger(__name__)


def _load_scene_ego_poses(
    scene_path: Path,
    num_reference_frames: int,
    scene_id: str,
) -> tuple[EgoPose, ...]:
    poses = tuple(load_ego_poses(scene_path))
    if poses and num_reference_frames != len(poses):
        logger.warning(
            "Scene %s: %d reference timestamps but %d ego poses in poses.txt.",
            scene_id,
            num_reference_frames,
            len(poses),
        )
    return poses


def _scenes_parent(root: Path) -> Path:
    """Directory holding split subfolders (mirrors HF: ``root/data/<split>/``)."""
    data_dir = root / _DATA_SUBDIR
    if data_dir.is_dir() and any(
        (data_dir / name).is_dir() for name in _KNOWN_SPLITS
    ):
        return data_dir
    return root


def _load_scene_ids(root: Path, split: str | None) -> tuple[Path, list[str]]:
    """Return (scene_root, sorted scene IDs).

    *scene_root* is the directory that scene IDs are relative to; it may
    equal *root*, ``root/data``, or ``root/data/<split>``.
    """
    parent = _scenes_parent(root)

    if split is not None:
        if split not in _SPLIT_FILES:
            raise ValueError(
                f"Unknown split {split!r}. Valid splits: {list(_SPLIT_FILES.keys())}"
            )
        scene_root = parent / split
        split_file = _SPLIT_DIR / _SPLIT_FILES[split]
        ids = [s for s in split_file.read_text().splitlines() if s.strip()]
        return scene_root, [sid for sid in ids if (scene_root / sid).is_dir()]

    subdirs = [d for d in parent.iterdir() if d.is_dir()]
    split_subdirs = [d for d in subdirs if d.name in _KNOWN_SPLITS]

    if split_subdirs:
        all_ids: list[str] = []
        for split_dir in sorted(split_subdirs):
            all_ids.extend(d.name for d in sorted(split_dir.iterdir()) if d.is_dir())
        return root, all_ids

    return parent, sorted(d.name for d in subdirs)


class KITScenesDataset:
    """Main entry point for the KITScenes dataset.

    Args:
        root: Path to the dataset root (HuggingFace layout: ``data/<split>/`` under
            this path). Legacy flat ``<split>/`` at the root is still supported.
            Falls back to ``$KITSCENES_ROOT`` when omitted.
        split: Restrict to one split (``'train'``, ``'val'``, ``'test'``,
            ``'test_e2e'``, ``'overlap_train_val'``).  When *None* the dataset
            auto-discovers all scenes under *root*.
    """

    def __init__(
            self,
            root: Path | str | None = None,
            split: str | None = None
    ) -> None:
        if root is None:
            env_root = os.environ.get(KITSCENES_ROOT_ENV_VAR)
            if env_root is None:
                raise ValueError(
                    f"No root provided and ${KITSCENES_ROOT_ENV_VAR} is not set. "
                    f"Pass root= explicitly or set the environment variable."
                )
            root = env_root

        self._root = Path(root)
        if not self._root.is_dir():
            raise FileNotFoundError(f"Dataset root not found: {self._root}")

        self._split = split
        self._scene_root, self._scene_ids = _load_scene_ids(self._root, split)

    def _scene_path(self, scene_id: str) -> Path:
        """Resolve the absolute path for *scene_id*.

        When scenes were aggregated from multiple split subfolders the stored
        ``_scene_root`` equals ``root``, so we need to locate the UUID under
        one of the split subdirectories.
        """
        direct = self._scene_root / scene_id
        if direct.is_dir():
            return direct
        parent = _scenes_parent(self._root)
        for child in sorted(parent.iterdir()):
            if not child.is_dir():
                continue
            candidate = child / scene_id
            if candidate.is_dir():
                return candidate
        raise FileNotFoundError(f"Scene directory not found for ID: {scene_id!r}")

    @property
    def root(self) -> Path:
        return self._root

    @property
    def split(self) -> str | None:
        return self._split

    @property
    def scene_ids(self) -> list[str]:
        return list(self._scene_ids)

    @functools.lru_cache(maxsize=None)
    def get_sensor_loader(self, scene_id: str) -> SensorDataLoader:
        if scene_id not in self._scene_ids:
            raise KeyError(f"Scene ID not found: {scene_id!r}")
        return SensorDataLoader(self._scene_path(scene_id))

    @functools.lru_cache(maxsize=None)
    def get_scene(self, scene_id: str) -> Scene:
        if scene_id not in self._scene_ids:
            raise KeyError(f"Scene ID not found: {scene_id!r}")

        scene_path = self._scene_path(scene_id)
        timestamp_ns = self.get_sensor_loader(scene_id).get_reference_timestamps()
        sensor_names = tuple(n for n in ALL_SENSOR_NAMES if (scene_path / n).is_dir())
        num_frames = len(timestamp_ns)

        return Scene(
            scene_id=scene_id,
            scene_path=scene_path,
            timestamps_ns=timestamp_ns,
            metadata=SceneMetadata(
                num_frames=num_frames,
                duration_s=(timestamp_ns[-1] - timestamp_ns[0]) / 1e9 if num_frames > 1 else 0.0,
                sensor_names=sensor_names,
            ),
            _ego_loader=lambda sp=scene_path, nf=num_frames, sid=scene_id: _load_scene_ego_poses(
                sp, nf, sid
            ),
            _map_loader=lambda: load_scene_map(scene_path),
        )

    def as_frames(self) -> "FrameDataset":
        from kitscenes.frame_dataset import FrameDataset
        return FrameDataset(self)
    
    def __len__(self) -> int:
        return len(self._scene_ids)
    
    def __iter__(self) -> Iterator[Scene]:
        for scene_id in self._scene_ids:
            yield self.get_scene(scene_id)
    
    def __getitem__(self, key: int | str) -> Scene:
        if isinstance(key, str):
            return self.get_scene(key)
        return self.get_scene(self._scene_ids[key])
    
    def __repr__(self) -> str:
        return (
            f"KITScenesDataset(root={self._root!r}, split={self._split!r}, num_scenes={len(self)})"
        )
