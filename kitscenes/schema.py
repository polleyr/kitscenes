"""Data schema / dataclass definitions for the kitscenes API."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

if TYPE_CHECKING:
    from kitscenes.map_api import SceneMap


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class EgoPose:
    """Ego-vehicle pose at a single timestamp, loaded from ``poses.txt``.

    Attributes:
        timestamp_ns: Timestamp in nanoseconds.
        translation: (3,) x, y, z position (UTM frame).
        rotation: (4,) quaternion (qx, qy, qz, qw).
    """

    timestamp_ns: int
    translation: np.ndarray  # (3,) float64
    rotation: np.ndarray  # (4,) float64, quaternion


@dataclasses.dataclass(frozen=True)
class SceneMetadata:
    """Aggregate metadata for a scene.

    Attributes:
        num_frames: Total number of reference-timeline frames.
        duration_s: Scene duration in seconds.
        sensor_names: Names of sensors present in the scene directory.
    """

    num_frames: int
    duration_s: float
    sensor_names: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class Scene:
    """A single recorded driving scene.

    Attributes:
        scene_id: Unique directory name of the scene.
        scene_path: Absolute path to the scene directory on disk.
        timestamps_ns: (T,) reference-timeline timestamps in nanoseconds.
        ego_poses: Ego-vehicle poses from ``poses.txt``, one per reference
            timestamp. Loaded lazily on first access.
        map: HD map for this scene, or ``None`` if unavailable. Loaded lazily
            on first access.
        metadata: Scene-level aggregate information.
    """

    scene_id: str
    scene_path: Path
    timestamps_ns: np.ndarray  # (T,) int64
    metadata: SceneMetadata
    _ego_loader: dataclasses.InitVar[Optional[Callable[[], tuple[EgoPose, ...]]]] = None
    _map_loader: dataclasses.InitVar[Optional[Callable[[], Optional[SceneMap]]]] = None

    def __post_init__(
        self,
        _ego_loader: Optional[Callable[[], tuple[EgoPose, ...]]],
        _map_loader: Optional[Callable[[], Optional[SceneMap]]],
    ) -> None:
        object.__setattr__(self, "_ego_loader", _ego_loader if _ego_loader is not None else lambda: ())
        object.__setattr__(self, "_map_loader", _map_loader if _map_loader is not None else lambda: None)

    @property
    def ego_poses(self) -> tuple[EgoPose, ...]:
        """Ego poses loaded from ``poses.txt``, lazily on first access."""
        try:
            return self.__dict__["_ego_poses"]  # type: ignore[return-value]
        except KeyError:
            poses = self._ego_loader()  # type: ignore[attr-defined]
            self.__dict__["_ego_poses"] = poses
            return poses

    @property
    def map(self) -> Optional[SceneMap]:
        """HD map loaded from ``maps/map.osm``, lazily on first access."""
        try:
            return self.__dict__["_map"]  # type: ignore[return-value]
        except KeyError:
            scene_map = self._map_loader()  # type: ignore[attr-defined]
            self.__dict__["_map"] = scene_map
            return scene_map

    @property
    def timestamp_s(self) -> np.ndarray:
        """Reference timestamps converted from nanoseconds to seconds."""
        return self.timestamps_ns.astype(np.float64) / 1e9
