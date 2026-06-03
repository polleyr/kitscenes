from __future__ import annotations

from typing import Optional

import numpy as np

from kitscenes.schema import EgoPose
from kitscenes.sensors import LidarSweep, RadarSweep, SensorDataLoader


class Frame:
    """A single reference-timeline frame with access to all sensor data.
    
    Args:
        scene_id: UUID of the parent scene.
        frame_idx: Index into the scene's reference timestamp.
        timestamp_ns: Timestamp of the frame in nanoseconds.
        loader: Sensor data loader for the parent scene
        ego_poses: Ego poses for the parent scene, indexed by frame_idx.
    """

    def __init__(
            self,
            scene_id: str,
            frame_idx: int,
            timestamp_ns: int,
            loader: SensorDataLoader,
            ego_poses: tuple[EgoPose, ...] = (),
    ) -> None:
        self.scene_id = scene_id
        self.frame_idx = frame_idx
        self.timestamp_ns = timestamp_ns
        self._loader = loader
        self._ego_poses = ego_poses

    @property
    def timestamp_s(self) -> float:
        return self.timestamp_ns / 1e9
    
    @property
    def ego_pose(self) -> Optional[EgoPose]:
        if not self._ego_poses or self.frame_idx >= len(self._ego_poses):
            return None
        return self._ego_poses[self.frame_idx]

    def lidar(self, name: str) -> Optional[LidarSweep]:
        try:
            return self._loader.get_lidar_sweep(name, self.frame_idx)
        except FileNotFoundError:
            return None

    def camera(self, name: str) -> Optional[np.ndarray]:
        try:
            return self._loader.get_camera_image(name, self.frame_idx)
        except FileNotFoundError:
            return None

    def radar(self, name: str) -> Optional[RadarSweep]:
        try:
            return self._loader.get_radar_sweep(name, self.frame_idx)
        except FileNotFoundError:
            return None
    
    def available_lidar_names(self) -> list[str]:
        return self._loader.get_lidar_names()
    
    def available_camera_names(self) -> list[str]:
        return self._loader.get_camera_names()
    
    def available_radar_names(self) -> list[str]:
        return self._loader.get_radar_names()
    
    @property
    def lidar_top(self) -> Optional[LidarSweep]:
        return self.lidar("lidar_top")
    
    @property
    def lidar_front(self) -> Optional[LidarSweep]:
        return self.lidar("lidar_front")
    
    @property
    def lidar_left(self) -> Optional[LidarSweep]:
        return self.lidar("lidar_left")
    
    @property
    def lidar_right(self) -> Optional[LidarSweep]:
        return self.lidar("lidar_right")
    
    @property
    def lidar_rear(self) -> Optional[LidarSweep]:
        return self.lidar("lidar_rear")

    @property
    def lidar_corner_left(self) -> Optional[LidarSweep]:
        return self.lidar("lidar_corner_left")
    
    @property
    def lidar_corner_right(self) -> Optional[LidarSweep]:
        return self.lidar("lidar_corner_right")
    
    @property
    def camera_base_front_center(self) -> Optional[np.ndarray]:
        return self.camera("camera_base_front_center")

    @property
    def camera_base_front_left_rect(self) -> Optional[np.ndarray]:
        return self.camera("camera_base_front_left_rect")

    @property
    def camera_base_front_right_rect(self) -> Optional[np.ndarray]:
        return self.camera("camera_base_front_right_rect")

    @property
    def camera_ring_front(self) -> Optional[np.ndarray]:
        return self.camera("camera_ring_front")

    @property
    def camera_ring_front_left(self) -> Optional[np.ndarray]:
        return self.camera("camera_ring_front_left")

    @property
    def camera_ring_front_right(self) -> Optional[np.ndarray]:
        return self.camera("camera_ring_front_right")

    @property
    def camera_ring_rear(self) -> Optional[np.ndarray]:
        return self.camera("camera_ring_rear")

    @property
    def camera_ring_rear_left(self) -> Optional[np.ndarray]:
        return self.camera("camera_ring_rear_left")

    @property
    def camera_ring_rear_right(self) -> Optional[np.ndarray]:
        return self.camera("camera_ring_rear_right")

    @property
    def radar_front(self) -> Optional[RadarSweep]:
        return self.radar("radar_front")
    
    @property
    def radar_left(self) -> Optional[RadarSweep]:
        return self.radar("radar_left")
    
    @property
    def radar_right(self) -> Optional[RadarSweep]:
        return self.radar("radar_right")
    
    def __repr__(self) -> str:
        return f"Frame(scene_id={self.scene_id}, frame_idx={self.frame_idx}, timestamp_ns={self.timestamp_ns})"
