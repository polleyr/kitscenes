from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

from kitscenes.frame import Frame

if TYPE_CHECKING:
    from kitscenes.dataset import KITScenesDataset


class FrameDataset:
    def __init__(self, dataset: KITScenesDataset) -> None:
        self._dataset = dataset

        self._index: list[tuple[str, int]] = []
        for scene_id in dataset.scene_ids:
            loader = dataset.get_sensor_loader(scene_id)
            num_frames = len(loader.get_reference_timestamps())
            for frame_idx in range(num_frames):
                self._index.append((scene_id, frame_idx))

    def __len__(self) -> int:
        return len(self._index)
    
    def __getitem__(self, global_idx: int) -> Frame:
        scene_id, frame_idx = self._index[global_idx]
        loader = self._dataset.get_sensor_loader(scene_id)
        timestamp_ns = int(loader.get_reference_timestamps()[frame_idx])

        return Frame(
            scene_id=scene_id,
            frame_idx=frame_idx,
            timestamp_ns=timestamp_ns,
            loader=loader,
            ego_poses=self._dataset.get_scene(scene_id).ego_poses,
        )

    def __iter__(self) -> Iterator[Frame]:
        for global_idx in range(len(self)):
            yield self[global_idx]

    def __repr__(self) -> str:
        return f"FrameDataset(num_scenes={len(self._dataset.scene_ids)}, num_frames={len(self)})"
