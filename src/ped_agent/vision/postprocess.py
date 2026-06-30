from __future__ import annotations

import numpy as np

from ped_agent.models.trajectory import PedestrianTrack, Position


class TrajectoryPostProcessor:
    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.min_track_length = self.config.get("min_track_length", 10)
        self.fps = float(self.config.get("fps", 25.0))

    def process(self, raw_tracks: dict[int, list[tuple[int, float, float, float]]]) -> list[PedestrianTrack]:
        tracks: list[PedestrianTrack] = []
        for track_id, points in raw_tracks.items():
            if len(points) < self.min_track_length:
                continue
            frames = [int(point[0]) for point in points]
            positions = [Position(x=float(point[1]), y=float(point[2])) for point in points]
            confidence = [float(point[3]) for point in points]
            timestamps = [frame / self.fps for frame in frames]
            tracks.append(
                PedestrianTrack(
                    track_id=int(track_id),
                    frames=frames,
                    positions=positions,
                    timestamps=timestamps,
                    confidence=confidence,
                )
            )
        return tracks


def bbox_bottom_centers(tracks: np.ndarray) -> np.ndarray:
    if tracks.size == 0:
        return np.empty((0, 2))
    return np.column_stack(((tracks[:, 0] + tracks[:, 2]) / 2.0, tracks[:, 3]))

