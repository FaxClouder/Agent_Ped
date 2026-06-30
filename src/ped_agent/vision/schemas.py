from __future__ import annotations

from pydantic import BaseModel

from ped_agent.models.trajectory import PedestrianTrack, Position, TrajectoryData, VideoMetadata


class Detection(BaseModel):
    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int = 0


class ROI(BaseModel):
    type: str = "polygon"
    points: list[tuple[float, float]]


__all__ = [
    "Detection",
    "PedestrianTrack",
    "Position",
    "ROI",
    "TrajectoryData",
    "VideoMetadata",
]

