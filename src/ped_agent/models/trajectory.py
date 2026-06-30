from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class Position(BaseModel):
    x: float
    y: float


class PedestrianTrack(BaseModel):
    track_id: int
    frames: list[int]
    positions: list[Position]
    timestamps: list[float]
    confidence: list[float] | None = None

    @field_validator("positions")
    @classmethod
    def positions_match_frames(cls, value: list[Position], info):
        frames = info.data.get("frames", [])
        if frames and len(value) != len(frames):
            raise ValueError("positions length must match frames length")
        return value

    @field_validator("timestamps")
    @classmethod
    def timestamps_match_frames(cls, value: list[float], info):
        frames = info.data.get("frames", [])
        if frames and len(value) != len(frames):
            raise ValueError("timestamps length must match frames length")
        return value


class VideoMetadata(BaseModel):
    source: str
    fps: float = Field(gt=0)
    total_frames: int = Field(ge=0)
    resolution: tuple[int, int]
    duration: float = Field(ge=0)


class TrajectoryData(BaseModel):
    video_meta: VideoMetadata
    tracks: list[PedestrianTrack] = Field(default_factory=list)

