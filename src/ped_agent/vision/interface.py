from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ped_agent.models.trajectory import TrajectoryData
from ped_agent.vision.schemas import Detection, ROI


@runtime_checkable
class VisionBackend(Protocol):
    def configure(self, config: dict) -> None: ...

    def process_video(self, video_path: str | Path, roi: ROI | None = None) -> TrajectoryData: ...

    def process_frame(self, frame: Any) -> list[Detection]: ...

    @property
    def capabilities(self) -> set[str]: ...

