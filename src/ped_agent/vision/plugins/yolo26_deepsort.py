from __future__ import annotations

from ped_agent.vision.plugins.yolo26_bytetrack import YOLO26ByteTrackBackend
from ped_agent.vision.registry import VisionRegistry


@VisionRegistry.register("yolo26_deepsort")
class YOLO26DeepSORTBackend(YOLO26ByteTrackBackend):
    def __init__(self, config: dict | None = None):
        merged = dict(config or {})
        merged.setdefault("tracker", {})["algorithm"] = "deepsort"
        super().__init__(merged)

    def configure(self, config: dict) -> None:
        config.setdefault("tracker", {})["algorithm"] = "deepsort"
        super().configure(config)

