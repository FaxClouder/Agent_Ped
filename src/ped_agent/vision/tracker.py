from __future__ import annotations

from typing import Any

import numpy as np

from ped_agent.vision.schemas import Detection


class PedestrianTracker:
    def __init__(self, config: dict):
        self.config = config
        self.algorithm = config.get("algorithm", "bytetrack")
        self.tracker = self._build_tracker()

    def _build_tracker(self):
        try:
            from boxmot import ByteTrack, DeepSORT
        except ImportError as exc:
            raise RuntimeError("Install ped-agent[vision] to use multi-object tracking.") from exc

        if self.algorithm == "bytetrack":
            return ByteTrack(
                track_buffer=self.config.get("track_buffer", 30),
                match_thresh=self.config.get("match_thresh", 0.8),
            )
        if self.algorithm == "deepsort":
            return DeepSORT(
                max_age=self.config.get("track_buffer", 30),
                max_cosine_distance=self.config.get("max_cosine_distance", 0.3),
            )
        raise ValueError(f"Unknown tracker: {self.algorithm}")

    def update(self, detections: list[Detection], frame: Any) -> np.ndarray:
        dets = np.array([[*det.bbox, det.confidence] for det in detections], dtype=float)
        if dets.size == 0:
            dets = np.empty((0, 5))
        return self.tracker.update(dets, frame)

    def reset(self) -> None:
        self.tracker = self._build_tracker()

