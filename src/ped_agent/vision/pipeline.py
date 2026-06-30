from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ped_agent.models.trajectory import TrajectoryData, VideoMetadata
from ped_agent.vision.detector import PedestrianDetector
from ped_agent.vision.postprocess import TrajectoryPostProcessor, bbox_bottom_centers
from ped_agent.vision.tracker import PedestrianTracker
from ped_agent.vision.transform import CoordinateTransformer


class VisionPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.detector = PedestrianDetector(config.get("detector", {}))
        self.tracker = PedestrianTracker(config.get("tracker", {}))
        self.transformer = CoordinateTransformer(config.get("coordinate_transform", {}))
        self.postprocessor = TrajectoryPostProcessor(config.get("postprocessing", {}))
        self.skip_frames = int(config.get("preprocessing", {}).get("skip_frames", 1))

    def process_video(self, video_path: str | Path, roi: Any = None) -> TrajectoryData:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("Install ped-agent[vision] to process video.") from exc

        cap = cv2.VideoCapture(str(video_path))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.postprocessor.fps = fps

        raw_tracks: dict[int, list[tuple[int, float, float, float]]] = {}
        frame_idx = 0
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % self.skip_frames != 0:
                frame_idx += 1
                continue

            detections = self.detector.detect(frame)
            tracks = self.tracker.update(detections, frame)
            if len(tracks):
                centers = self.transformer.transform(bbox_bottom_centers(tracks))
                for track, center in zip(tracks, centers, strict=True):
                    track_id = int(track[4])
                    confidence = float(track[5]) if len(track) > 5 else 1.0
                    raw_tracks.setdefault(track_id, []).append(
                        (frame_idx, float(center[0]), float(center[1]), confidence)
                    )
            frame_idx += 1

        cap.release()
        return TrajectoryData(
            video_meta=VideoMetadata(
                source=str(video_path),
                fps=fps,
                total_frames=total_frames,
                resolution=(width, height),
                duration=total_frames / fps if fps else 0.0,
            ),
            tracks=self.postprocessor.process(raw_tracks),
        )

