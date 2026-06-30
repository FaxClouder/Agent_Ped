from __future__ import annotations

from typing import Any

from ped_agent.vision.schemas import Detection


class PedestrianDetector:
    def __init__(self, config: dict):
        self.config = config
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install ped-agent[vision] to use YOLO detection.") from exc
        self.model = YOLO(config.get("model", "yolo26x.pt"))

    def detect(self, frame: Any) -> list[Detection]:
        results = self.model(
            frame,
            conf=self.config.get("confidence", 0.5),
            iou=self.config.get("iou_threshold", 0.45),
            classes=self.config.get("classes", [0]),
            imgsz=self.config.get("input_size", 1280),
            device=self.config.get("device", "cpu"),
            half=self.config.get("half_precision", False),
            verbose=False,
        )
        detections: list[Detection] = []
        boxes = getattr(results[0], "boxes", None)
        if boxes is None:
            return detections
        for idx in range(len(boxes)):
            xyxy = tuple(float(x) for x in boxes.xyxy[idx].cpu().numpy())
            detections.append(
                Detection(
                    bbox=xyxy,
                    confidence=float(boxes.conf[idx]),
                    class_id=int(boxes.cls[idx]),
                )
            )
        return detections

