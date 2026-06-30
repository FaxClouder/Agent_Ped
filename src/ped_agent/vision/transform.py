from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml


class CoordinateTransformer:
    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.method = self.config.get("method", "none")
        self.pixel_per_meter = self.config.get("pixel_per_meter")
        self.homography_matrix = None
        if self.method == "homography" and self.config.get("calibration_file"):
            self.homography_matrix = self._load_matrix(self.config["calibration_file"])

    def transform(self, pixel_points: np.ndarray) -> np.ndarray:
        if self.method == "none":
            return pixel_points
        if self.method == "scale":
            if not self.pixel_per_meter:
                raise ValueError("pixel_per_meter is required for scale transform.")
            return pixel_points / float(self.pixel_per_meter)
        if self.method == "homography":
            if self.homography_matrix is None:
                raise ValueError("homography_matrix is not configured.")
            try:
                import cv2
            except ImportError as exc:
                raise RuntimeError("Install ped-agent[vision] for homography transforms.") from exc
            points = pixel_points.reshape(-1, 1, 2).astype(np.float64)
            return cv2.perspectiveTransform(points, self.homography_matrix).reshape(-1, 2)
        raise ValueError(f"Unknown coordinate transform: {self.method}")

    @staticmethod
    def _load_matrix(path: str | Path) -> np.ndarray:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return np.array(payload["homography_matrix"], dtype=float)

