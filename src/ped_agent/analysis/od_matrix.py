from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ped_agent.analysis.schemas import ODMatrix
from ped_agent.models.trajectory import PedestrianTrack, Position


@dataclass(frozen=True)
class RectZone:
    zone_id: str
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def contains(self, point: Position) -> bool:
        return self.xmin <= point.x <= self.xmax and self.ymin <= point.y <= self.ymax


def build_od_matrix(trajectories: list[PedestrianTrack], zones: list[RectZone]) -> ODMatrix:
    counts = np.zeros((len(zones), len(zones)), dtype=int)
    for track in trajectories:
        if len(track.positions) < 2:
            continue
        origin = _find_zone(track.positions[0], zones)
        destination = _find_zone(track.positions[-1], zones)
        if origin is not None and destination is not None:
            counts[origin, destination] += 1

    return ODMatrix(
        zones=[zone.zone_id for zone in zones],
        matrix=counts.tolist(),
        total_trips=int(counts.sum()),
    )


def _find_zone(point: Position, zones: list[RectZone]) -> int | None:
    for idx, zone in enumerate(zones):
        if zone.contains(point):
            return idx
    return None

