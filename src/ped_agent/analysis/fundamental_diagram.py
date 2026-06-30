from __future__ import annotations

import numpy as np

from ped_agent.analysis.metrics import compute_track_speeds
from ped_agent.analysis.schemas import FundamentalDiagram
from ped_agent.models.trajectory import PedestrianTrack


def compute_fundamental_diagram(
    trajectories: list[PedestrianTrack],
    area_m2: float,
    time_bin: float = 1.0,
) -> FundamentalDiagram:
    if not trajectories or area_m2 <= 0:
        return FundamentalDiagram(density_bins=[], flow_values=[], speed_values=[])

    max_time = max((max(track.timestamps) for track in trajectories if track.timestamps), default=0)
    bins = np.arange(0, max_time + time_bin, time_bin)
    densities: list[float] = []
    flows: list[float] = []
    speeds: list[float] = []

    for start in bins:
        end = start + time_bin
        active = []
        speed_values = []
        for track in trajectories:
            if any(start <= timestamp < end for timestamp in track.timestamps):
                active.append(track.track_id)
                track_speeds = compute_track_speeds(track)
                if track_speeds.size:
                    speed_values.append(float(track_speeds.mean()))
        if not active:
            continue
        density = len(set(active)) / area_m2
        speed = float(np.mean(speed_values)) if speed_values else 0.0
        densities.append(density)
        speeds.append(speed)
        flows.append(density * speed)

    params: dict[str, float] = {}
    r_squared = 0.0
    if len(densities) >= 2 and len(set(densities)) >= 2:
        x = np.array(densities)
        y = np.array(flows)
        slope, intercept = np.polyfit(x, y, 1)
        predicted = slope * x + intercept
        denom = float(((y - y.mean()) ** 2).sum())
        r_squared = 1.0 - float(((y - predicted) ** 2).sum()) / denom if denom else 0.0
        params = {"slope": float(slope), "intercept": float(intercept)}

    return FundamentalDiagram(
        density_bins=densities,
        flow_values=flows,
        speed_values=speeds,
        model_params=params,
        r_squared=r_squared,
    )
