from __future__ import annotations

from collections import defaultdict

import numpy as np

from ped_agent.analysis.schemas import DensityMetrics, FlowMetrics, VelocityMetrics
from ped_agent.models.scenario_data import ScenarioInput
from ped_agent.models.trajectory import PedestrianTrack


def compute_density_series(scenario: ScenarioInput, time_bin: float = 1.0) -> DensityMetrics:
    if not scenario.trajectories:
        return DensityMetrics()

    bins: dict[int, set[int]] = defaultdict(set)
    for track in scenario.trajectories:
        for timestamp in track.timestamps:
            bins[int(timestamp // time_bin)].add(track.track_id)

    if not bins:
        return DensityMetrics()

    max_bin = max(bins)
    series = [len(bins.get(idx, set())) / scenario.area_m2 for idx in range(max_bin + 1)]
    values = np.array(series, dtype=float)
    return DensityMetrics(
        mean_density=float(values.mean()),
        max_density=float(values.max()),
        min_density=float(values.min()),
        std_density=float(values.std()),
        density_time_series=series,
    )


def compute_track_speeds(track: PedestrianTrack) -> np.ndarray:
    if len(track.positions) < 2:
        return np.array([], dtype=float)
    positions = np.array([[point.x, point.y] for point in track.positions], dtype=float)
    timestamps = np.array(track.timestamps, dtype=float)
    dt = np.diff(timestamps)
    valid = dt > 0
    if not valid.any():
        return np.array([], dtype=float)
    distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    return distances[valid] / dt[valid]


def compute_velocity_metrics(trajectories: list[PedestrianTrack]) -> VelocityMetrics:
    speeds = np.concatenate([compute_track_speeds(track) for track in trajectories] or [[]])
    speeds = speeds[np.isfinite(speeds)]
    if speeds.size == 0:
        return VelocityMetrics(speed_distribution={"bins": [], "counts": []})

    counts, bins = np.histogram(speeds, bins=min(10, max(1, speeds.size)))
    return VelocityMetrics(
        mean_speed=float(speeds.mean()),
        max_speed=float(speeds.max()),
        min_speed=float(speeds.min()),
        std_speed=float(speeds.std()),
        speed_distribution={"bins": bins[:-1].tolist(), "counts": counts.tolist()},
    )


def compute_global_flow(scenario: ScenarioInput) -> FlowMetrics:
    if scenario.metadata.duration <= 0:
        return FlowMetrics()
    completed_tracks = sum(1 for track in scenario.trajectories if len(track.positions) >= 2)
    return FlowMetrics(flow_rate=completed_tracks / scenario.metadata.duration)


def compute_los(density: float) -> str:
    if density < 0.27:
        return "A"
    if density < 0.43:
        return "B"
    if density < 0.72:
        return "C"
    if density < 1.08:
        return "D"
    if density < 2.17:
        return "E"
    return "F"

