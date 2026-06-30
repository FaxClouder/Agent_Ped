from __future__ import annotations

from typing import Any

from ped_agent.analysis.fundamental_diagram import compute_fundamental_diagram
from ped_agent.analysis.metrics import (
    compute_density_series,
    compute_global_flow,
    compute_velocity_metrics,
)
from ped_agent.analysis.schemas import AnalysisResult
from ped_agent.analysis.visualizer import Visualizer
from ped_agent.models.scenario_data import ScenarioInput
from ped_agent.utils.config import select


class AnalysisPipeline:
    def __init__(self, config: Any | None = None):
        self.config = config or {}
        self.visualizer = Visualizer(select(config, "analysis.visualization", {}) if config else {})

    def analyze_scenario(self, scenario: ScenarioInput) -> AnalysisResult:
        density = compute_density_series(scenario)
        velocity = compute_velocity_metrics(scenario.trajectories)
        flows = [compute_global_flow(scenario)]
        fd = None
        if select(self.config, "analysis.fundamental_diagram.enabled", True):
            fd = compute_fundamental_diagram(scenario.trajectories, scenario.area_m2)

        result = AnalysisResult(
            scenario_id=scenario.metadata.scenario_id,
            density=density,
            velocity=velocity,
            flows=flows,
            fundamental_diagram=fd,
        )
        result.visualizations = self.visualizer.generate_all(result, scenario)
        return result

