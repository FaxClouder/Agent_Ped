from __future__ import annotations

from ped_agent.analysis.metrics import compute_los
from ped_agent.analysis.pipeline import AnalysisPipeline
from ped_agent.models.scenario_data import ScenarioInput, ScenarioMetadata
from ped_agent.models.trajectory import PedestrianTrack, Position


def test_compute_los_thresholds():
    assert compute_los(0.1) == "A"
    assert compute_los(0.5) == "C"
    assert compute_los(3.0) == "F"


def test_analysis_pipeline_smoke():
    track = PedestrianTrack(
        track_id=1,
        frames=[0, 25],
        positions=[Position(x=0, y=0), Position(x=1, y=0)],
        timestamps=[0.0, 1.0],
        confidence=[0.9, 0.95],
    )
    scenario = ScenarioInput(
        metadata=ScenarioMetadata(
            scenario_id="demo",
            fps=25,
            duration=1.0,
            area_definition={"type": "rectangle"},
        ),
        trajectories=[track],
        area_m2=10.0,
    )

    result = AnalysisPipeline().analyze_scenario(scenario)

    assert result.scenario_id == "demo"
    assert result.density.mean_density > 0
    assert result.velocity.mean_speed == 1.0

