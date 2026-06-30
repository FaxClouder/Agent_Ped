from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class DensityMetrics(BaseModel):
    mean_density: float = 0.0
    max_density: float = 0.0
    min_density: float = 0.0
    std_density: float = 0.0
    density_time_series: list[float] = Field(default_factory=list)


class VelocityMetrics(BaseModel):
    mean_speed: float = 0.0
    max_speed: float = 0.0
    min_speed: float = 0.0
    std_speed: float = 0.0
    speed_distribution: dict = Field(default_factory=dict)


class FlowMetrics(BaseModel):
    flow_rate: float = 0.0
    cross_section_id: str = "global"
    direction: str = "bidirectional"


class ODMatrix(BaseModel):
    zones: list[str]
    matrix: list[list[int]]
    total_trips: int


class FundamentalDiagram(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    density_bins: list[float]
    flow_values: list[float]
    speed_values: list[float]
    model_params: dict = Field(default_factory=dict)
    r_squared: float = 0.0


class AnalysisResult(BaseModel):
    scenario_id: str
    density: DensityMetrics
    velocity: VelocityMetrics
    flows: list[FlowMetrics] = Field(default_factory=list)
    od_matrix: ODMatrix | None = None
    fundamental_diagram: FundamentalDiagram | None = None
    visualizations: list[str] = Field(default_factory=list)
