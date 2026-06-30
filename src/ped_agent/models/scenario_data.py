from __future__ import annotations

from pydantic import BaseModel, Field

from ped_agent.models.trajectory import PedestrianTrack


class ScenarioMetadata(BaseModel):
    scenario_id: str
    fps: float = Field(default=25.0, gt=0)
    duration: float = Field(ge=0)
    area_definition: dict = Field(default_factory=dict)


class ScenarioInput(BaseModel):
    metadata: ScenarioMetadata
    trajectories: list[PedestrianTrack] = Field(default_factory=list)
    area_m2: float = Field(gt=0)

