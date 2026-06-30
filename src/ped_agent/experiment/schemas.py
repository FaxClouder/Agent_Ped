from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ScoreLevel(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    ADEQUATE = "adequate"
    INSUFFICIENT = "insufficient"
    POOR = "poor"


class ExperimentPlan(BaseModel):
    title: str = "Untitled experiment plan"
    research_question: str = ""
    hypothesis: str | None = None
    methodology: str = ""
    scenario_description: str = ""
    variables: dict = Field(default_factory=dict)
    sample_size: int | None = None
    equipment: list[str] = Field(default_factory=list)
    duration: str | None = None
    data_collection: str = ""
    analysis_methods: list[str] = Field(default_factory=list)


class DimensionScore(BaseModel):
    dimension: str
    score: float
    level: ScoreLevel
    checklist_results: list[dict] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)


class EvaluationResult(BaseModel):
    plan_title: str
    overall_score: float
    overall_level: ScoreLevel
    dimension_scores: list[DimensionScore]
    key_issues: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    literature_support: list[dict] = Field(default_factory=list)
    confidence: float = 0.5

