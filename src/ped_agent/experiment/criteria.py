from __future__ import annotations

from pydantic import BaseModel


class DimensionCriteria(BaseModel):
    dimension: str
    weight: float
    checklist: list[str]


EVALUATION_CRITERIA = [
    DimensionCriteria(
        dimension="feasibility",
        weight=0.25,
        checklist=[
            "site_or_data_access",
            "equipment_availability",
            "participant_recruitment",
            "timeline_and_budget",
            "ethics_review",
        ],
    ),
    DimensionCriteria(
        dimension="completeness",
        weight=0.25,
        checklist=[
            "control_or_baseline",
            "variables_defined",
            "sample_size_defined",
            "confounders_controlled",
            "analysis_plan_defined",
        ],
    ),
    DimensionCriteria(
        dimension="methodology",
        weight=0.25,
        checklist=[
            "realistic_scenario",
            "standardized_measurements",
            "trajectory_quality_plan",
            "statistical_method_fit",
            "uncertainty_considered",
        ],
    ),
    DimensionCriteria(
        dimension="innovation",
        weight=0.15,
        checklist=[
            "clear_research_gap",
            "new_method_or_setting",
            "theoretical_or_practical_value",
        ],
    ),
    DimensionCriteria(
        dimension="reproducibility",
        weight=0.10,
        checklist=[
            "detailed_steps",
            "parameter_settings",
            "data_format_defined",
            "code_or_data_release_plan",
        ],
    ),
]


def criteria_weight(dimension: str) -> float:
    for criterion in EVALUATION_CRITERIA:
        if criterion.dimension == dimension:
            return criterion.weight
    raise KeyError(dimension)

