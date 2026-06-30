from __future__ import annotations

from datetime import UTC, datetime

from ped_agent.experiment.criteria import criteria_weight
from ped_agent.experiment.schemas import EvaluationResult


class ReportGenerator:
    def generate_markdown(self, result: EvaluationResult) -> str:
        dimension_table = "\n".join(
            "| {dimension} | {score:.1f} | {level} | {weight:.2f} |".format(
                dimension=score.dimension,
                score=score.score,
                level=score.level.value,
                weight=criteria_weight(score.dimension),
            )
            for score in result.dimension_scores
        )
        issues = "\n".join(f"- {issue}" for issue in result.key_issues) or "- None"
        recommendations = (
            "\n".join(f"{idx}. {item}" for idx, item in enumerate(result.recommendations, 1))
            or "1. No recommendations generated."
        )
        timestamp = datetime.now(UTC).isoformat()
        return f"""# Experiment Plan Evaluation Report

## Summary
- Title: {result.plan_title}
- Evaluated at: {timestamp}
- Overall score: {result.overall_score:.1f}/10 ({result.overall_level.value})
- Confidence: {result.confidence:.0%}

## Dimension Scores
| Dimension | Score | Level | Weight |
|---|---:|---|---:|
{dimension_table}

## Key Issues
{issues}

## Recommendations
{recommendations}
"""

    def generate_json(self, result: EvaluationResult) -> str:
        return result.model_dump_json(indent=2)

