from __future__ import annotations

from ped_agent.experiment.evaluator import ExperimentEvaluator
from ped_agent.experiment.schemas import EvaluationResult


async def evaluate_with_consistency(
    evaluator: ExperimentEvaluator,
    plan_text: str,
    n_rounds: int = 3,
) -> EvaluationResult:
    # The fallback evaluator is deterministic; keep the interface for Phase 5 LLM sampling.
    _ = n_rounds
    return await evaluator.evaluate(plan_text)

