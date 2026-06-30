from __future__ import annotations

import re

from ped_agent.experiment.criteria import EVALUATION_CRITERIA, criteria_weight
from ped_agent.experiment.schemas import DimensionScore, EvaluationResult, ScoreLevel


class ExperimentEvaluator:
    """Experiment-plan evaluator with an LLM-ready interface and heuristic fallback."""

    def __init__(self, llm=None, retriever=None):
        self.llm = llm
        self.retriever = retriever

    async def evaluate(self, plan_text: str) -> EvaluationResult:
        if self.llm is not None and hasattr(self.llm, "with_structured_output"):
            return await self._llm_evaluate(plan_text)
        return self.evaluate_sync(plan_text)

    def evaluate_sync(self, plan_text: str) -> EvaluationResult:
        lower = plan_text.lower()
        dimension_scores = [
            self._score_dimension(criterion.dimension, criterion.checklist, lower)
            for criterion in EVALUATION_CRITERIA
        ]
        overall = sum(score.score * criteria_weight(score.dimension) for score in dimension_scores)
        issues = [
            weakness
            for score in dimension_scores
            for weakness in score.weaknesses
            if score.score < 7
        ]
        recommendations = [
            f"Strengthen {score.dimension}: {', '.join(score.weaknesses[:2])}"
            for score in dimension_scores
            if score.weaknesses
        ]
        return EvaluationResult(
            plan_title=_infer_title(plan_text),
            overall_score=round(overall, 2),
            overall_level=_score_to_level(overall),
            dimension_scores=dimension_scores,
            key_issues=issues[:8],
            recommendations=recommendations[:8],
            confidence=0.45,
        )

    async def _llm_evaluate(self, plan_text: str) -> EvaluationResult:
        raise NotImplementedError("LLM-backed evaluation will be implemented in Phase 5.")

    @staticmethod
    def _score_dimension(dimension: str, checklist: list[str], lower_text: str) -> DimensionScore:
        hits = [_keyword_hit(item, lower_text) for item in checklist]
        passed = sum(hits)
        score = max(2.0, min(9.0, 2.0 + 7.0 * passed / max(1, len(checklist))))
        weaknesses = [item for item, hit in zip(checklist, hits, strict=True) if not hit]
        strengths = [item for item, hit in zip(checklist, hits, strict=True) if hit]
        return DimensionScore(
            dimension=dimension,
            score=round(score, 1),
            level=_score_to_level(score),
            checklist_results=[
                {"item": item, "passed": hit} for item, hit in zip(checklist, hits, strict=True)
            ],
            strengths=strengths,
            weaknesses=weaknesses,
        )


def _keyword_hit(item: str, lower_text: str) -> bool:
    groups = {
        "sample": ("sample", "participant", "power", "\u6837\u672c", "\u53c2\u4e0e\u8005"),
        "control": ("control", "baseline", "\u5bf9\u7167", "\u57fa\u7ebf"),
        "variable": ("variable", "independent", "dependent", "\u53d8\u91cf"),
        "trajectory": ("trajectory", "tracking", "video", "\u8f68\u8ff9", "\u89c6\u9891"),
        "statistical": ("statistic", "regression", "anova", "\u7edf\u8ba1"),
        "data": ("data", "dataset", "csv", "json", "\u6570\u636e"),
        "equipment": ("camera", "sensor", "equipment", "\u8bbe\u5907", "\u6444\u50cf"),
        "ethics": ("ethic", "consent", "privacy", "\u4f26\u7406", "\u9690\u79c1"),
    }
    tokens = tuple(re.split(r"[_\s]+", item))
    candidates = set(tokens)
    for token in tokens:
        candidates.update(groups.get(token, ()))
    return any(candidate in lower_text for candidate in candidates if candidate)


def _score_to_level(score: float) -> ScoreLevel:
    if score >= 9:
        return ScoreLevel.EXCELLENT
    if score >= 7:
        return ScoreLevel.GOOD
    if score >= 5:
        return ScoreLevel.ADEQUATE
    if score >= 3:
        return ScoreLevel.INSUFFICIENT
    return ScoreLevel.POOR


def _infer_title(plan_text: str) -> str:
    first_line = next((line.strip() for line in plan_text.splitlines() if line.strip()), "")
    return first_line[:80] or "Untitled experiment plan"

