from __future__ import annotations

from ped_agent.experiment.evaluator import ExperimentEvaluator
from ped_agent.experiment.report import ReportGenerator


def test_experiment_evaluator_fallback_report():
    plan = """
    Station evacuation experiment
    We will recruit 40 participants, define control and baseline conditions,
    record video trajectory data, and analyze density and velocity statistics.
    """

    result = ExperimentEvaluator().evaluate_sync(plan)
    report = ReportGenerator().generate_markdown(result)

    assert result.overall_score > 0
    assert "Experiment Plan Evaluation Report" in report

