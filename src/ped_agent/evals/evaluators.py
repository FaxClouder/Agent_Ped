from __future__ import annotations


def citation_accuracy(run, example) -> dict:
    predicted = set(run.outputs.get("citations", []))
    expected = set(example.outputs.get("citations", []))
    if not predicted:
        return {"key": "citation_accuracy", "score": 0.0 if expected else 1.0}
    return {"key": "citation_accuracy", "score": len(predicted & expected) / len(predicted)}


def score_in_expected_range(run, example) -> dict:
    score = float(run.outputs.get("overall_score", 0.0))
    lower, upper = example.outputs.get("expected_overall_range", [0.0, 10.0])
    return {"key": "score_in_expected_range", "score": float(lower <= score <= upper)}

