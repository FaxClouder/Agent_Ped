from __future__ import annotations


def literature_qa_seed_examples() -> list[dict]:
    return [
        {
            "inputs": {"question": "What are common parameters in social-force models?"},
            "outputs": {"citations": [], "answer": ""},
        }
    ]


def experiment_eval_seed_examples() -> list[dict]:
    return [
        {
            "inputs": {"plan_text": "Observe pedestrian flow with video tracking."},
            "outputs": {"expected_overall_range": [4, 8], "expected_issues": []},
        }
    ]

