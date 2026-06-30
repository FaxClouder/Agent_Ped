from __future__ import annotations

from typing import Literal, TypedDict

QueryType = Literal["literature_qa", "experiment_eval", "data_analysis", "recommendation"]


class AgentState(TypedDict, total=False):
    query: str
    query_type: QueryType
    context: list[str]
    artifacts: list[str]
    answer: str
    result: str
    errors: list[str]

