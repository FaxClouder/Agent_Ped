from __future__ import annotations

from ped_agent.agent.graph import build_agent_graph
from ped_agent.agent.nodes import classify_text


def test_classify_query_types():
    assert classify_text("evaluate this experiment plan") == "experiment_eval"
    assert classify_text("compute density and velocity") == "data_analysis"
    assert classify_text("recommend improvements") == "recommendation"
    assert classify_text("find literature about social force model") == "literature_qa"


def test_graph_fallback_invokes_route():
    graph = build_agent_graph()
    result = graph.invoke({"query": "compute density from trajectory data"})

    assert result["query_type"] == "data_analysis"
    assert "data analysis" in result["answer"]

