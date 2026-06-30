from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ped_agent.agent.nodes import (
    analyze_data,
    classify_query,
    evaluate_experiment,
    generate_recommendation,
    retrieve_literature,
    synthesize,
)
from ped_agent.agent.state import AgentState


class FallbackAgentGraph:
    """Small in-process graph used when LangGraph is not installed."""

    def __init__(self):
        self.handlers: dict[str, Callable[[AgentState], AgentState]] = {
            "literature_qa": retrieve_literature,
            "experiment_eval": evaluate_experiment,
            "data_analysis": analyze_data,
            "recommendation": generate_recommendation,
        }

    def invoke(self, state: AgentState) -> AgentState:
        state = classify_query(dict(state))
        handler = self.handlers[state["query_type"]]
        state = handler(state)
        return synthesize(state)


def build_agent_graph(config: Any | None = None):
    """Build the Phase 1 routing graph.

    If LangGraph is installed, this returns a compiled StateGraph. Otherwise it returns a
    compatible fallback object with an invoke() method so tests and local smoke runs work.
    """
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError:
        return FallbackAgentGraph()

    builder = StateGraph(AgentState)
    builder.add_node("classify_query", classify_query)
    builder.add_node("retrieve_literature", retrieve_literature)
    builder.add_node("evaluate_experiment", evaluate_experiment)
    builder.add_node("analyze_data", analyze_data)
    builder.add_node("generate_recommendation", generate_recommendation)
    builder.add_node("synthesize", synthesize)

    builder.add_edge(START, "classify_query")
    builder.add_conditional_edges(
        "classify_query",
        lambda state: state["query_type"],
        {
            "literature_qa": "retrieve_literature",
            "experiment_eval": "evaluate_experiment",
            "data_analysis": "analyze_data",
            "recommendation": "generate_recommendation",
        },
    )
    for node_name in (
        "retrieve_literature",
        "evaluate_experiment",
        "analyze_data",
        "generate_recommendation",
    ):
        builder.add_edge(node_name, "synthesize")
    builder.add_edge("synthesize", END)
    return builder.compile(name="ped-agent-phase1")

