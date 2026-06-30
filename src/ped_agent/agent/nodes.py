from __future__ import annotations

from ped_agent.agent.state import AgentState, QueryType


KEYWORDS: dict[QueryType, tuple[str, ...]] = {
    "experiment_eval": (
        "experiment",
        "evaluate",
        "evaluation",
        "plan",
        "\u5b9e\u9a8c",
        "\u8bc4\u4f30",
        "\u65b9\u6848",
    ),
    "data_analysis": (
        "density",
        "velocity",
        "flow",
        "trajectory",
        "od matrix",
        "\u5bc6\u5ea6",
        "\u901f\u5ea6",
        "\u6d41\u91cf",
        "\u8f68\u8ff9",
        "\u5206\u6790",
    ),
    "recommendation": (
        "recommend",
        "suggest",
        "improve",
        "\u63a8\u8350",
        "\u5efa\u8bae",
        "\u6539\u8fdb",
    ),
    "literature_qa": (
        "paper",
        "literature",
        "citation",
        "method",
        "\u6587\u732e",
        "\u8bba\u6587",
        "\u5f15\u7528",
        "\u65b9\u6cd5",
    ),
}


def classify_query(state: AgentState) -> AgentState:
    query = state.get("query", "")
    state["query_type"] = classify_text(query)
    return state


def classify_text(query: str) -> QueryType:
    normalized = query.lower()
    for query_type, keywords in KEYWORDS.items():
        if any(_matches_keyword(normalized, keyword) for keyword in keywords):
            return query_type
    return "literature_qa"


def _matches_keyword(text: str, keyword: str) -> bool:
    if keyword.isascii() and keyword.replace(" ", "").isalnum():
        return keyword in text.split() or keyword in text
    return keyword in text


def retrieve_literature(state: AgentState) -> AgentState:
    state["context"] = [
        "RAG scaffold is ready. Index papers with scripts/index_papers.py before live retrieval."
    ]
    state["answer"] = (
        "I routed this as a literature QA task. The retrieval interface is scaffolded; "
        "connect embeddings and a vector store to return cited answers."
    )
    return state


def evaluate_experiment(state: AgentState) -> AgentState:
    state["answer"] = (
        "I routed this as an experiment evaluation task. The evaluator scaffold is ready "
        "for an LLM and retriever-backed scoring workflow."
    )
    return state


def analyze_data(state: AgentState) -> AgentState:
    state["answer"] = (
        "I routed this as a data analysis task. Provide ScenarioInput data to run density, "
        "velocity, flow, OD, and visualization pipelines."
    )
    return state


def generate_recommendation(state: AgentState) -> AgentState:
    state["answer"] = (
        "I routed this as a recommendation task. The synthesis layer can combine retrieved "
        "literature, scenario metrics, and experiment findings."
    )
    return state


def synthesize(state: AgentState) -> AgentState:
    state["result"] = state.get("answer", "")
    return state
