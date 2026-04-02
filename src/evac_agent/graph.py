from __future__ import annotations

import re
from typing import Literal, Type

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from evac_agent.config import get_chat_model_kwargs, get_settings
from evac_agent.models import AgentState, AnswerAudit, QueryAssessment, RetrievalAudit
from evac_agent.prompts import (
    ANSWER_PROMPT,
    ASSESSMENT_PROMPT,
    AUDIT_PROMPT,
    RETRIEVAL_AUDIT_PROMPT,
    REVISION_PROMPT,
)
from evac_agent.rag import retrieve_context


MAX_RETRIEVAL_ATTEMPTS = 2


def _llm() -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(**get_chat_model_kwargs(settings))


def _extract_json_object(text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        return fenced.group(1)

    bare = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if bare:
        return bare.group(1)

    raise ValueError("No JSON object found in model output.")


def _invoke_structured(prompt: str, schema: Type[BaseModel]):
    llm = _llm()
    try:
        return llm.with_structured_output(schema).invoke(prompt)
    except Exception:
        fallback_prompt = (
            prompt
            + "\n\n请严格仅输出一个 JSON 对象，不要输出 Markdown、解释或多余文本。"
            + "\nJSON Schema:\n"
            + str(schema.model_json_schema())
        )
        raw = llm.invoke(fallback_prompt)
        content = raw.content if isinstance(raw.content, str) else str(raw.content)
        json_text = _extract_json_object(content)
        return schema.model_validate_json(json_text)


def assess_question(state: AgentState) -> AgentState:
    assessment = _invoke_structured(ASSESSMENT_PROMPT.format(question=state["question"]), QueryAssessment)
    queries = assessment.search_queries or [state["question"]]
    return {
        "assessment": assessment,
        "active_queries": queries,
        "retrieval_attempt": 0,
    }


def retrieve_knowledge(state: AgentState) -> AgentState:
    queries = state.get("active_queries") or state["assessment"].search_queries or [state["question"]]
    attempt = state.get("retrieval_attempt", 0) + 1
    context, sources = retrieve_context(question=state["question"], queries=queries)
    return {
        "active_queries": queries,
        "retrieval_attempt": attempt,
        "retrieved_context": context,
        "retrieved_sources": sources,
    }


def audit_retrieval(state: AgentState) -> AgentState:
    audit = _invoke_structured(
        RETRIEVAL_AUDIT_PROMPT.format(
            question=state["question"],
            assessment=state["assessment"].model_dump_json(indent=2, ensure_ascii=False),
            queries="\n".join(state.get("active_queries", [])),
            context=state.get("retrieved_context", ""),
            sources="\n".join(state.get("retrieved_sources", [])),
        ),
        RetrievalAudit,
    )

    next_queries = state.get("active_queries", [])
    if not audit.sufficient and audit.refined_queries:
        next_queries = audit.refined_queries
    elif not audit.sufficient and not next_queries:
        next_queries = [state["question"]]

    return {
        "retrieval_audit": audit,
        "active_queries": next_queries,
    }


def route_after_retrieval(state: AgentState) -> Literal["retrieve", "draft"]:
    audit = state["retrieval_audit"]
    attempts = state.get("retrieval_attempt", 1)
    if not audit.sufficient and attempts < MAX_RETRIEVAL_ATTEMPTS:
        return "retrieve"
    return "draft"


def draft_answer(state: AgentState) -> AgentState:
    llm = _llm()
    response = llm.invoke(
        ANSWER_PROMPT.format(
            question=state["question"],
            assessment=state["assessment"].model_dump_json(indent=2, ensure_ascii=False),
            context=state.get("retrieved_context", ""),
        )
    )
    return {"draft_answer": response.content}


def audit_answer(state: AgentState) -> AgentState:
    audit = _invoke_structured(
        AUDIT_PROMPT.format(
            question=state["question"],
            assessment=state["assessment"].model_dump_json(indent=2, ensure_ascii=False),
            context=state.get("retrieved_context", ""),
            draft=state.get("draft_answer", ""),
        ),
        AnswerAudit,
    )
    return {"audit": audit}


def revise_answer(state: AgentState) -> AgentState:
    llm = _llm()
    issues = "\n".join(f"- {issue}" for issue in state["audit"].issues) or "- 未说明"
    response = llm.invoke(
        REVISION_PROMPT.format(
            question=state["question"],
            assessment=state["assessment"].model_dump_json(indent=2, ensure_ascii=False),
            context=state.get("retrieved_context", ""),
            issues=issues,
            draft=state.get("draft_answer", ""),
        )
    )
    return {"final_answer": response.content}


def finalize_answer(state: AgentState) -> AgentState:
    answer = state.get("final_answer") or state.get("draft_answer", "")
    sources = state.get("retrieved_sources", [])
    if sources:
        source_block = "\n\n参考来源：\n" + "\n".join(f"- {source}" for source in sources)
        answer += source_block
    return {"final_answer": answer}


def route_after_answer_audit(state: AgentState) -> Literal["revise", "finalize"]:
    audit = state["audit"]
    if audit.needs_revision or not audit.grounded or not audit.safe:
        return "revise"
    return "finalize"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("assess", assess_question)
    graph.add_node("retrieve", retrieve_knowledge)
    graph.add_node("retrieve_audit", audit_retrieval)
    graph.add_node("draft", draft_answer)
    graph.add_node("audit", audit_answer)
    graph.add_node("revise", revise_answer)
    graph.add_node("finalize", finalize_answer)

    graph.add_edge(START, "assess")
    graph.add_edge("assess", "retrieve")
    graph.add_edge("retrieve", "retrieve_audit")
    graph.add_conditional_edges(
        "retrieve_audit",
        route_after_retrieval,
        {"retrieve": "retrieve", "draft": "draft"},
    )
    graph.add_edge("draft", "audit")
    graph.add_conditional_edges("audit", route_after_answer_audit, {"revise": "revise", "finalize": "finalize"})
    graph.add_edge("revise", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()
