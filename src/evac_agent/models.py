from __future__ import annotations

from typing import Literal, TypedDict

from pydantic import BaseModel, Field


RiskLevel = Literal["low", "medium", "high"]
IntentType = Literal["instructional", "scenario_analysis", "policy_knowledge", "real_time_emergency"]


class QueryAssessment(BaseModel):
    intent: IntentType = Field(description="User intent category.")
    risk_level: RiskLevel = Field(description="Risk level for evacuation answering.")
    building_type: str = Field(default="unknown", description="Building or venue type.")
    incident_type: str = Field(default="unknown", description="Type of evacuation-related incident.")
    vulnerable_groups: list[str] = Field(default_factory=list, description="Groups needing special assistance.")
    missing_information: list[str] = Field(default_factory=list, description="Critical missing context.")
    search_queries: list[str] = Field(default_factory=list, description="Expanded queries for retrieval.")
    response_mode: str = Field(default="balanced", description="Response mode such as conservative or balanced.")


class RetrievalAudit(BaseModel):
    sufficient: bool = Field(description="Whether retrieved evidence is sufficient for grounded answering.")
    issues: list[str] = Field(default_factory=list, description="Problems found in retrieval coverage/quality.")
    refined_queries: list[str] = Field(
        default_factory=list,
        description="Refined search queries for the next retrieval attempt when evidence is insufficient.",
    )


class AnswerAudit(BaseModel):
    grounded: bool = Field(description="Whether the answer is grounded in retrieved context.")
    safe: bool = Field(description="Whether the answer avoids unsafe or overconfident guidance.")
    needs_revision: bool = Field(description="Whether the draft should be revised.")
    issues: list[str] = Field(default_factory=list, description="Problems found during answer audit.")


class AgentState(TypedDict, total=False):
    question: str
    assessment: QueryAssessment
    active_queries: list[str]
    retrieval_attempt: int
    retrieved_context: str
    retrieved_sources: list[str]
    retrieval_audit: RetrievalAudit
    draft_answer: str
    audit: AnswerAudit
    final_answer: str
