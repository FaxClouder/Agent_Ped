from __future__ import annotations

from pydantic import BaseModel, Field


class Citation(BaseModel):
    citation_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    url: str | None = None


class LiteratureDocument(BaseModel):
    document_id: str
    title: str
    abstract: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    source: str | None = None
    doi: str | None = None
    url: str | None = None
    full_text_path: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)

