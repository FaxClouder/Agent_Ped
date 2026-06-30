from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CitationEdge:
    source_id: str
    target_id: str


class CitationGraphStore:
    def add_citation(self, edge: CitationEdge) -> None:
        raise NotImplementedError("Neo4j citation graph integration is planned for Phase 2/3.")

