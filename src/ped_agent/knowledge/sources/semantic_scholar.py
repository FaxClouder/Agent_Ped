from __future__ import annotations

from ped_agent.models.literature import LiteratureDocument


class SemanticScholarSource:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key

    def search(self, query: str, max_results: int = 20) -> list[LiteratureDocument]:
        raise NotImplementedError("Semantic Scholar API integration is planned for Phase 2.")

