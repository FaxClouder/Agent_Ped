from __future__ import annotations

from ped_agent.models.literature import LiteratureDocument


class ArxivSource:
    def search(self, query: str, max_results: int = 20) -> list[LiteratureDocument]:
        raise NotImplementedError("arXiv integration is planned for Phase 2.")

