from __future__ import annotations

from ped_agent.models.literature import LiteratureDocument


class WanfangSource:
    def search(self, query: str, max_results: int = 20) -> list[LiteratureDocument]:
        raise NotImplementedError("Wanfang integration requires a compliance review in Phase 3.")

