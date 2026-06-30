from __future__ import annotations

from ped_agent.models.literature import LiteratureDocument


class CNKISource:
    def __init__(self, session_cookie: str | None = None):
        self.session_cookie = session_cookie

    def search(self, query: str, max_results: int = 20) -> list[LiteratureDocument]:
        raise NotImplementedError("CNKI integration requires a compliance review in Phase 3.")

