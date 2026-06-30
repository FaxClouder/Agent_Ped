from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class RetrievedDocument:
    text: str
    score: float = 0.0
    metadata: dict = field(default_factory=dict)


class RetrieverBackend(Protocol):
    def search(self, query: str, top_k: int) -> list[RetrievedDocument]: ...


class HybridRetriever:
    def __init__(self, dense_backend: RetrieverBackend | None = None, sparse_backend=None):
        self.dense_backend = dense_backend
        self.sparse_backend = sparse_backend

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDocument]:
        if self.dense_backend is None and self.sparse_backend is None:
            return []
        candidates: list[RetrievedDocument] = []
        if self.dense_backend is not None:
            candidates.extend(self.dense_backend.search(query, top_k))
        if self.sparse_backend is not None:
            candidates.extend(self.sparse_backend.search(query, top_k))
        return sorted(candidates, key=lambda item: item.score, reverse=True)[:top_k]

    async def ainvoke(self, query: str) -> list[RetrievedDocument]:
        return self.retrieve(query)

