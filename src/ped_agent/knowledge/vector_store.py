from __future__ import annotations

from ped_agent.knowledge.indexer import DocumentChunk


class InMemoryVectorStore:
    """Minimal placeholder store for scaffold tests and early development."""

    def __init__(self):
        self.documents: list[DocumentChunk] = []

    def add_documents(self, chunks: list[DocumentChunk]) -> list[str]:
        start = len(self.documents)
        self.documents.extend(chunks)
        return [f"doc-{idx}" for idx in range(start, start + len(chunks))]

    def search(self, query: str, top_k: int = 5):
        query_lower = query.lower()
        scored = []
        for chunk in self.documents:
            score = float(chunk.text.lower().count(query_lower)) if query_lower else 0.0
            if score:
                scored.append((score, chunk))
        return [chunk for _, chunk in sorted(scored, key=lambda item: item[0], reverse=True)[:top_k]]

