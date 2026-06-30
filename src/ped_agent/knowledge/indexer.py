from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DocumentChunk:
    text: str
    metadata: dict = field(default_factory=dict)


class Indexer:
    """Phase 2 will wire parsers, splitters, embeddings, and vector stores here."""

    def __init__(self, parser=None, splitter=None, vector_store=None):
        self.parser = parser
        self.splitter = splitter
        self.vector_store = vector_store

    def load_text_file(self, path: str | Path) -> list[DocumentChunk]:
        source = Path(path)
        text = source.read_text(encoding="utf-8")
        return [DocumentChunk(text=text, metadata={"source": str(source)})]

    def index_chunks(self, chunks: list[DocumentChunk]) -> list[str]:
        if self.vector_store is None:
            return [f"chunk-{idx}" for idx, _ in enumerate(chunks)]
        return self.vector_store.add_documents(chunks)

