from __future__ import annotations

from ped_agent.knowledge.retriever import RetrievedDocument


def reciprocal_rank_fusion(
    rankings: list[list[RetrievedDocument]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[RetrievedDocument]:
    weights = weights or [1.0] * len(rankings)
    scores: dict[str, float] = {}
    documents: dict[str, RetrievedDocument] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, document in enumerate(ranking, start=1):
            key = document.metadata.get("id") or document.text
            scores[key] = scores.get(key, 0.0) + weight / (k + rank)
            documents[key] = document
    fused = [
        RetrievedDocument(text=documents[key].text, score=score, metadata=documents[key].metadata)
        for key, score in scores.items()
    ]
    return sorted(fused, key=lambda item: item.score, reverse=True)

