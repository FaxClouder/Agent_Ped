from __future__ import annotations


def expand_query(query: str, count: int = 3) -> list[str]:
    """Deterministic placeholder for multi-query expansion."""
    if count <= 1:
        return [query]
    return [query, f"{query} methodology", f"{query} evidence"][:count]

