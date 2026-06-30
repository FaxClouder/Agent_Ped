from __future__ import annotations

import numpy as np


def summarize_distribution(values: list[float] | np.ndarray) -> dict[str, float]:
    array = np.array(values, dtype=float)
    if array.size == 0:
        return {"count": 0.0, "mean": 0.0, "std": 0.0}
    return {
        "count": float(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "p15": float(np.percentile(array, 15)),
        "p50": float(np.percentile(array, 50)),
        "p85": float(np.percentile(array, 85)),
    }

