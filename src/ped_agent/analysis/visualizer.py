from __future__ import annotations

from pathlib import Path
from typing import Any

from ped_agent.analysis.schemas import AnalysisResult


class Visualizer:
    """Placeholder visualizer; real Plotly/Matplotlib outputs land here in Phase 4."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.output_dir = Path(self.config.get("output_dir", "./outputs/figures"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all(self, result: AnalysisResult, scenario: Any) -> list[str]:
        return []

