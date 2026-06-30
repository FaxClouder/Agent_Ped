from __future__ import annotations

from pathlib import Path


class MarkerParser:
    def parse(self, pdf_path: str | Path) -> str:
        raise NotImplementedError("Install and wire Marker in Phase 2.")

