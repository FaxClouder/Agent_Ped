from __future__ import annotations

from pathlib import Path


class CajConverter:
    def convert_to_pdf(self, caj_path: str | Path, output_path: str | Path | None = None) -> Path:
        raise NotImplementedError("CAJ conversion is planned for Phase 3.")

