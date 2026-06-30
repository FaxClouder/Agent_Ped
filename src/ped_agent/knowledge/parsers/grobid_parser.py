from __future__ import annotations

from pathlib import Path


class GrobidParser:
    def __init__(self, grobid_url: str = "http://localhost:8070"):
        self.grobid_url = grobid_url

    def extract_metadata(self, pdf_path: str | Path) -> dict:
        raise NotImplementedError("GROBID metadata extraction is planned for Phase 2.")

