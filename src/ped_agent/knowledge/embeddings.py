from __future__ import annotations

from typing import Any


class BGEM3Embeddings:
    """Lazy wrapper around FlagEmbedding.BGEM3FlagModel."""

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cpu", use_fp16: bool = False):
        self.model_name = model_name
        self.device = device
        self.use_fp16 = use_fp16
        self._model: Any | None = None

    @property
    def model(self):
        if self._model is None:
            try:
                from FlagEmbedding import BGEM3FlagModel
            except ImportError as exc:
                raise RuntimeError("Install ped-agent[rag] to use BGE-M3 embeddings.") from exc
            self._model = BGEM3FlagModel(self.model_name, use_fp16=self.use_fp16, device=self.device)
        return self._model

    def encode(self, texts: list[str], dense: bool = True, sparse: bool = True) -> dict:
        return self.model.encode(
            texts,
            return_dense=dense,
            return_sparse=sparse,
            return_colbert_vecs=False,
        )

