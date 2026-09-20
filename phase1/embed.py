"""Pluggable embedder — SQLite text is durable truth; embeddings regeneratable."""
from __future__ import annotations

from typing import List, Protocol, Sequence

DIM = 384
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class Embedder(Protocol):
    dim: int
    model_id: str

    def embed_documents(self, texts: Sequence[str]) -> List[bytes]: ...
    def embed_query(self, text: str) -> bytes: ...


class FastEmbedEmbedder:
    """Local fastembed BGE-small; replaceable without schema change."""

    def __init__(self, model_id: str = DEFAULT_MODEL):
        from fastembed import TextEmbedding
        import numpy as np

        self.model_id = model_id
        self.dim = DIM
        self._np = np
        self._model = TextEmbedding(model_name=model_id)

    def _to_blob(self, vec) -> bytes:
        arr = self._np.asarray(list(vec), dtype=self._np.float32)
        if arr.shape[0] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {arr.shape[0]}")
        return arr.tobytes()

    def embed_documents(self, texts: Sequence[str]) -> List[bytes]:
        out = []
        for v in self._model.embed(list(texts)):
            out.append(self._to_blob(v))
        return out

    def embed_query(self, text: str) -> bytes:
        for v in self._model.embed([text]):
            return self._to_blob(v)
        raise RuntimeError("embed_query produced no vector")


_default: Embedder | None = None


def get_embedder() -> Embedder:
    global _default
    if _default is None:
        _default = FastEmbedEmbedder()
    return _default


def set_embedder(embedder: Embedder | None) -> None:
    global _default
    _default = embedder
