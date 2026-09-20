"""Deterministic 384-d embedder for unit tests (not a production model)."""
from __future__ import annotations

import re
import zlib
from typing import List, Sequence

import numpy as np

DIM = 384


class HashEmbedder:
    dim = DIM
    model_id = "hash-test"

    def _vec(self, text: str) -> bytes:
        tokens = re.findall(r"[a-z0-9]{2,}", (text or "").lower())
        acc = np.zeros(self.dim, dtype=np.float32)
        for tok in tokens:
            rng = np.random.RandomState(zlib.crc32(tok.encode("utf-8")) & 0xFFFFFFFF)
            acc += rng.randn(self.dim).astype(np.float32)
        n = float(np.linalg.norm(acc))
        if n == 0:
            acc[0] = 1.0
        else:
            acc /= n
        return acc.tobytes()

    def embed_documents(self, texts: Sequence[str]) -> List[bytes]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> bytes:
        return self._vec(text)
