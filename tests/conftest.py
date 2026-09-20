from __future__ import annotations

from pathlib import Path

import pytest

from phase1.embed import set_embedder
from phase1.schema import connect, ensure_schema
from tests.fakes import HashEmbedder


@pytest.fixture
def db_path(tmp_path: Path):
    set_embedder(HashEmbedder())
    path = tmp_path / "agent_memory.sqlite"
    ensure_schema(connect(path))
    yield path
    set_embedder(None)


@pytest.fixture
def conn(db_path):
    c = connect(db_path)
    try:
        yield c
    finally:
        c.close()
