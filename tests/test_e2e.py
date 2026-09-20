"""End-to-end path with installed fastembed + sqlite-vec (skipped if unavailable)."""
from __future__ import annotations

from pathlib import Path

import pytest

from phase1.consolidate import apply_memory
from phase1.embed import FastEmbedEmbedder, set_embedder
from phase1.extract import extract_candidates
from phase1.memory_pack import build_pack
from phase1.retrieve import hybrid_search
from phase1.schema import connect, ensure_schema


def _real_embedder():
    try:
        import fastembed  # noqa: F401
        import sqlite_vec  # noqa: F401
    except Exception as exc:
        pytest.skip(f"fastembed/sqlite-vec unavailable: {exc}")
    try:
        return FastEmbedEmbedder()
    except Exception as exc:
        pytest.skip(f"could not load FastEmbed model: {exc}")


@pytest.fixture
def e2e_db(tmp_path: Path):
    emb = _real_embedder()
    set_embedder(emb)
    path = tmp_path / "e2e.sqlite"
    ensure_schema(connect(path))
    yield path
    set_embedder(None)


def test_e2e_extract_consolidate_retrieve_pack(e2e_db):
    text = (
        "We decided to use SQLite for the local store. "
        "Never send calendar tokens to any cloud model. "
        "By November we need twenty test users."
    )
    cands = extract_candidates(text, role="user", message_id="e2e-msg", conversation_id="e2e-c")
    assert len(cands) >= 2
    ids = []
    for c in cands:
        out = apply_memory(c, db_path=e2e_db)
        ids.append(out["id"])

    # Paraphrase of the store decision must merge, not create a second truth.
    para = apply_memory(
        {
            "memory_type": "decision",
            "title": "On-device store",
            "canonical_text": "The on-device memory database will be SQLite.",
            "importance": 0.8,
            "confidence": 0.9,
            "origin": "user_stated",
            "source_message_id": "e2e-msg-2",
            "source_kind": "test",
        },
        db_path=e2e_db,
    )
    assert para["action"] in ("MERGE", "IGNORE", "UPDATE")

    hits, _, _ = hybrid_search("which database holds local agent memory", k=8, db_path=e2e_db)
    assert hits
    assert any("sqlite" in h["canonical_text"].lower() for h in hits)

    pack = build_pack(
        "which database holds local agent memory",
        k=8,
        db_path=e2e_db,
        request_id="e2e-req",
        turn_id="e2e-turn",
        destination="external_model",
    )
    assert pack["pack_size"] >= 1
    assert pack["pack_size"] <= 8
    assert any("sqlite" in m["canonical_text"].lower() for m in pack["memories"])

    empty = build_pack(
        "purple helicopter warranty claim for mars potatoes",
        k=8,
        db_path=e2e_db,
        log=False,
    )
    assert empty["pack_size"] == 0

    conn = connect(e2e_db)
    n_src = conn.execute("SELECT COUNT(*) c FROM memory_sources").fetchone()["c"]
    n_vec = conn.execute("SELECT COUNT(*) c FROM memories_vec").fetchone()["c"]
    conn.close()
    assert n_src >= 1
    assert n_vec >= 1
