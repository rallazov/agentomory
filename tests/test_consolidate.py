from __future__ import annotations

from phase1.consolidate import apply_memory
from phase1.provenance import list_sources
from phase1.schema import connect


def _cand(text, **kw):
    base = {
        "memory_type": "decision",
        "title": text[:60],
        "canonical_text": text,
        "importance": 0.8,
        "confidence": 0.9,
        "origin": "user_stated",
        "source_kind": "test",
        "provenance_note": kw.pop("note", "n1"),
        "provenance_conversation_id": kw.pop("conversation_id", "c1"),
        "source_message_id": kw.pop("message_id", "msg-a"),
    }
    base.update(kw)
    return base


def test_duplicate_detection(db_path):
    a = apply_memory(_cand("Use SQLite for the local memory store."), db_path=db_path)
    b = apply_memory(_cand("Use SQLite for the local memory store."), db_path=db_path)
    assert a["action"] == "CREATE"
    assert b["action"] in ("IGNORE", "MERGE", "UPDATE")
    assert b["id"] == a["id"]
    conn = connect(db_path)
    n = conn.execute("SELECT COUNT(*) c FROM memories WHERE status='active'").fetchone()["c"]
    conn.close()
    assert n == 1


def test_semantic_paraphrase_merging(db_path):
    a = apply_memory(_cand("We locked SQLite as the local memory store."), db_path=db_path)
    b = apply_memory(_cand("The local memory store is SQLite."), db_path=db_path)
    assert a["action"] == "CREATE"
    assert b["action"] in ("MERGE", "IGNORE", "UPDATE")
    assert b["id"] == a["id"]


def test_similar_but_different_decisions_preserved(db_path):
    a = apply_memory(_cand("Ship the Alpha Store beta to testers on Friday."), db_path=db_path)
    b = apply_memory(_cand("Ship the Alpha Store beta to testers on Monday."), db_path=db_path)
    assert a["action"] == "CREATE"
    assert b["action"] in ("CREATE", "CONFLICT")
    assert b["id"] != a["id"]
    if b["action"] == "CONFLICT":
        assert b.get("possible_conflict") == a["id"]
    conn = connect(db_path)
    n = conn.execute("SELECT COUNT(*) c FROM memories WHERE status='active'").fetchone()["c"]
    conn.close()
    assert n == 2


def test_correction_supersession_is_conservative(db_path):
    a = apply_memory(_cand("Use PostgreSQL for the local memory store."), db_path=db_path)
    # Uncertain correction (low confidence) must not destroy current truth.
    weak = apply_memory(
        _cand(
            "That's wrong — use SQLite for the local memory store.",
            memory_type="correction",
            appears_to_correct=True,
            confidence=0.5,
            message_id="msg-weak",
        ),
        db_path=db_path,
    )
    assert weak["action"] in ("CREATE", "CONFLICT")
    conn = connect(db_path)
    old = conn.execute("SELECT status FROM memories WHERE id=?", (a["id"],)).fetchone()
    assert old["status"] == "active"
    conn.close()

    strong = apply_memory(
        _cand(
            "That's wrong — use SQLite for the local memory store.",
            memory_type="correction",
            appears_to_correct=True,
            confidence=0.95,
            message_id="msg-strong",
        ),
        db_path=db_path,
    )
    # Highly confident correction of the same store decision may supersede.
    assert strong["action"] in ("SUPERSEDE", "CONFLICT", "CREATE", "MERGE")
    if strong["action"] == "SUPERSEDE":
        conn = connect(db_path)
        old = conn.execute("SELECT status FROM memories WHERE id=?", (a["id"],)).fetchone()
        assert old["status"] == "superseded"
        conn.close()


def test_merge_keeps_prior_provenance(db_path):
    a = apply_memory(
        _cand("Use SQLite for the local memory store.", message_id="msg-old", note="first"),
        db_path=db_path,
    )
    apply_memory(
        _cand("The local memory store is SQLite.", message_id="msg-new", note="second", conversation_id="c2"),
        db_path=db_path,
    )
    sources = list_sources(a["id"], db_path=db_path)
    mids = {s["message_id"] for s in sources if s["message_id"]}
    assert "msg-old" in mids
    assert "msg-new" in mids
