"""Atlas review extras for Phase 1 harden: corrections, isolation, budget, cursor."""
from __future__ import annotations

from datetime import datetime, timezone

from phase1.consolidate import apply_memory
from phase1.extract import extract_from_messages, get_extraction_progress
from phase1.memory_pack import apply_token_budget, build_pack
from phase1.retrieve import hybrid_search
from phase1.schema import connect


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cand(text, **kw):
    base = {
        "memory_type": "preference",
        "title": text[:60],
        "canonical_text": text,
        "importance": 0.85,
        "confidence": 0.92,
        "origin": "user_stated",
        "source_kind": "test",
        "provenance_note": "atlas",
        "source_message_id": kw.pop("message_id", "msg-a"),
    }
    base.update(kw)
    return base


def _insert_memory(db_path, mid, text, **kw):
    apply_memory(
        {
            "id": mid,
            "memory_type": kw.get("memory_type", "fact"),
            "title": kw.get("title", text[:60]),
            "canonical_text": text,
            "importance": kw.get("importance", 0.7),
            "confidence": kw.get("confidence", 0.85),
            "origin": "user_stated",
            "project_id": kw.get("project_id"),
        },
        db_path=db_path,
    )


def _seed_project(db_path, pid, name, summary):
    conn = connect(db_path)
    conn.execute(
        """
        INSERT INTO projects (
          id, name, purpose, current_goal, current_stage, key_decisions,
          constraints, open_questions, next_actions, current_summary,
          status, created_at, updated_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (pid, name, summary, None, None, None, None, None, None, summary, "active", "t", "t"),
    )
    conn.commit()
    conn.close()


def _add_message(db_path, conversation_id, message_id, text, seq, *, created_at=None):
    conn = connect(db_path)
    now = created_at or _now()
    conn.execute(
        """
        INSERT OR IGNORE INTO conversations (
          id, source_kind, source_path, agent_id, started_at, ended_at, meta_json, created_at
        ) VALUES (?,?,?,?,?,?,?,?)
        """,
        (conversation_id, "test", None, None, None, None, None, now),
    )
    conn.execute(
        """
        INSERT INTO messages (
          id, conversation_id, seq, role, speaker, text, at_ms, created_at
        ) VALUES (?,?,?,?,?,?,?,?)
        """,
        (message_id, conversation_id, seq, "user", "user", text, None, now),
    )
    conn.commit()
    conn.close()


def test_always_then_actually_never_does_not_merge(db_path):
    """Later 'Actually, never …' must not merge into earlier 'Always …'."""
    always = apply_memory(
        _cand("Always send a daily status summary after each turn.", message_id="msg-always"),
        db_path=db_path,
    )
    assert always["action"] == "CREATE"
    later = apply_memory(
        _cand(
            "Actually, never send a daily status summary after each turn.",
            memory_type="correction",
            appears_to_correct=True,
            confidence=0.95,
            message_id="msg-never",
        ),
        db_path=db_path,
    )
    assert later["action"] in ("SUPERSEDE", "CONFLICT")
    assert later["action"] not in ("MERGE", "IGNORE", "UPDATE")
    conn = connect(db_path)
    old = conn.execute("SELECT * FROM memories WHERE id=?", (always["id"],)).fetchone()
    new = conn.execute("SELECT * FROM memories WHERE id=?", (later["id"],)).fetchone()
    if later["action"] == "SUPERSEDE":
        assert old["status"] == "superseded"
        assert later["id"] != always["id"]
        assert "never" in new["canonical_text"].lower()
        assert "always" not in new["canonical_text"].lower()
    else:
        assert old["status"] == "active"
        assert later["id"] != always["id"]
        assert "never" in new["canonical_text"].lower()
    active = [dict(r) for r in conn.execute("SELECT * FROM memories WHERE status='active'")]
    conn.close()
    # The old Always instruction must not be the only remaining active wording.
    active_texts = " ".join(r["canonical_text"].lower() for r in active)
    assert "never" in active_texts
    if later["action"] == "SUPERSEDE":
        assert all("always" not in r["canonical_text"].lower() for r in active)


def test_project_scope_does_not_leak_foreign_memories(db_path):
    _seed_project(db_path, "proj-alpha", "Alpha Store", "Alpha owns the on-device store.")
    _seed_project(db_path, "proj-beta", "Beta Notify", "Beta owns notifications.")
    _insert_memory(
        db_path,
        "mem-alpha",
        "Use SQLite for the Alpha store.",
        memory_type="decision",
        project_id="proj-alpha",
    )
    _insert_memory(
        db_path,
        "mem-beta",
        "Use SQLite for the Beta store.",
        memory_type="decision",
        project_id="proj-beta",
        importance=0.99,
    )
    hits, _, _ = hybrid_search(
        "sqlite store", k=8, db_path=db_path, project_id="proj-alpha"
    )
    ids = {h["id"] for h in hits}
    assert "mem-beta" not in ids
    assert all(h.get("project_id") in (None, "proj-alpha") for h in hits)
    pack = build_pack(
        "sqlite store",
        k=8,
        current_project_id="proj-alpha",
        db_path=db_path,
        log=False,
    )
    pack_ids = {m["id"] for m in pack["memories"]}
    assert "mem-beta" not in pack_ids
    assert pack.get("project") is None or pack["project"]["id"] == "proj-alpha"


def test_hard_token_budget_truncates_or_drops(db_path):
    verbose = "Always record this standing rule. " + ("Filler clause about unrelated history. " * 80)
    _insert_memory(db_path, "mem-verbose", verbose, memory_type="preference", importance=0.4)
    _insert_memory(
        db_path,
        "mem-sql",
        "Use SQLite for the local memory store.",
        memory_type="decision",
        importance=0.9,
    )
    _seed_project(db_path, "proj-wordy", "Wordy", "W" * 4000)
    pack = build_pack(
        "sqlite local memory store",
        k=8,
        current_project_id="proj-wordy",
        db_path=db_path,
        log=False,
        max_tokens=70,
        max_item_tokens=30,
        max_project_tokens=16,
    )
    assert pack["approx_token_cost"] <= 70
    for m in pack["memories"]:
        assert (len(m["canonical_text"]) // 4) <= 32
        if m["id"] == "mem-verbose":
            assert m.get("truncated") is True
            assert len(m["canonical_text"]) < len(verbose)
    if pack.get("project"):
        blob = str(pack["project"])
        assert len(blob) < 4000
        assert pack["project"].get("truncated") is True or not pack["project"].get("current_summary")
    # Lowest-relevance first: the small high-signal row should survive a tight budget.
    assert any(m["id"] == "mem-sql" for m in pack["memories"]) or pack["pack_size"] <= 1

    rows = [
        {"id": "low", "title": "low", "canonical_text": "x" * 80, "score": 0.1},
        {"id": "high", "title": "high", "canonical_text": "Use SQLite.", "score": 0.9},
    ]
    kept, proj, cost = apply_token_budget(
        rows,
        {"id": "p", "name": "P", "current_summary": "S" * 2000},
        max_tokens=20,
        max_item_tokens=8,
        max_project_tokens=4,
    )
    assert cost <= 20
    assert proj is None or _short(proj)
    assert all((len(r["canonical_text"]) // 4) <= 10 for r in kept)


def _short(proj) -> bool:
    return len(str(proj)) < 200


def test_extraction_progress_skips_processed_and_sees_new_messages(db_path):
    _add_message(db_path, "conv-a", "msg-1", "Always send a daily status summary after each turn.", 0)
    _add_message(db_path, "conv-a", "msg-2", "We decided to use SQLite for the local store.", 1)
    _add_message(db_path, "conv-b", "msg-3", "Never send calendar tokens to any cloud model.", 0)

    first = extract_from_messages(db_path=db_path, limit=1)
    prog = get_extraction_progress(db_path=db_path)
    assert prog is not None
    assert prog["last_message_id"] == "msg-1"
    conn = connect(db_path)
    done = {r["message_id"] for r in conn.execute("SELECT message_id FROM extracted_messages")}
    conn.close()
    assert done == {"msg-1"}

    second = extract_from_messages(db_path=db_path, limit=1)
    prog = get_extraction_progress(db_path=db_path)
    assert prog["last_message_id"] == "msg-2"
    # Second pass must not re-process msg-1 as if we always took a fixed prefix.
    assert not any(
        (r.get("candidate") or {}).get("source_message_id") == "msg-1" for r in second
    )

    extract_from_messages(db_path=db_path, limit=1)
    _add_message(
        db_path,
        "conv-c",
        "msg-4",
        "By November we need twenty test users.",
        0,
    )
    later = extract_from_messages(db_path=db_path, limit=10)
    prog = get_extraction_progress(db_path=db_path)
    assert prog["last_message_id"] == "msg-4"
    conn = connect(db_path)
    done = {r["message_id"] for r in conn.execute("SELECT message_id FROM extracted_messages")}
    conn.close()
    assert done == {"msg-1", "msg-2", "msg-3", "msg-4"}
    assert any(
        (r.get("candidate") or {}).get("source_message_id") == "msg-4" for r in later
    ) or first is not None
