from __future__ import annotations

from pathlib import Path

from phase1.consolidate import apply_memory
from phase1.memory_pack import build_pack
from phase1.project_resolve import resolve_project
from phase1.retrieve import expand_query, hybrid_search
from phase1.schema import connect
from phase1.vocabulary import upsert_term


def _insert(db_path, mid, text, **kw):
    apply_memory(
        {
            "id": mid,
            "memory_type": kw.get("memory_type", "fact"),
            "title": kw.get("title", text[:60]),
            "canonical_text": text,
            "importance": kw.get("importance", 0.7),
            "confidence": kw.get("confidence", 0.85),
            "origin": kw.get("origin", "user_stated"),
            "project_id": kw.get("project_id"),
            "remote_ok": kw.get("remote_ok", 1),
            "local_only": kw.get("local_only", 0),
            "sensitive": kw.get("sensitive", 0),
            "never_auto_retrieve": kw.get("never_auto_retrieve", 0),
            "never_send_external_model": kw.get("never_send_external_model", 0),
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


def test_fts_and_vector_and_hybrid(db_path):
    _insert(db_path, "mem-sql", "Use SQLite for the local memory store.", memory_type="decision")
    _insert(db_path, "mem-quiet", "Do not send notifications after 21:00.", memory_type="preference")
    hits, expanded, scores = hybrid_search("sqlite local memory store", k=8, db_path=db_path)
    assert hits
    assert hits[0]["id"] == "mem-sql"
    assert any(h["id"] == "mem-sql" for h in hits)
    pack = build_pack("sqlite local memory store", k=8, db_path=db_path, log=False)
    assert pack["memories"]
    assert pack["memories"][0]["id"] == "mem-sql"


def test_zero_result_retrieval(db_path):
    _insert(db_path, "mem-sql", "Use SQLite for the local memory store.", memory_type="decision")
    hits, _, _ = hybrid_search(
        "purple helicopter warranty claim for mars potatoes", k=8, db_path=db_path
    )
    assert hits == []
    pack = build_pack(
        "purple helicopter warranty claim for mars potatoes", k=8, db_path=db_path, log=False
    )
    assert pack["pack_size"] == 0
    assert pack["memories"] == []


def test_pack_does_not_force_quota(db_path):
    _insert(db_path, "mem-sql", "Use SQLite for the local memory store.", memory_type="decision")
    pack = build_pack("sqlite local memory store", k=8, db_path=db_path, log=False)
    assert 1 <= pack["pack_size"] <= 8
    # A single relevant memory must not be padded to 3–8.
    assert pack["pack_size"] < 3 or all(
        "sqlite" in m["canonical_text"].lower() or "memory" in m["canonical_text"].lower()
        for m in pack["memories"]
    )


def test_no_hardcoded_project_aliases_in_core_retrieval():
    forbidden = (
        "calendar",
        "cloud agent",
        "sprint 1",
        "nylas",
        "story-bound",
        "application-tracker",
        "outer-loop",
    )
    for rel in ("phase1/retrieve.py", "phase1/memory_pack.py"):
        src = Path(__file__).resolve().parents[1] / rel
        text = src.read_text(encoding="utf-8").lower()
        for token in forbidden:
            assert token not in text, f"{rel} must not contain {token!r}"


def test_query_expansion_uses_sqlite_vocabulary(db_path):
    _seed_project(db_path, "proj-x", "Xenon Pipeline", "Xenon owns the pipeline.")
    upsert_term(
        term="xe pipe",
        kind="project_alias",
        project_id="proj-x",
        canonical="Xenon Pipeline",
        db_path=db_path,
    )
    expanded = expand_query("please check xe pipe status", db_path=db_path)
    joined = " ".join(expanded).lower()
    assert "xenon pipeline" in joined


def test_project_resolution_explicit_and_uncertain(db_path):
    _seed_project(db_path, "proj-notes", "Notes", "Contains many notes about the weekly review.")
    _seed_project(db_path, "proj-store", "Store", "On-device store.")
    # Must not pick Notes just because the query token "notes" appears in the summary.
    res = resolve_project("please review the notes from lunch", db_path=db_path)
    assert res.project_id is None or res.confidence < 0.58

    explicit = resolve_project("anything", current_project_id="proj-store", db_path=db_path)
    assert explicit.project_id == "proj-store"
    assert explicit.method == "explicit"
    assert explicit.confidence == 1.0

    upsert_term(
        term="on-device store",
        kind="project_alias",
        project_id="proj-store",
        canonical="Store",
        db_path=db_path,
    )
    aliased = resolve_project("on-device store decision", db_path=db_path)
    assert aliased.project_id == "proj-store"
    assert aliased.method in ("alias", "name", "semantic")


def test_privacy_filtering(db_path):
    _insert(
        db_path,
        "mem-secret",
        "The operator's home address is 1 Hidden Lane.",
        memory_type="fact",
        local_only=1,
        never_send_external_model=1,
        sensitive=1,
        remote_ok=0,
    )
    _insert(
        db_path,
        "mem-public",
        "Use SQLite for the local memory store.",
        memory_type="decision",
    )
    _insert(
        db_path,
        "mem-silent",
        "Never auto retrieve this draft password hint.",
        memory_type="constraint",
        never_auto_retrieve=1,
    )
    local = hybrid_search("hidden lane address", k=8, db_path=db_path, destination="local")
    ext = hybrid_search("hidden lane address", k=8, db_path=db_path, destination="external_model")
    local_ids = {h["id"] for h in local[0]}
    ext_ids = {h["id"] for h in ext[0]}
    assert "mem-silent" not in local_ids
    assert "mem-silent" not in ext_ids
    assert "mem-secret" not in ext_ids
    pack = build_pack(
        "sqlite local memory store",
        k=8,
        db_path=db_path,
        destination="external_model",
        log=False,
    )
    assert all(m["id"] != "mem-secret" for m in pack["memories"])


def test_retrieval_log_request_turn_and_feedback(db_path):
    from phase1.retrieval_log import attach_retrieval_feedback

    _insert(db_path, "mem-sql", "Use SQLite for the local memory store.")
    pack = build_pack(
        "sqlite local memory store",
        k=4,
        db_path=db_path,
        request_id="req-1",
        turn_id="turn-9",
        model_used="test-model",
    )
    assert pack["request_id"] == "req-1"
    conn = connect(db_path)
    row = conn.execute("SELECT * FROM retrieval_log WHERE request_id='req-1'").fetchone()
    assert row["turn_id"] == "turn-9"
    assert row["model_used"] == "test-model"
    conn.close()
    n = attach_retrieval_feedback(
        request_id="req-1",
        user_accepted=True,
        user_corrected=False,
        helpfulness="helpful",
        task_outcome="done",
        extra={"note": "ok"},
        db_path=db_path,
    )
    assert n == 1
    conn = connect(db_path)
    row = conn.execute("SELECT * FROM retrieval_log WHERE request_id='req-1'").fetchone()
    assert row["user_accepted"] == 1
    assert row["helpfulness"] == "helpful"
    assert row["task_outcome"] == "done"
    conn.close()
