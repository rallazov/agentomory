from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from phase1.schema import (
    SCHEMA_VERSION,
    connect,
    ensure_schema,
    migrate_schema,
    table_columns,
)


def test_schema_creation(db_path):
    conn = connect(db_path)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for name in (
        "conversations",
        "messages",
        "memories",
        "projects",
        "entities",
        "artifacts",
        "memory_links",
        "memory_sources",
        "memory_entities",
        "vocabulary",
        "retrieval_log",
        "schema_migrations",
        "memories_vec",
        "extraction_progress",
        "extracted_messages",
    ):
        assert name in tables
    fts = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='memories_fts'"
    ).fetchone()
    assert fts
    mem_cols = table_columns(conn, "memories")
    for col in (
        "remote_ok",
        "local_only",
        "sensitive",
        "never_auto_retrieve",
        "never_send_external_model",
        "temporal_meaning",
        "appears_to_correct",
    ):
        assert col in mem_cols
    log_cols = table_columns(conn, "retrieval_log")
    for col in ("request_id", "turn_id", "model_used", "user_accepted", "feedback_json"):
        assert col in log_cols
    ver = conn.execute("SELECT version FROM schema_migrations").fetchone()
    assert ver["version"] == SCHEMA_VERSION
    conn.close()


def test_schema_migration_preserves_existing_rows(tmp_path):
    """Old Phase 1 DB (pre-privacy / memory_sources) upgrades in place."""
    path = tmp_path / "legacy.sqlite"
    raw = sqlite3.connect(str(path))
    raw.execute(
        """
        CREATE TABLE memories (
          id TEXT PRIMARY KEY,
          memory_type TEXT NOT NULL,
          status TEXT NOT NULL,
          title TEXT NOT NULL,
          canonical_text TEXT NOT NULL,
          importance REAL NOT NULL DEFAULT 0.5,
          confidence REAL NOT NULL DEFAULT 0.5,
          origin TEXT NOT NULL,
          project_id TEXT,
          supersedes_id TEXT,
          why_failed TEXT,
          provenance_conversation_id TEXT,
          provenance_message_ids TEXT,
          provenance_artifact_id TEXT,
          provenance_note TEXT,
          source_kind TEXT,
          phase0_entry_id TEXT,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL
        )
        """
    )
    raw.execute(
        """
        CREATE TABLE retrieval_log (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          ts TEXT NOT NULL,
          query TEXT NOT NULL,
          expanded_queries_json TEXT,
          candidate_ids_json TEXT,
          scores_json TEXT,
          injected_ids_json TEXT,
          approx_token_cost INTEGER,
          project_id TEXT,
          context_json TEXT,
          response_note TEXT,
          correction_note TEXT
        )
        """
    )
    now = datetime.now(timezone.utc).isoformat()
    raw.execute(
        """
        INSERT INTO memories (
          id, memory_type, status, title, canonical_text, importance, confidence,
          origin, provenance_conversation_id, provenance_message_ids,
          provenance_note, source_kind, created_at, updated_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            "mem-legacy-1",
            "decision",
            "active",
            "Legacy",
            "Keep the legacy row.",
            0.7,
            0.8,
            "user_stated",
            "conv-1",
            '["msg-1"]',
            "from old db",
            "voice_call_message",
            now,
            now,
        ),
    )
    raw.commit()
    raw.close()

    conn = ensure_schema(connect(path))
    row = conn.execute("SELECT * FROM memories WHERE id='mem-legacy-1'").fetchone()
    assert row["canonical_text"] == "Keep the legacy row."
    assert row["remote_ok"] == 1
    assert row["never_auto_retrieve"] == 0
    src = conn.execute("SELECT * FROM memory_sources WHERE memory_id='mem-legacy-1'").fetchall()
    assert src
    assert src[0]["message_id"] == "msg-1"
    assert src[0]["conversation_id"] == "conv-1"
    log_cols = table_columns(conn, "retrieval_log")
    assert "request_id" in log_cols
    migrate_schema(conn)
    still = conn.execute("SELECT COUNT(*) c FROM memories").fetchone()["c"]
    assert still == 1
    conn.close()


def test_memory_insertion(db_path):
    from phase1.consolidate import apply_memory

    out = apply_memory(
        {
            "id": "mem-insert-1",
            "memory_type": "fact",
            "title": "Inserted",
            "canonical_text": "The package name is agentomory.",
            "importance": 0.6,
            "confidence": 0.8,
            "origin": "user_stated",
        },
        db_path=db_path,
    )
    assert out["action"] == "CREATE"
    conn = connect(db_path)
    row = conn.execute("SELECT * FROM memories WHERE id=?", (out["id"],)).fetchone()
    assert row["canonical_text"].startswith("The package name")
    fts = conn.execute(
        "SELECT memory_id FROM memories_fts WHERE memories_fts MATCH 'agentomory'"
    ).fetchall()
    assert fts
    vec = conn.execute("SELECT memory_id FROM memories_vec WHERE memory_id=?", (out["id"],)).fetchone()
    assert vec
    conn.close()
