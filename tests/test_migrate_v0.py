from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from phase1.migrate_from_v0 import migrate
from phase1.schema import connect


def test_phase0_migration_and_provenance(tmp_path, db_path):
    v0 = tmp_path / "decisions.sqlite"
    raw = sqlite3.connect(str(v0))
    raw.execute(
        """
        CREATE TABLE entries (
          id TEXT PRIMARY KEY,
          category TEXT NOT NULL,
          status TEXT NOT NULL,
          title TEXT NOT NULL,
          body TEXT NOT NULL,
          why_failed TEXT,
          supersedes_id TEXT,
          source TEXT NOT NULL,
          source_kind TEXT NOT NULL,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          ingested_from_call_id TEXT
        )
        """
    )
    now = datetime.now(timezone.utc).isoformat()
    raw.execute(
        """
        INSERT INTO entries VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            "e1",
            "decisions",
            "active",
            "v0 decision",
            "Use a local SQLite file for agent memory.",
            None,
            None,
            "backfill",
            "durable_md",
            now,
            now,
            "call-1",
        ),
    )
    raw.commit()
    raw.close()

    out = migrate(v0_path=v0, db_path=db_path)
    assert out["count_entries"] == 1
    conn = connect(db_path)
    row = conn.execute("SELECT * FROM memories WHERE phase0_entry_id='e1'").fetchone()
    assert row is not None
    assert row["memory_type"] == "decision"
    src = conn.execute("SELECT * FROM memory_sources WHERE memory_id=?", (row["id"],)).fetchall()
    assert src
    conn.close()
