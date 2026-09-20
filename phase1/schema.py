"""Phase 1 schema for agent_memory.sqlite (version 2: privacy, sources, vocabulary)."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import sqlite_vec

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "agent_memory.sqlite"
EMBED_DIM = 384
SCHEMA_VERSION = 2

MEMORY_TYPES = (
    "fact",
    "preference",
    "goal",
    "project_state",
    "decision",
    "constraint",
    "task",
    "open_question",
    "correction",
    "outcome",
    "lesson",
    "strategy",
)

STATUSES = ("active", "superseded", "reversed", "merged")
ORIGINS = ("user_stated", "durable_record", "agent_inferred", "phase0_import")
PRIVACY_FLAGS = (
    "remote_ok",
    "local_only",
    "sensitive",
    "never_auto_retrieve",
    "never_send_external_model",
)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS conversations (
  id TEXT PRIMARY KEY,
  source_kind TEXT NOT NULL,
  source_path TEXT,
  agent_id TEXT,
  started_at TEXT,
  ended_at TEXT,
  meta_json TEXT,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
  id TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL REFERENCES conversations(id),
  seq INTEGER NOT NULL,
  role TEXT NOT NULL CHECK (role IN ('user','assistant','system','tool')),
  speaker TEXT,
  text TEXT NOT NULL,
  at_ms INTEGER,
  created_at TEXT NOT NULL,
  UNIQUE(conversation_id, seq)
);
-- messages are immutable: no UPDATE triggers; app code must not rewrite text.

CREATE TABLE IF NOT EXISTS projects (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  purpose TEXT,
  current_goal TEXT,
  current_stage TEXT,
  key_decisions TEXT,
  constraints TEXT,
  open_questions TEXT,
  next_actions TEXT,
  current_summary TEXT,
  status TEXT NOT NULL DEFAULT 'active',
  source_artifact_id TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS entities (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  entity_type TEXT NOT NULL,
  meta_json TEXT,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS artifacts (
  id TEXT PRIMARY KEY,
  path TEXT NOT NULL UNIQUE,
  kind TEXT NOT NULL,
  title TEXT,
  content_hash TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memories (
  id TEXT PRIMARY KEY,
  memory_type TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('active','superseded','reversed','merged')),
  title TEXT NOT NULL,
  canonical_text TEXT NOT NULL,
  importance REAL NOT NULL DEFAULT 0.5,
  confidence REAL NOT NULL DEFAULT 0.5,
  origin TEXT NOT NULL CHECK (origin IN ('user_stated','durable_record','agent_inferred','phase0_import')),
  project_id TEXT REFERENCES projects(id),
  supersedes_id TEXT,
  why_failed TEXT,
  provenance_conversation_id TEXT,
  provenance_message_ids TEXT,
  provenance_artifact_id TEXT,
  provenance_note TEXT,
  source_kind TEXT,
  phase0_entry_id TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  remote_ok INTEGER NOT NULL DEFAULT 1,
  local_only INTEGER NOT NULL DEFAULT 0,
  sensitive INTEGER NOT NULL DEFAULT 0,
  never_auto_retrieve INTEGER NOT NULL DEFAULT 0,
  never_send_external_model INTEGER NOT NULL DEFAULT 0,
  temporal_meaning TEXT,
  appears_to_correct INTEGER NOT NULL DEFAULT 0,
  CHECK (status != 'reversed' OR why_failed IS NOT NULL)
);

CREATE TABLE IF NOT EXISTS memory_links (
  id TEXT PRIMARY KEY,
  from_memory_id TEXT NOT NULL REFERENCES memories(id),
  to_memory_id TEXT NOT NULL REFERENCES memories(id),
  relationship TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_sources (
  id TEXT PRIMARY KEY,
  memory_id TEXT NOT NULL REFERENCES memories(id),
  source_kind TEXT NOT NULL,
  conversation_id TEXT,
  message_id TEXT,
  artifact_id TEXT,
  note TEXT,
  role TEXT NOT NULL DEFAULT 'supporting',
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_entities (
  memory_id TEXT NOT NULL REFERENCES memories(id),
  entity_id TEXT NOT NULL REFERENCES entities(id),
  PRIMARY KEY (memory_id, entity_id)
);

CREATE TABLE IF NOT EXISTS vocabulary (
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  term TEXT NOT NULL,
  normalized TEXT NOT NULL,
  canonical TEXT,
  project_id TEXT,
  entity_id TEXT,
  weight REAL NOT NULL DEFAULT 1.0,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS retrieval_log (
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
  correction_note TEXT,
  request_id TEXT,
  turn_id TEXT,
  model_used TEXT,
  user_accepted INTEGER,
  user_corrected INTEGER,
  helpfulness TEXT,
  task_outcome TEXT,
  feedback_json TEXT
);

CREATE TABLE IF NOT EXISTS ingest_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source_id TEXT NOT NULL,
  source_path TEXT,
  processed_at TEXT NOT NULL,
  action TEXT,
  notes TEXT
);

CREATE TABLE IF NOT EXISTS processed_sources (
  source_id TEXT PRIMARY KEY,
  source_kind TEXT NOT NULL,
  path TEXT,
  content_hash TEXT,
  processed_at TEXT NOT NULL,
  notes TEXT
);

CREATE TABLE IF NOT EXISTS schema_migrations (
  version INTEGER PRIMARY KEY,
  applied_at TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
  memory_id UNINDEXED,
  title,
  canonical_text,
  memory_type UNINDEXED,
  tokenize = 'porter'
);

CREATE INDEX IF NOT EXISTS idx_memory_sources_memory ON memory_sources(memory_id);
CREATE INDEX IF NOT EXISTS idx_vocabulary_norm ON vocabulary(kind, normalized);
CREATE INDEX IF NOT EXISTS idx_retrieval_log_request ON retrieval_log(request_id);
CREATE INDEX IF NOT EXISTS idx_retrieval_log_turn ON retrieval_log(turn_id);
"""

_MEMORY_NEW_COLUMNS = (
    ("remote_ok", "INTEGER NOT NULL DEFAULT 1"),
    ("local_only", "INTEGER NOT NULL DEFAULT 0"),
    ("sensitive", "INTEGER NOT NULL DEFAULT 0"),
    ("never_auto_retrieve", "INTEGER NOT NULL DEFAULT 0"),
    ("never_send_external_model", "INTEGER NOT NULL DEFAULT 0"),
    ("temporal_meaning", "TEXT"),
    ("appears_to_correct", "INTEGER NOT NULL DEFAULT 0"),
)

_RETRIEVAL_LOG_NEW_COLUMNS = (
    ("request_id", "TEXT"),
    ("turn_id", "TEXT"),
    ("model_used", "TEXT"),
    ("user_accepted", "INTEGER"),
    ("user_corrected", "INTEGER"),
    ("helpfulness", "TEXT"),
    ("task_outcome", "TEXT"),
    ("feedback_json", "TEXT"),
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=60.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=60000")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(r[1]) for r in conn.execute(f"PRAGMA table_info({table})")}


def add_columns_if_missing(
    conn: sqlite3.Connection, table: str, columns: Iterable[tuple[str, str]]
) -> list[str]:
    existing = table_columns(conn, table)
    added: list[str] = []
    for name, decl in columns:
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
            added.append(name)
    return added


def _backfill_memory_sources(conn: sqlite3.Connection) -> int:
    """Copy legacy single-provenance fields into memory_sources without dropping them."""
    rows = conn.execute(
        """
        SELECT m.id, m.provenance_conversation_id, m.provenance_message_ids,
               m.provenance_artifact_id, m.provenance_note, m.source_kind, m.created_at
        FROM memories m
        WHERE NOT EXISTS (SELECT 1 FROM memory_sources s WHERE s.memory_id = m.id)
        """
    ).fetchall()
    n = 0
    for r in rows:
        msg_ids = []
        raw = r["provenance_message_ids"]
        if raw:
            try:
                import json

                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    msg_ids = [str(x) for x in parsed if x]
            except Exception:
                msg_ids = []
        if not any(
            [
                r["provenance_conversation_id"],
                msg_ids,
                r["provenance_artifact_id"],
                r["provenance_note"],
            ]
        ):
            continue
        if msg_ids:
            for mid in msg_ids:
                conn.execute(
                    """
                    INSERT INTO memory_sources (
                      id, memory_id, source_kind, conversation_id, message_id,
                      artifact_id, note, role, created_at
                    ) VALUES (?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        f"src-{r['id']}-{mid}"[:64],
                        r["id"],
                        r["source_kind"] or "legacy",
                        r["provenance_conversation_id"],
                        mid,
                        r["provenance_artifact_id"],
                        r["provenance_note"],
                        "original",
                        r["created_at"] or _now(),
                    ),
                )
                n += 1
        else:
            conn.execute(
                """
                INSERT INTO memory_sources (
                  id, memory_id, source_kind, conversation_id, message_id,
                  artifact_id, note, role, created_at
                ) VALUES (?,?,?,?,?,?,?,?,?)
                """,
                (
                    f"src-{r['id']}-legacy"[:64],
                    r["id"],
                    r["source_kind"] or "legacy",
                    r["provenance_conversation_id"],
                    None,
                    r["provenance_artifact_id"],
                    r["provenance_note"],
                    "original",
                    r["created_at"] or _now(),
                ),
            )
            n += 1
    return n


def migrate_schema(conn: sqlite3.Connection) -> dict:
    """Idempotent upgrade for existing Phase 1 DBs. Preserves rows."""
    added_mem = []
    added_log = []
    if "memories" in {
        r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }:
        added_mem = add_columns_if_missing(conn, "memories", _MEMORY_NEW_COLUMNS)
    if "retrieval_log" in {
        r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }:
        added_log = add_columns_if_missing(conn, "retrieval_log", _RETRIEVAL_LOG_NEW_COLUMNS)

    sources = 0
    if "memory_sources" in {
        r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }:
        sources = _backfill_memory_sources(conn)

    conn.execute(
        "INSERT OR IGNORE INTO schema_migrations (version, applied_at) VALUES (?, ?)",
        (SCHEMA_VERSION, _now()),
    )
    conn.commit()
    return {
        "schema_version": SCHEMA_VERSION,
        "added_memory_columns": added_mem,
        "added_retrieval_log_columns": added_log,
        "backfilled_sources": sources,
    }


def ensure_schema(conn: sqlite3.Connection | None = None) -> sqlite3.Connection:
    own = conn is None
    if own:
        conn = connect()
    conn.executescript(SCHEMA_SQL)
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='memories_vec'"
    ).fetchone()
    if not row:
        conn.execute(
            f"""
            CREATE VIRTUAL TABLE memories_vec USING vec0(
              memory_id TEXT PRIMARY KEY,
              embedding float[{EMBED_DIM}]
            )
            """
        )
    migrate_schema(conn)
    return conn
