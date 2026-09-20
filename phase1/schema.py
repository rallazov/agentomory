"""Phase 1 schema for agent_memory.sqlite."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import sqlite_vec

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "agent_memory.sqlite"
EMBED_DIM = 384

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
  CHECK (status != 'reversed' OR why_failed IS NOT NULL)
);

CREATE TABLE IF NOT EXISTS memory_links (
  id TEXT PRIMARY KEY,
  from_memory_id TEXT NOT NULL REFERENCES memories(id),
  to_memory_id TEXT NOT NULL REFERENCES memories(id),
  relationship TEXT NOT NULL,
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
  correction_note TEXT
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

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
  memory_id UNINDEXED,
  title,
  canonical_text,
  memory_type UNINDEXED,
  tokenize = 'porter'
);
"""


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
    conn.commit()
    return conn
