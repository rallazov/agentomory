"""Seed projects from real durable markdown only — do not invent product facts."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .extract import register_artifact
from .schema import connect, ensure_schema

PROJECT_SEEDS = [
    {
        "id": "proj-calendar-sprint1",
        "name": "Calendar Sprint 1",
        "purpose": "Ship calendar strip on Today via existing Nylas path; acceptance fenced in sprint docs.",
        "current_goal": "Calendar view + connector path usable for counted test users (per ceo-call-summary / sprint-1).",
        "current_stage": "sprint-1",
        "key_decisions": "Sprint 1 calendar strip on Today via existing Nylas; calendar path before tenant RAG.",
        "constraints": "AI/LLM/vector store not a build in the two-month plan; story-bound Cloud Agent launches only.",
        "open_questions": "Ash written Nylas vs Microsoft 365 connector choice still owed before connector locked.",
        "next_actions": "See open items in ceo-call-summary and sprint-1-check.",
        "current_summary": "Grounded in local sprint / ceo-call durable markdown (configure AGENTOMORY_DURABLE_MD).",
        "artifact": Path.home() / ".agentomory" / "durable" / "sprint-1.md",
    },
    {
        "id": "proj-cloud-agent-workflow",
        "name": "Cloud Agent workflow",
        "purpose": "Story-bound Cloud Agent launches; Grok Bot as outer-loop babysitter.",
        "current_goal": "No Cloud Agent launch without written story, branch bound to story, approval gate.",
        "current_stage": "active-rule",
        "key_decisions": "Cloud Agent launch rule (story-bound); template at cloud-agent-story-template.md.",
        "constraints": "Do not invent roadmap work; do not write product code from Grok Bot.",
        "open_questions": "",
        "next_actions": "Use template before any launch.",
        "current_summary": "Grounded in ceo-call-summary.md and cloud-agent-story-template.md.",
        "artifact": Path.home() / ".agentomory" / "durable" / "cloud-agent-story-template.md",
    },
    {
        "id": "proj-local-agent-memory",
        "name": "Local agent memory",
        "purpose": "Persistent local agent memory (Agentomory) — not product RAG.",
        "current_goal": "Phase 1 agent_memory.sqlite with Layer A/B, hybrid packs, eval.",
        "current_stage": "phase-1",
        "key_decisions": "CHAT HISTORY ≠ MEMORY; local-only boundary; new DB not in Domayn repo.",
        "constraints": "Do not ship into application-tracker; no Neo4j/IntentGraph in Phase 1.",
        "open_questions": "",
        "next_actions": "Run phase1 CLI query/pack + eval.",
        "current_summary": "Grounded in ARCHITECTURE.md / MEMORY_SPEC.md in this repo.",
        "artifact": Path(__file__).resolve().parent.parent / "ARCHITECTURE.md",
    },
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def seed(*, db_path=None) -> List[Dict[str, Any]]:
    # Register artifacts first (own short-lived connections) — do not nest writers.
    art_ids = {}
    for p in PROJECT_SEEDS:
        if p["artifact"].exists():
            art_ids[p["id"]] = register_artifact(p["artifact"], "durable_md", db_path=db_path)

    conn = ensure_schema(connect(db_path) if db_path else None)
    out = []
    try:
        for p in PROJECT_SEEDS:
            art_id = art_ids.get(p["id"])
            now = _now()
            conn.execute(
                """
                INSERT INTO projects (
                  id, name, purpose, current_goal, current_stage, key_decisions,
                  constraints, open_questions, next_actions, current_summary,
                  status, source_artifact_id, created_at, updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                  purpose=excluded.purpose,
                  current_goal=excluded.current_goal,
                  current_stage=excluded.current_stage,
                  key_decisions=excluded.key_decisions,
                  constraints=excluded.constraints,
                  open_questions=excluded.open_questions,
                  next_actions=excluded.next_actions,
                  current_summary=excluded.current_summary,
                  source_artifact_id=excluded.source_artifact_id,
                  updated_at=excluded.updated_at
                """,
                (
                    p["id"], p["name"], p["purpose"], p["current_goal"], p["current_stage"],
                    p["key_decisions"], p["constraints"], p["open_questions"], p["next_actions"],
                    p["current_summary"], "active", art_id, now, now,
                ),
            )
            out.append({"id": p["id"], "name": p["name"], "artifact_id": art_id})
        conn.commit()
    finally:
        conn.close()
    # Aliases live in SQLite so retrieve.py / memory_pack.py stay engine-generic.
    _seed_vocabulary(db_path=db_path)
    return out


# Optional seed data only — never imported by retrieve.py / memory_pack.py.
_VOCAB_SEEDS = [
    {"term": "calendar", "kind": "project_alias", "project_id": "proj-calendar-sprint1", "canonical": "Calendar Sprint 1"},
    {"term": "calendar sprint", "kind": "project_alias", "project_id": "proj-calendar-sprint1", "canonical": "Calendar Sprint 1"},
    {"term": "sprint 1", "kind": "project_alias", "project_id": "proj-calendar-sprint1", "canonical": "Calendar Sprint 1"},
    {"term": "nylas", "kind": "project_alias", "project_id": "proj-calendar-sprint1", "canonical": "Calendar Sprint 1"},
    {"term": "cloud agent", "kind": "project_alias", "project_id": "proj-cloud-agent-workflow", "canonical": "Cloud Agent workflow"},
    {"term": "story-bound", "kind": "synonym", "project_id": "proj-cloud-agent-workflow", "canonical": "story-bound launch rule"},
    {"term": "outer-loop", "kind": "synonym", "project_id": "proj-cloud-agent-workflow", "canonical": "outer-loop babysitter"},
    {"term": "local memory", "kind": "project_alias", "project_id": "proj-local-agent-memory", "canonical": "Local agent memory"},
    {"term": "agentomory", "kind": "project_alias", "project_id": "proj-local-agent-memory", "canonical": "Local agent memory"},
]


def _seed_vocabulary(*, db_path=None) -> None:
    from .vocabulary import upsert_term

    for row in _VOCAB_SEEDS:
        upsert_term(db_path=db_path, **row)
