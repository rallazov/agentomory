"""Eval: questions needing older info; score right/miss/irrelevant/superseded-used/token cost."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .memory_pack import build_pack
from .schema import connect, ensure_schema

EVAL_PATH = Path(__file__).resolve().parent / "eval_set.json"
RESULTS_PATH = Path(__file__).resolve().parent / "eval_results.json"

# Grounded in Phase 0 / durable markdown content — not invented product claims.
DEFAULT_EVAL = [
    {
        "id": "q-cloud-agent-launch",
        "query": "Cloud Agent launch rule",
        "expect_any_substrings": [
            "story-bound",
            "Cloud Agent",
            "story",
            "non-negotiable",
            "outer-loop",
        ],
        "expect_memory_types": ["preference", "constraint", "strategy", "decision", "fact"],
        "project_hint": "proj-cloud-agent-workflow",
    },
    {
        "id": "q-calendar-sprint",
        "query": "calendar sprint",
        "expect_any_substrings": [
            "calendar",
            "Sprint 1",
            "Nylas",
            "Today",
        ],
        "expect_memory_types": ["decision", "project_state", "fact", "goal", "task"],
        "project_hint": "proj-calendar-sprint1",
    },
    {
        "id": "q-local-memory-boundary",
        "query": "local call memory not product RAG",
        "expect_any_substrings": [
            "not product",
            "vector-memory",
            "local",
            "RAG",
        ],
        "expect_memory_types": ["fact", "constraint", "preference"],
        "project_hint": "proj-local-agent-memory",
    },
    {
        "id": "q-team-ownership",
        "query": "team ownership Jules Kit",
        "expect_any_substrings": [
            "Jules",
            "Kit",
            "ownership",
            "Rae",
        ],
        "expect_memory_types": ["decision", "fact", "preference"],
        "project_hint": None,
    },
    {
        "id": "q-reversed-should-not-dominate",
        "query": "call rule snippet mining turns",
        "expect_any_substrings": [
            "durable",
            "not",
            "transcript",
            "reversed",
            "six-month",
            "hook",
        ],
        "forbid_status_in_pack": ["superseded"],
        "expect_memory_types": ["lesson", "preference", "constraint", "fact"],
        "project_hint": None,
        "notes": "Pack must not treat superseded/reversed as current truth",
    },
]


def ensure_eval_set() -> List[Dict[str, Any]]:
    if not EVAL_PATH.exists():
        EVAL_PATH.write_text(json.dumps(DEFAULT_EVAL, indent=2), encoding="utf-8")
        return DEFAULT_EVAL
    return json.loads(EVAL_PATH.read_text(encoding="utf-8"))


def _score_one(case: Dict[str, Any], pack: Dict[str, Any], conn) -> Dict[str, Any]:
    texts = " ".join(
        m["title"] + " " + m["canonical_text"] for m in pack.get("memories", [])
    )
    ids = [m["id"] for m in pack.get("memories", [])]
    expect = [s.lower() for s in case.get("expect_any_substrings", [])]
    hit_subs = [s for s in expect if s.lower() in texts.lower()]
    right = len(hit_subs) > 0

    # superseded-used: injected id has status superseded (should be 0)
    superseded_used = 0
    for mid in ids:
        row = conn.execute("SELECT status FROM memories WHERE id=?", (mid,)).fetchone()
        if row and row["status"] == "superseded":
            superseded_used += 1

    # irrelevant: no substring hit and no overlapping expected types
    types = {m["memory_type"] for m in pack.get("memories", [])}
    expect_types = set(case.get("expect_memory_types") or [])
    type_overlap = bool(types & expect_types) if expect_types else True
    irrelevant = (not right) and (not type_overlap)

    miss = not right

    return {
        "id": case["id"],
        "query": case["query"],
        "right": right,
        "miss": miss,
        "irrelevant": irrelevant,
        "superseded_used": superseded_used,
        "approx_token_cost": pack.get("approx_token_cost", 0),
        "pack_size": pack.get("pack_size", 0),
        "hit_substrings": hit_subs,
        "injected_ids": ids,
        "origins": [m.get("origin") for m in pack.get("memories", [])],
        "project": (pack.get("project") or {}).get("id") if pack.get("project") else None,
    }


def run_eval(*, db_path=None, out_path: Path | None = None) -> Dict[str, Any]:
    cases = ensure_eval_set()
    conn = ensure_schema(connect(db_path) if db_path else None)
    results = []
    for case in cases:
        pack = build_pack(
            case["query"],
            project_id=case.get("project_hint"),
            db_path=db_path,
            log=True,
        )
        results.append(_score_one(case, pack, conn))

    n = len(results) or 1
    summary = {
        "n": len(results),
        "right_rate": sum(1 for r in results if r["right"]) / n,
        "miss_rate": sum(1 for r in results if r["miss"]) / n,
        "irrelevant_rate": sum(1 for r in results if r["irrelevant"]) / n,
        "superseded_used_total": sum(r["superseded_used"] for r in results),
        "avg_token_cost": sum(r["approx_token_cost"] for r in results) / n,
        "avg_pack_size": sum(r["pack_size"] for r in results) / n,
    }
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "results": results,
    }
    dest = out_path or RESULTS_PATH
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    conn.close()
    return payload
