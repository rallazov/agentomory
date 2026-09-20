"""Build small memory packs for CEO turns."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .retrieve import hybrid_search
from .retrieval_log import log_retrieval
from .schema import connect, ensure_schema

DEFAULT_PACK_SIZE = 6  # within 3–8


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def get_project(project_id: str, *, db_path=None) -> Optional[Dict[str, Any]]:
    conn = ensure_schema(connect(db_path) if db_path else None)
    row = conn.execute("SELECT * FROM projects WHERE id=?", (project_id,)).fetchone()
    if not row:
        conn.close()
        return None
    d = dict(row)
    conn.close()
    return d


def find_project_for_query(query: str, *, db_path=None) -> Optional[str]:
    conn = ensure_schema(connect(db_path) if db_path else None)
    q = query.lower()
    rows = conn.execute("SELECT id, name, purpose, current_summary FROM projects WHERE status='active'").fetchall()
    found = None
    for r in rows:
        blob = f"{r['name']} {r['purpose'] or ''} {r['current_summary'] or ''}".lower()
        if any(tok in blob for tok in q.split() if len(tok) > 3):
            found = r["id"]; break
        if "calendar" in q and "calendar" in blob:
            found = r["id"]; break
        if "cloud agent" in q and "cloud" in blob:
            found = r["id"]; break
    conn.close()
    return found


def build_pack(
    query: str,
    *,
    k: int = DEFAULT_PACK_SIZE,
    project_id: Optional[str] = None,
    weights: Optional[Dict[str, float]] = None,
    db_path=None,
    log: bool = True,
) -> Dict[str, Any]:
    k = max(3, min(8, k))
    if not project_id:
        project_id = find_project_for_query(query, db_path=db_path)

    hits, expanded, score_map = hybrid_search(
        query, k=max(k * 2, 12), project_id=project_id, weights=weights, db_path=db_path
    )

    # Prefer diversity of types in pack
    pack: List[Dict[str, Any]] = []
    seen_types = set()
    for h in hits:
        if len(pack) >= k:
            break
        # first pass: fill unique types
        if h["memory_type"] not in seen_types or len(pack) < k // 2:
            pack.append(h)
            seen_types.add(h["memory_type"])
    for h in hits:
        if len(pack) >= k:
            break
        if h["id"] not in {p["id"] for p in pack}:
            pack.append(h)

    project = get_project(project_id, db_path=db_path) if project_id else None
    project_slice = None
    if project:
        project_slice = {
            "id": project["id"],
            "name": project["name"],
            "purpose": project.get("purpose"),
            "current_goal": project.get("current_goal"),
            "current_stage": project.get("current_stage"),
            "key_decisions": project.get("key_decisions"),
            "constraints": project.get("constraints"),
            "open_questions": project.get("open_questions"),
            "next_actions": project.get("next_actions"),
            "current_summary": project.get("current_summary"),
        }

    # Strip component detail for injection; keep provenance
    injected = []
    token_cost = 0
    for h in pack:
        item = {
            "id": h["id"],
            "memory_type": h["memory_type"],
            "title": h["title"],
            "canonical_text": h["canonical_text"],
            "importance": h["importance"],
            "confidence": h["confidence"],
            "origin": h["origin"],
            "provenance_conversation_id": h["provenance_conversation_id"],
            "provenance_message_ids": json.loads(h["provenance_message_ids"] or "[]"),
            "score": round(h["score"], 4),
        }
        # Label agent-inferred clearly
        if h["origin"] == "agent_inferred":
            item["label"] = "AGENT_INFERRED_NOT_USER_STATED"
        injected.append(item)
        token_cost += _approx_tokens(h["title"] + " " + h["canonical_text"])

    if project_slice:
        token_cost += _approx_tokens(json.dumps(project_slice))

    result = {
        "query": query,
        "expanded_queries": expanded,
        "project": project_slice,
        "memories": injected,
        "approx_token_cost": token_cost,
        "pack_size": len(injected),
    }

    if log:
        log_retrieval(
            query,
            expanded_queries=expanded,
            candidate_ids=[h["id"] for h in hits],
            scores={i: score_map.get(i, 0.0) for i in [h["id"] for h in hits]},
            injected_ids=[m["id"] for m in injected],
            approx_token_cost=token_cost,
            project_id=project_id,
            context={"pack_size": len(injected)},
            db_path=db_path,
        )
    return result
