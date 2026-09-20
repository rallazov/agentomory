"""Build small high-signal memory packs. k is a useful maximum, not a fill quota."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .privacy import DESTINATION_LOCAL
from .project_resolve import resolve_project
from .retrieve import DEFAULT_MIN_SCORE, hybrid_search
from .retrieval_log import log_retrieval
from .schema import connect, ensure_schema

DEFAULT_PACK_SIZE = 8  # useful maximum; may return 0


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


def find_project_for_query(
    query: str,
    *,
    db_path=None,
    current_project_id: Optional[str] = None,
) -> Optional[str]:
    """Backward-compatible wrapper around confidence-aware resolution."""
    return resolve_project(
        query, current_project_id=current_project_id, db_path=db_path
    ).project_id


def build_pack(
    query: str,
    *,
    k: int = DEFAULT_PACK_SIZE,
    project_id: Optional[str] = None,
    current_project_id: Optional[str] = None,
    weights: Optional[Dict[str, float]] = None,
    db_path=None,
    log: bool = True,
    min_score: float = DEFAULT_MIN_SCORE,
    destination: str = DESTINATION_LOCAL,
    request_id: Optional[str] = None,
    turn_id: Optional[str] = None,
    model_used: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a memory pack.

    k is the useful *maximum*. Fewer (or zero) memories are returned when
    nothing clears the relevance floor. Explicit current_project_id / project_id
    from the calling agent wins; otherwise resolution may abstain.
    """
    k = max(0, min(16, int(k)))
    explicit = current_project_id or project_id
    resolution = resolve_project(query, current_project_id=explicit, db_path=db_path)
    resolved_id = resolution.project_id

    hits, expanded, score_map = hybrid_search(
        query,
        k=max(k * 2, 12) if k else 12,
        project_id=resolved_id,
        weights=weights,
        db_path=db_path,
        min_score=min_score,
        destination=destination,
        apply_floor=True,
    )

    # Prefer type diversity but never pad with sub-floor hits.
    pack: List[Dict[str, Any]] = []
    if k > 0:
        seen_types = set()
        for h in hits:
            if len(pack) >= k:
                break
            if h["memory_type"] not in seen_types or len(pack) < max(1, k // 2):
                pack.append(h)
                seen_types.add(h["memory_type"])
        for h in hits:
            if len(pack) >= k:
                break
            if h["id"] not in {p["id"] for p in pack}:
                pack.append(h)

    project = get_project(resolved_id, db_path=db_path) if resolved_id else None
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
        "project_resolution": resolution.as_dict(),
        "memories": injected,
        "approx_token_cost": token_cost,
        "pack_size": len(injected),
        "destination": destination,
        "request_id": request_id,
        "turn_id": turn_id,
    }

    if log:
        log_retrieval(
            query,
            expanded_queries=expanded,
            candidate_ids=[h["id"] for h in hits],
            scores={i: score_map.get(i, 0.0) for i in [h["id"] for h in hits]},
            injected_ids=[m["id"] for m in injected],
            approx_token_cost=token_cost,
            project_id=resolved_id,
            context={
                "pack_size": len(injected),
                "destination": destination,
                "project_resolution": resolution.as_dict(),
            },
            request_id=request_id,
            turn_id=turn_id,
            model_used=model_used,
            db_path=db_path,
        )
    return result
