"""Build small high-signal memory packs. k is a useful maximum, not a fill quota."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from .privacy import DESTINATION_LOCAL
from .project_resolve import resolve_project
from .retrieve import DEFAULT_MIN_SCORE, hybrid_search
from .retrieval_log import log_retrieval
from .schema import connect, ensure_schema

DEFAULT_PACK_SIZE = 8  # useful maximum; may return 0
DEFAULT_MAX_PACK_TOKENS = 480
DEFAULT_MAX_ITEM_TOKENS = 96
DEFAULT_MAX_PROJECT_TOKENS = 96
PROJECT_SLICE_FIELDS = (
    "purpose",
    "current_goal",
    "current_stage",
    "key_decisions",
    "constraints",
    "open_questions",
    "next_actions",
    "current_summary",
)


def _approx_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def _truncate_to_tokens(text: str, max_tokens: int) -> Tuple[str, bool]:
    if max_tokens <= 0 or not text:
        return "", bool(text)
    max_chars = max(1, int(max_tokens) * 4)
    if len(text) <= max_chars:
        return text, False
    cut = text[: max(1, max_chars - 1)].rstrip()
    return cut + "…", True


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


def _project_slice(project: Dict[str, Any], *, max_tokens: int) -> Tuple[Dict[str, Any], bool]:
    slice_ = {
        "id": project["id"],
        "name": project["name"],
    }
    truncated = False
    remaining = max(0, max_tokens - _approx_tokens(project.get("name") or "") - 8)
    per_field = max(8, remaining // max(1, len(PROJECT_SLICE_FIELDS))) if remaining else 0
    for key in PROJECT_SLICE_FIELDS:
        raw = project.get(key)
        if not raw:
            slice_[key] = raw
            continue
        text, was_cut = _truncate_to_tokens(str(raw), per_field)
        slice_[key] = text
        truncated = truncated or was_cut
    # If the compact slice still exceeds the project budget, drop verbose fields.
    while _approx_tokens(json.dumps(slice_)) > max_tokens:
        dropped = False
        for key in reversed(PROJECT_SLICE_FIELDS):
            if slice_.get(key):
                slice_[key] = None
                truncated = True
                dropped = True
                break
        if not dropped:
            break
    return slice_, truncated


def apply_token_budget(
    memories: List[Dict[str, Any]],
    project_slice: Optional[Dict[str, Any]],
    *,
    max_tokens: int = DEFAULT_MAX_PACK_TOKENS,
    max_item_tokens: int = DEFAULT_MAX_ITEM_TOKENS,
    max_project_tokens: int = DEFAULT_MAX_PROJECT_TOKENS,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]], int]:
    """
    Enforce a hard token budget. Drop lowest-relevance first; truncate oversized
    rows. Never silently keep a verbose summary that blows the pack.
    """
    kept: List[Dict[str, Any]] = []
    for item in memories:
        text, cut = _truncate_to_tokens(item.get("canonical_text") or "", max_item_tokens)
        title, tcut = _truncate_to_tokens(item.get("title") or "", min(24, max_item_tokens))
        row = {**item, "canonical_text": text, "title": title}
        if cut or tcut:
            row["truncated"] = True
        kept.append(row)

    proj = None
    if project_slice:
        proj, pcut = _project_slice(project_slice, max_tokens=max_project_tokens)
        if pcut:
            proj = {**proj, "truncated": True}

    def _cost(rows, project) -> int:
        n = sum(_approx_tokens((r.get("title") or "") + " " + (r.get("canonical_text") or "")) for r in rows)
        if project:
            n += _approx_tokens(json.dumps(project))
        return n

    # Drop lowest-relevance last items until we fit.
    while kept and _cost(kept, proj) > max_tokens:
        kept.pop()
    if proj and _cost(kept, proj) > max_tokens:
        # Do not keep an oversized project slice.
        proj = None
    if kept and _cost(kept, None) > max_tokens:
        last = kept[-1]
        room = max_tokens - _cost(kept[:-1], None)
        last["canonical_text"], cut = _truncate_to_tokens(last.get("canonical_text") or "", max(1, room - 4))
        if cut:
            last["truncated"] = True
        if _cost(kept, None) > max_tokens:
            kept.pop()
    return kept, proj, _cost(kept, proj)


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
    max_tokens: int = DEFAULT_MAX_PACK_TOKENS,
    max_item_tokens: int = DEFAULT_MAX_ITEM_TOKENS,
    max_project_tokens: int = DEFAULT_MAX_PROJECT_TOKENS,
) -> Dict[str, Any]:
    """
    Build a memory pack.

    k is the useful *maximum*. Fewer (or zero) memories are returned when
    nothing clears the relevance floor. Explicit current_project_id / project_id
    from the calling agent wins; otherwise resolution may abstain.
    A hard token budget is applied after ranking.
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
    if resolved_id:
        hits = [
            h
            for h in hits
            if not h.get("project_id") or h.get("project_id") == resolved_id
        ]

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
    for h in pack:
        item = {
            "id": h["id"],
            "memory_type": h["memory_type"],
            "title": h["title"],
            "canonical_text": h["canonical_text"],
            "importance": h["importance"],
            "confidence": h["confidence"],
            "origin": h["origin"],
            "project_id": h.get("project_id"),
            "provenance_conversation_id": h["provenance_conversation_id"],
            "provenance_message_ids": json.loads(h["provenance_message_ids"] or "[]"),
            "score": round(h["score"], 4),
        }
        if h["origin"] == "agent_inferred":
            item["label"] = "AGENT_INFERRED_NOT_USER_STATED"
        injected.append(item)

    injected, project_slice, token_cost = apply_token_budget(
        injected,
        project_slice,
        max_tokens=max_tokens,
        max_item_tokens=max_item_tokens,
        max_project_tokens=max_project_tokens,
    )

    result = {
        "query": query,
        "expanded_queries": expanded,
        "project": project_slice,
        "project_resolution": resolution.as_dict(),
        "memories": injected,
        "approx_token_cost": token_cost,
        "pack_size": len(injected),
        "max_tokens": max_tokens,
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
                "max_tokens": max_tokens,
            },
            request_id=request_id,
            turn_id=turn_id,
            model_used=model_used,
            db_path=db_path,
        )
    return result
