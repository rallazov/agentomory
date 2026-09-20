"""Hybrid retrieval: vector + FTS + filters + configurable weights. No project-specific aliases."""
from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .embed import get_embedder
from .privacy import DESTINATION_LOCAL, allowed_for_destination
from .schema import connect, ensure_schema
from .vocabulary import expansions_for_query

DEFAULT_WEIGHTS = {
    "vector": 0.35,
    "fts": 0.25,
    "importance": 0.15,
    "confidence": 0.10,
    "recency": 0.08,
    "type_boost": 0.05,
    "origin_penalty": 0.07,  # subtract for agent_inferred
    "correction_boost": 0.12,
}

TYPE_BOOST = {
    "correction": 1.0,
    "decision": 0.9,
    "constraint": 0.9,
    "preference": 0.85,
    "project_state": 0.8,
    "task": 0.7,
    "goal": 0.75,
    "lesson": 0.8,
    "outcome": 0.7,
    "strategy": 0.65,
    "fact": 0.6,
    "open_question": 0.5,
}

# Absolute relevance floor. Pack size is a maximum, not a quota.
DEFAULT_MIN_SCORE = 0.55
DEFAULT_RELATIVE_KEEP = 0.70
# BGE-small unrelated English often sits ~0.45–0.55 cosine; require above that.
DEFAULT_MIN_COSINE = 0.62


def expand_query(query: str, *, conn=None, db_path=None) -> List[str]:
    """Expand using vocabulary stored in SQLite. Engine stays project-agnostic."""
    expansions = [query.strip()]
    seen = {query.strip().lower()}
    try:
        extra = expansions_for_query(query, conn=conn, db_path=db_path)
    except Exception:
        extra = []
    for e in extra:
        key = e.strip().lower()
        if key and key not in seen:
            seen.add(key)
            expansions.append(e)
    return expansions


def _recency_score(updated_at: str) -> float:
    try:
        dt = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
        age_days = max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0)
        return math.exp(-age_days / 45.0)
    except Exception:
        return 0.5


_FTS_STOP = {
    "the", "and", "for", "are", "but", "not", "you", "all", "any", "can",
    "had", "her", "was", "one", "our", "out", "has", "have", "been", "from",
    "they", "this", "that", "with", "what", "when", "your", "how", "why",
    "about", "into", "just", "over", "also", "than",
}


def _fts_query(text: str) -> str:
    words = [
        w
        for w in re.findall(r"[A-Za-z0-9]{3,}", text)
        if w.lower() not in _FTS_STOP
    ]
    return " OR ".join(words[:10]) if words else ""


def _content_tokens(text: str) -> set:
    return {
        w
        for w in re.findall(r"[a-z0-9]{3,}", (text or "").lower())
        if w not in _FTS_STOP
    }


def _token_overlap(query: str, text: str) -> float:
    q = _content_tokens(query)
    if not q:
        return 0.0
    m = _content_tokens(text)
    return len(q & m) / len(q)


def _l2_to_cosine(dist: float) -> float:
    return max(0.0, 1.0 - (float(dist) ** 2) / 2.0)


def apply_relevance_floor(
    hits: List[Dict[str, Any]],
    *,
    min_score: float = DEFAULT_MIN_SCORE,
    relative: float = DEFAULT_RELATIVE_KEEP,
) -> List[Dict[str, Any]]:
    """Drop weak hits. May return an empty list — that is correct."""
    if not hits:
        return []
    best = hits[0]["score"]
    if best < min_score:
        return []
    floor = max(min_score * 0.85, best * relative)
    return [h for h in hits if h["score"] >= floor]


def hybrid_search(
    query: str,
    *,
    k: int = 12,
    project_id: Optional[str] = None,
    memory_types: Optional[List[str]] = None,
    weights: Optional[Dict[str, float]] = None,
    db_path=None,
    min_score: float = DEFAULT_MIN_SCORE,
    destination: str = DESTINATION_LOCAL,
    apply_floor: bool = True,
) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, float]]:
    """
    Returns (scored hits, expanded_queries, per-id scores).
    Only active memories. k is a maximum; relevance floor may yield zero hits.
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    conn = ensure_schema(connect(db_path) if db_path else None)
    expanded = expand_query(query, conn=conn)

    # --- vector (absolute cosine, not intra-set min-max) ---
    fetch_n = max(k * 5, 40)
    vec_score: Dict[str, float] = {}
    try:
        emb = get_embedder().embed_query(query)
        vec_rows = conn.execute(
            """
            SELECT v.memory_id AS memory_id, v.distance AS distance
            FROM memories_vec v
            WHERE v.embedding MATCH ? AND k = ?
            """,
            (emb, fetch_n),
        ).fetchall()
        for r in vec_rows:
            cos = _l2_to_cosine(float(r["distance"]))
            if cos >= DEFAULT_MIN_COSINE:
                vec_score[r["memory_id"]] = cos
    except Exception:
        vec_score = {}

    # --- FTS ---
    fts_score: Dict[str, float] = {}
    for eq in expanded:
        fq = _fts_query(eq)
        if not fq.strip():
            continue
        try:
            rows = conn.execute(
                """
                SELECT f.memory_id AS memory_id, bm25(memories_fts) AS rank
                FROM memories_fts f
                JOIN memories m ON m.id = f.memory_id
                WHERE memories_fts MATCH ? AND m.status = 'active'
                ORDER BY rank
                LIMIT ?
                """,
                (fq, fetch_n),
            ).fetchall()
        except Exception:
            continue
        if not rows:
            continue
        ranks = [float(r["rank"]) for r in rows]
        rmin, rmax = min(ranks), max(ranks)
        for r in rows:
            raw = float(r["rank"])
            norm = 1.0 - ((raw - rmin) / (rmax - rmin + 1e-6))
            fts_score[r["memory_id"]] = max(fts_score.get(r["memory_id"], 0.0), norm)

    ids = set(vec_score) | set(fts_score)
    # Do not union every project memory — that forced irrelevant fills.

    hits: List[Dict[str, Any]] = []
    score_map: Dict[str, float] = {}
    for mid in ids:
        row = conn.execute("SELECT * FROM memories WHERE id=?", (mid,)).fetchone()
        if not row or row["status"] != "active":
            continue
        if not allowed_for_destination(row, destination, auto_retrieve=True):
            continue
        if memory_types and row["memory_type"] not in memory_types:
            continue

        vs = vec_score.get(mid, 0.0)
        fs = fts_score.get(mid, 0.0)
        overlap = _token_overlap(query, f"{row['title']} {row['canonical_text']}")
        # BGE baseline cosine is high; require a real neighbor or lexical overlap.
        if vs < DEFAULT_MIN_COSINE and overlap < 0.15:
            continue

        imp = float(row["importance"])
        conf = float(row["confidence"])
        rec = _recency_score(row["updated_at"])
        tb = TYPE_BOOST.get(row["memory_type"], 0.5)
        origin_pen = 1.0 if row["origin"] == "agent_inferred" else 0.0
        corr = 1.0 if row["memory_type"] == "correction" else 0.0
        project_boost = 0.08 if project_id and row["project_id"] == project_id else 0.0
        if project_id and row["project_id"] and row["project_id"] != project_id:
            project_boost = -0.06

        score = (
            w["vector"] * vs
            + w["fts"] * fs
            + w["importance"] * imp
            + w["confidence"] * conf
            + w["recency"] * rec
            + w["type_boost"] * tb
            + w["correction_boost"] * corr
            + project_boost
            - w["origin_penalty"] * origin_pen
        )
        if row["origin"] == "agent_inferred" and conf < 0.5:
            score *= 0.5

        score_map[mid] = score
        hits.append(
            {
                "id": mid,
                "score": score,
                "memory_type": row["memory_type"],
                "title": row["title"],
                "canonical_text": row["canonical_text"],
                "importance": imp,
                "confidence": conf,
                "origin": row["origin"],
                "status": row["status"],
                "project_id": row["project_id"],
                "supersedes_id": row["supersedes_id"],
                "provenance_conversation_id": row["provenance_conversation_id"],
                "provenance_message_ids": row["provenance_message_ids"],
                "provenance_artifact_id": row["provenance_artifact_id"],
                "remote_ok": row["remote_ok"] if "remote_ok" in row.keys() else 1,
                "local_only": row["local_only"] if "local_only" in row.keys() else 0,
                "sensitive": row["sensitive"] if "sensitive" in row.keys() else 0,
                "never_auto_retrieve": row["never_auto_retrieve"] if "never_auto_retrieve" in row.keys() else 0,
                "never_send_external_model": (
                    row["never_send_external_model"] if "never_send_external_model" in row.keys() else 0
                ),
                "temporal_meaning": row["temporal_meaning"] if "temporal_meaning" in row.keys() else None,
                "components": {
                    "vector": vs,
                    "fts": fs,
                    "importance": imp,
                    "confidence": conf,
                    "recency": rec,
                    "type_boost": tb,
                    "origin_penalty": origin_pen,
                    "overlap": overlap,
                },
            }
        )

    hits.sort(key=lambda h: -h["score"])
    if apply_floor:
        hits = apply_relevance_floor(hits, min_score=min_score)
    conn.close()
    return hits[:k], expanded, score_map
