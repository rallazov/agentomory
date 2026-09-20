"""Hybrid retrieval: vector + FTS + filters + configurable weights."""
from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .embed import get_embedder
from .schema import connect, ensure_schema

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


def expand_query(query: str) -> List[str]:
    q = query.strip()
    expansions = [q]
    aliases = {
        r"cloud agent": ["story-bound", "cloud agent launch", "story template"],
        r"calendar": ["sprint 1", "calendar strip", "nylas", "today calendar"],
        r"sprint": ["sprint 1", "calendar sprint", "two-month plan"],
        r"launch rule": ["cloud agent launch", "non-negotiable", "story-bound"],
    }
    low = q.lower()
    for pat, extras in aliases.items():
        if re.search(pat, low):
            expansions.extend(extras)
    # unique preserve order
    seen = set()
    out = []
    for e in expansions:
        if e.lower() not in seen:
            seen.add(e.lower())
            out.append(e)
    return out


def _recency_score(updated_at: str) -> float:
    try:
        dt = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
        age_days = max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0)
        return math.exp(-age_days / 45.0)
    except Exception:
        return 0.5


def _fts_query(text: str) -> str:
    words = re.findall(r"[A-Za-z0-9]{3,}", text)
    return " OR ".join(words[:10]) if words else text


def hybrid_search(
    query: str,
    *,
    k: int = 12,
    project_id: Optional[str] = None,
    memory_types: Optional[List[str]] = None,
    weights: Optional[Dict[str, float]] = None,
    db_path=None,
) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, float]]:
    """
    Returns (scored hits, expanded_queries, per-id raw component scores summary).
    Only active memories; corrections/supersession already reflected in status.
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    conn = ensure_schema(connect(db_path) if db_path else None)
    expanded = expand_query(query)

    # --- vector ---
    emb = get_embedder().embed_query(query)
    fetch_n = max(k * 5, 40)
    vec_rows = conn.execute(
        """
        SELECT v.memory_id AS memory_id, v.distance AS distance, m.*
        FROM memories_vec v
        JOIN memories m ON m.id = v.memory_id
        WHERE v.embedding MATCH ? AND k = ?
          AND m.status = 'active'
        ORDER BY v.distance
        """,
        (emb, fetch_n),
    ).fetchall()
    vec_score: Dict[str, float] = {}
    if vec_rows:
        dmax = max(float(r["distance"]) for r in vec_rows) or 1.0
        for r in vec_rows:
            # lower distance better → 1 - normalized
            vec_score[r["memory_id"]] = 1.0 - (float(r["distance"]) / (dmax + 1e-6))

    # --- FTS ---
    fts_score: Dict[str, float] = {}
    for eq in expanded:
        fq = _fts_query(eq)
        if not fq.strip():
            continue
        try:
            rows = conn.execute(
                """
                SELECT f.memory_id AS memory_id, bm25(memories_fts) AS rank, m.*
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
        # bm25: lower is better in sqlite fts5
        ranks = [float(r["rank"]) for r in rows]
        rmin, rmax = min(ranks), max(ranks)
        for r in rows:
            raw = float(r["rank"])
            norm = 1.0 - ((raw - rmin) / (rmax - rmin + 1e-6))
            fts_score[r["memory_id"]] = max(fts_score.get(r["memory_id"], 0.0), norm)

    # candidate universe
    ids = set(vec_score) | set(fts_score)
    if project_id:
        proj_rows = conn.execute(
            "SELECT id FROM memories WHERE status='active' AND project_id=?",
            (project_id,),
        ).fetchall()
        ids |= {r["id"] for r in proj_rows}

    hits: List[Dict[str, Any]] = []
    score_map: Dict[str, float] = {}
    for mid in ids:
        row = conn.execute("SELECT * FROM memories WHERE id=?", (mid,)).fetchone()
        if not row or row["status"] != "active":
            continue
        if memory_types and row["memory_type"] not in memory_types:
            continue
        if project_id and row["project_id"] and row["project_id"] != project_id:
            # soft: still allow but no project boost
            pass

        vs = vec_score.get(mid, 0.0)
        fs = fts_score.get(mid, 0.0)
        imp = float(row["importance"])
        conf = float(row["confidence"])
        rec = _recency_score(row["updated_at"])
        tb = TYPE_BOOST.get(row["memory_type"], 0.5)
        origin_pen = 1.0 if row["origin"] == "agent_inferred" else 0.0
        corr = 1.0 if row["memory_type"] == "correction" else 0.0
        project_boost = 0.08 if project_id and row["project_id"] == project_id else 0.0

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
        # never present agent speculation as user fact: downrank hard if low conf inferred
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
                "components": {
                    "vector": vs,
                    "fts": fs,
                    "importance": imp,
                    "confidence": conf,
                    "recency": rec,
                    "type_boost": tb,
                    "origin_penalty": origin_pen,
                },
            }
        )

    hits.sort(key=lambda h: -h["score"])
    conn.close()
    return hits[:k], expanded, score_map
