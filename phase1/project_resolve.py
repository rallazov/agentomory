"""Project resolution: explicit caller hint, aliases, entities, semantic search, abstain."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .embed import get_embedder
from .schema import connect, ensure_schema
from .vocabulary import terms_matching_query

DEFAULT_MIN_CONFIDENCE = 0.58
AMBIGUITY_GAP = 0.10


@dataclass
class ProjectResolution:
    project_id: Optional[str]
    confidence: float
    method: str
    candidates: List[Tuple[str, float]] = field(default_factory=list)
    reason: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "project_id": self.project_id,
            "confidence": round(self.confidence, 4),
            "method": self.method,
            "candidates": [{"id": i, "score": round(s, 4)} for i, s in self.candidates],
            "reason": self.reason,
        }


def _none(reason: str, candidates: Optional[List[Tuple[str, float]]] = None) -> ProjectResolution:
    return ProjectResolution(
        project_id=None,
        confidence=0.0,
        method="none",
        candidates=candidates or [],
        reason=reason,
    )


def _project_text(row) -> str:
    parts = [
        row["name"],
        row["purpose"] or "",
        row["current_goal"] or "",
        row["current_stage"] or "",
        row["current_summary"] or "",
    ]
    return " ".join(p for p in parts if p).strip()


def _cosine_bytes(a: bytes, b: bytes) -> float:
    import numpy as np

    va = np.frombuffer(a, dtype=np.float32)
    vb = np.frombuffer(b, dtype=np.float32)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def resolve_project(
    query: str,
    *,
    current_project_id: Optional[str] = None,
    db_path=None,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    conn=None,
) -> ProjectResolution:
    """
    Resolve a project for a query.

    Order:
      1. Explicit current project from the calling agent (if the id exists).
      2. Vocabulary aliases / entity aliases stored in SQLite.
      3. Semantic similarity of the query to project name+purpose+summary.
      4. If uncertain (low score or two close winners) → no project.
    """
    own = conn is None
    if own:
        conn = ensure_schema(connect(db_path) if db_path else None)
    try:
        if current_project_id:
            row = conn.execute(
                "SELECT id FROM projects WHERE id=? AND status='active'",
                (current_project_id,),
            ).fetchone()
            if row:
                return ProjectResolution(
                    project_id=row["id"],
                    confidence=1.0,
                    method="explicit",
                    candidates=[(row["id"], 1.0)],
                    reason="caller_current_project",
                )

        projects = conn.execute("SELECT * FROM projects WHERE status='active'").fetchall()
        if not projects:
            return _none("no_active_projects")

        scores: Dict[str, float] = {p["id"]: 0.0 for p in projects}
        methods: Dict[str, str] = {}

        # Alias / entity hits from stored vocabulary (generic).
        vocab_hits = terms_matching_query(query, conn=conn)
        for hit in vocab_hits:
            pid = hit.get("project_id")
            if pid and pid in scores:
                # Longer / more specific alias → higher confidence.
                nlen = len(hit.get("normalized") or "")
                bump = 0.72 + min(0.2, nlen / 80.0)
                bump *= float(hit.get("weight") or 1.0)
                if bump > scores[pid]:
                    scores[pid] = min(0.97, bump)
                    methods[pid] = "alias"
            eid = hit.get("entity_id")
            if eid:
                erow = conn.execute(
                    "SELECT meta_json FROM entities WHERE id=?", (eid,)
                ).fetchone()
                # entity may name a project via vocabulary.project_id already handled;
                # also: memories/projects linked only through project_id on the vocab row.

        # Name mention (full name or distinctive multi-word name), not first token-in-summary.
        qlow = (query or "").strip().lower()
        for p in projects:
            name = (p["name"] or "").strip().lower()
            # Require a distinctive name (multi-word or long). Do not bind on a
            # short token that happens to appear in a project name or summary.
            distinctive = bool(name) and ((" " in name and name in qlow) or (len(name) >= 10 and name in qlow))
            if distinctive:
                scores[p["id"]] = max(scores[p["id"]], 0.88)
                methods[p["id"]] = methods.get(p["id"]) or "name"

        # Semantic similarity over project text.
        try:
            emb = get_embedder()
            qvec = emb.embed_query(query)
            for p in projects:
                blob = _project_text(p)
                if not blob:
                    continue
                pvec = emb.embed_documents([blob])[0]
                sim = _cosine_bytes(qvec, pvec)
                # Semantic alone must be fairly high to win; do not accept weak overlap.
                if sim >= 0.52:
                    combined = 0.55 + 0.4 * ((sim - 0.52) / 0.48)
                    if combined > scores[p["id"]]:
                        scores[p["id"]] = min(0.9, combined)
                        methods[p["id"]] = "semantic"
        except Exception:
            pass

        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        ranked = [(i, s) for i, s in ranked if s > 0]
        if not ranked:
            return _none("no_signal")

        best_id, best = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        if best < min_confidence:
            return _none("below_confidence", ranked)
        if second > 0 and (best - second) < AMBIGUITY_GAP and second >= min_confidence - 0.05:
            return _none("ambiguous", ranked)
        return ProjectResolution(
            project_id=best_id,
            confidence=best,
            method=methods.get(best_id, "unknown"),
            candidates=ranked[:5],
            reason="resolved",
        )
    finally:
        if own:
            conn.close()
