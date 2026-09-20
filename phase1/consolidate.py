"""Before insert: search existing → CREATE | MERGE | UPDATE | SUPERSEDE | IGNORE."""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .embed import get_embedder
from .schema import connect, ensure_schema

ACTION_CREATE = "CREATE"
ACTION_MERGE = "MERGE"
ACTION_UPDATE = "UPDATE"
ACTION_SUPERSEDE = "SUPERSEDE"
ACTION_IGNORE = "IGNORE"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _norm(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"\s+", " ", t)
    return t


def _token_set(text: str) -> set:
    return set(re.findall(r"[a-z0-9]{3,}", text.lower()))


def jaccard(a: str, b: str) -> float:
    sa, sb = _token_set(a), _token_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def find_similar(
    conn,
    canonical_text: str,
    memory_type: Optional[str] = None,
    *,
    limit: int = 8,
) -> List[Any]:
    # FTS candidates
    words = [w for w in re.findall(r"[A-Za-z0-9]{3,}", canonical_text)[:12]]
    rows: List[Any] = []
    if words:
        q = " OR ".join(words)
        try:
            fts = conn.execute(
                """
                SELECT m.* FROM memories_fts f
                JOIN memories m ON m.id = f.memory_id
                WHERE memories_fts MATCH ? AND m.status = 'active'
                LIMIT ?
                """,
                (q, limit),
            ).fetchall()
            rows.extend(fts)
        except Exception:
            pass
    # Fallback scan of active same-type
    if memory_type:
        typed = conn.execute(
            "SELECT * FROM memories WHERE status='active' AND memory_type=? LIMIT 200",
            (memory_type,),
        ).fetchall()
        rows.extend(typed)
    else:
        typed = conn.execute(
            "SELECT * FROM memories WHERE status='active' LIMIT 300"
        ).fetchall()
        rows.extend(typed)

    seen = set()
    out = []
    scored: List[Tuple[float, Any]] = []
    for r in rows:
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        score = jaccard(canonical_text, r["canonical_text"])
        if _norm(canonical_text) == _norm(r["canonical_text"]):
            score = 1.0
        if score >= 0.45:
            scored.append((score, r))
    scored.sort(key=lambda x: -x[0])
    return [r for _, r in scored[:limit]]


def decide_action(
    candidate: Dict[str, Any], similar: List[Any]
) -> Tuple[str, Optional[Any]]:
    """Return (action, target_row)."""
    if not similar:
        return ACTION_CREATE, None
    best = similar[0]
    sim = jaccard(candidate["canonical_text"], best["canonical_text"])
    if sim >= 0.92:
        # identical-ish
        if candidate.get("memory_type") == "correction":
            return ACTION_SUPERSEDE, best
        # same content → ignore duplicate
        if abs(float(candidate.get("importance", 0.5)) - float(best["importance"])) < 0.05:
            return ACTION_IGNORE, best
        return ACTION_UPDATE, best
    if sim >= 0.75:
        if candidate.get("memory_type") == "correction":
            return ACTION_SUPERSEDE, best
        # merge into existing if same type
        if candidate.get("memory_type") == best["memory_type"]:
            return ACTION_MERGE, best
        return ACTION_CREATE, None
    if candidate.get("memory_type") == "correction" and sim >= 0.5:
        return ACTION_SUPERSEDE, best
    return ACTION_CREATE, None


def _upsert_vec(conn, memory_id: str, text: str) -> None:
    emb = get_embedder().embed_documents([text])[0]
    conn.execute("DELETE FROM memories_vec WHERE memory_id = ?", (memory_id,))
    conn.execute(
        "INSERT INTO memories_vec (memory_id, embedding) VALUES (?, ?)",
        (memory_id, emb),
    )


def _upsert_fts(conn, memory_id: str, title: str, canonical_text: str, memory_type: str) -> None:
    conn.execute("DELETE FROM memories_fts WHERE memory_id = ?", (memory_id,))
    conn.execute(
        "INSERT INTO memories_fts (memory_id, title, canonical_text, memory_type) VALUES (?,?,?,?)",
        (memory_id, title, canonical_text, memory_type),
    )



def apply_memory(candidate: Dict[str, Any], *, db_path=None) -> Dict[str, Any]:
    """
    candidate keys: memory_type, title, canonical_text, importance, confidence,
    origin, project_id, provenance_*, supersedes_id, why_failed, source_kind, phase0_entry_id
    """
    conn = ensure_schema(connect(db_path) if db_path else None)
    try:
        similar = find_similar(conn, candidate["canonical_text"], candidate.get("memory_type"))
        action, target = decide_action(candidate, similar)
        now = _now()

        if action == ACTION_IGNORE:
            return {"action": action, "id": target["id"], "reason": "near_duplicate"}

        if action == ACTION_UPDATE and target is not None:
            conn.execute(
                """
                UPDATE memories SET
                  title=?, canonical_text=?, importance=?, confidence=?,
                  origin=?, project_id=COALESCE(?, project_id),
                  provenance_conversation_id=COALESCE(?, provenance_conversation_id),
                  provenance_message_ids=COALESCE(?, provenance_message_ids),
                  provenance_artifact_id=COALESCE(?, provenance_artifact_id),
                  provenance_note=COALESCE(?, provenance_note),
                  updated_at=?
                WHERE id=?
                """,
                (
                    candidate.get("title") or target["title"],
                    candidate["canonical_text"],
                    float(candidate.get("importance", target["importance"])),
                    float(candidate.get("confidence", target["confidence"])),
                    candidate.get("origin") or target["origin"],
                    candidate.get("project_id"),
                    candidate.get("provenance_conversation_id"),
                    candidate.get("provenance_message_ids"),
                    candidate.get("provenance_artifact_id"),
                    candidate.get("provenance_note"),
                    now,
                    target["id"],
                ),
            )
            _upsert_fts(conn, target["id"], candidate.get("title") or target["title"], candidate["canonical_text"], target["memory_type"])
            _upsert_vec(conn, target["id"], candidate["canonical_text"])
            conn.commit()
            return {"action": action, "id": target["id"]}

        if action == ACTION_MERGE and target is not None:
            note = (target["provenance_note"] or "") + f" | merged:{candidate.get('title','')}"
            imp = max(float(target["importance"]), float(candidate.get("importance", 0.5)))
            conf = max(float(target["confidence"]), float(candidate.get("confidence", 0.5)))
            text_body = target["canonical_text"]
            if len(candidate["canonical_text"]) > len(text_body) + 20:
                text_body = candidate["canonical_text"]
            conn.execute(
                """
                UPDATE memories SET canonical_text=?, importance=?, confidence=?,
                  provenance_note=?, updated_at=? WHERE id=?
                """,
                (text_body, imp, conf, note[:2000], now, target["id"]),
            )
            _upsert_fts(conn, target["id"], target["title"], text_body, target["memory_type"])
            _upsert_vec(conn, target["id"], text_body)
            conn.commit()
            return {"action": action, "id": target["id"]}

        if action == ACTION_SUPERSEDE and target is not None:
            new_id = candidate.get("id") or f"mem-{uuid.uuid4().hex[:12]}"
            conn.execute(
                "UPDATE memories SET status='superseded', updated_at=? WHERE id=?",
                (now, target["id"]),
            )
            conn.execute(
                """
                INSERT INTO memories (
                  id, memory_type, status, title, canonical_text, importance, confidence,
                  origin, project_id, supersedes_id, why_failed,
                  provenance_conversation_id, provenance_message_ids, provenance_artifact_id,
                  provenance_note, source_kind, phase0_entry_id, created_at, updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    new_id,
                    candidate["memory_type"],
                    "active",
                    candidate["title"],
                    candidate["canonical_text"],
                    float(candidate.get("importance", 0.7)),
                    float(candidate.get("confidence", 0.8)),
                    candidate.get("origin", "user_stated"),
                    candidate.get("project_id"),
                    target["id"],
                    candidate.get("why_failed"),
                    candidate.get("provenance_conversation_id"),
                    candidate.get("provenance_message_ids"),
                    candidate.get("provenance_artifact_id"),
                    candidate.get("provenance_note"),
                    candidate.get("source_kind"),
                    candidate.get("phase0_entry_id"),
                    now,
                    now,
                ),
            )
            link_id = f"link-{uuid.uuid4().hex[:10]}"
            conn.execute(
                "INSERT INTO memory_links (id, from_memory_id, to_memory_id, relationship, created_at) VALUES (?,?,?,?,?)",
                (link_id, new_id, target["id"], "supersedes", now),
            )
            _upsert_fts(conn, new_id, candidate["title"], candidate["canonical_text"], candidate["memory_type"])
            _upsert_vec(conn, new_id, candidate["canonical_text"])
            conn.commit()
            return {"action": action, "id": new_id, "superseded": target["id"]}

        new_id = candidate.get("id") or f"mem-{uuid.uuid4().hex[:12]}"
        conn.execute(
            """
            INSERT INTO memories (
              id, memory_type, status, title, canonical_text, importance, confidence,
              origin, project_id, supersedes_id, why_failed,
              provenance_conversation_id, provenance_message_ids, provenance_artifact_id,
              provenance_note, source_kind, phase0_entry_id, created_at, updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                new_id,
                candidate["memory_type"],
                candidate.get("status", "active"),
                candidate["title"],
                candidate["canonical_text"],
                float(candidate.get("importance", 0.5)),
                float(candidate.get("confidence", 0.5)),
                candidate.get("origin", "agent_inferred"),
                candidate.get("project_id"),
                candidate.get("supersedes_id"),
                candidate.get("why_failed"),
                candidate.get("provenance_conversation_id"),
                candidate.get("provenance_message_ids"),
                candidate.get("provenance_artifact_id"),
                candidate.get("provenance_note"),
                candidate.get("source_kind"),
                candidate.get("phase0_entry_id"),
                now,
                now,
            ),
        )
        if candidate.get("status", "active") == "active":
            _upsert_fts(conn, new_id, candidate["title"], candidate["canonical_text"], candidate["memory_type"])
            _upsert_vec(conn, new_id, candidate["canonical_text"])
        conn.commit()
        return {"action": ACTION_CREATE, "id": new_id}
    finally:
        conn.close()
