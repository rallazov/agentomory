"""Multi-source provenance: one canonical memory, many supporting sources."""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional

from .schema import connect, ensure_schema


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _source_id(memory_id: str, source_kind: str, message_id, artifact_id, note: str) -> str:
    raw = f"{memory_id}|{source_kind}|{message_id or ''}|{artifact_id or ''}|{note or ''}"
    return "src-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def candidate_message_ids(candidate: Mapping[str, Any]) -> List[str]:
    raw = candidate.get("provenance_message_ids")
    if isinstance(raw, list):
        return [str(x) for x in raw if x]
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(x) for x in parsed if x]
        except Exception:
            return []
    mid = candidate.get("source_message_id")
    return [str(mid)] if mid else []


def add_sources_from_candidate(
    conn,
    memory_id: str,
    candidate: Mapping[str, Any],
    *,
    role: str = "supporting",
) -> List[str]:
    """Append sources for a candidate without replacing prior provenance rows."""
    now = _now()
    kind = candidate.get("source_kind") or "unknown"
    conv = candidate.get("provenance_conversation_id") or candidate.get("source_conversation_id")
    artifact = candidate.get("provenance_artifact_id")
    note = candidate.get("provenance_note")
    ids: List[str] = []
    msg_ids = candidate_message_ids(candidate)
    if not msg_ids and not artifact and not conv and not note:
        return []
    targets = msg_ids or [None]
    for mid in targets:
        sid = candidate.get("source_row_id") or _source_id(memory_id, kind, mid, artifact, note or "")
        conn.execute(
            """
            INSERT OR IGNORE INTO memory_sources (
              id, memory_id, source_kind, conversation_id, message_id,
              artifact_id, note, role, created_at
            ) VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (sid, memory_id, kind, conv, mid, artifact, note, role, now),
        )
        ids.append(sid)
    return ids


def list_sources(memory_id: str, *, db_path=None, conn=None) -> List[Dict[str, Any]]:
    own = conn is None
    if own:
        conn = ensure_schema(connect(db_path) if db_path else None)
    rows = conn.execute(
        "SELECT * FROM memory_sources WHERE memory_id=? ORDER BY created_at",
        (memory_id,),
    ).fetchall()
    out = [dict(r) for r in rows]
    if own:
        conn.close()
    return out


def link_memories(
    conn,
    from_id: str,
    to_id: str,
    relationship: str,
) -> str:
    lid = f"link-{uuid.uuid4().hex[:10]}"
    conn.execute(
        """
        INSERT INTO memory_links (id, from_memory_id, to_memory_id, relationship, created_at)
        VALUES (?,?,?,?,?)
        """,
        (lid, from_id, to_id, relationship, _now()),
    )
    return lid
