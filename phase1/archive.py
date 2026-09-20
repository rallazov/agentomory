"""Layer A: import voice-call JSON as immutable conversations + messages."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import os

from .schema import connect, ensure_schema


def _default_calls_dir() -> Path:
    """Voice-call JSON directory. Override with AGENTOMORY_CALLS_DIR."""
    env = os.environ.get("AGENTOMORY_CALLS_DIR")
    if env:
        return Path(env)
    return Path.home() / ".agentomory" / "voice-calls"

CEO_CALLS = _default_calls_dir()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ms_to_iso(ms: Optional[int]) -> Optional[str]:
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat()


def _role(speaker: str) -> str:
    s = (speaker or "").lower()
    if s in ("user", "human", "ramin"):
        return "user"
    if s in ("assistant", "agent", "ceo", "bot"):
        return "assistant"
    if s == "system":
        return "system"
    if s == "tool":
        return "tool"
    return "assistant" if s else "user"


def import_call_file(path: Path, *, db_path=None, force: bool = False) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    call_id = data.get("callId") or path.stem
    conn = ensure_schema(connect(db_path) if db_path else None)

    existing = conn.execute(
        "SELECT id FROM conversations WHERE id = ?", (call_id,)
    ).fetchone()
    if existing and not force:
        conn.close()
        return {"call_id": call_id, "skipped": True, "reason": "already_archived"}

    if existing and force:
        # Immutability: do not rewrite messages; only allow re-import if empty
        n = conn.execute(
            "SELECT COUNT(*) AS c FROM messages WHERE conversation_id = ?", (call_id,)
        ).fetchone()["c"]
        if n:
            conn.close()
            return {"call_id": call_id, "skipped": True, "reason": "messages_immutable"}

    meta = {
        "model": data.get("model"),
        "realtimeConversationId": data.get("realtimeConversationId"),
        "ending": data.get("ending"),
        "durationMs": data.get("durationMs"),
    }
    conn.execute(
        """
        INSERT OR REPLACE INTO conversations (
          id, source_kind, source_path, agent_id, started_at, ended_at, meta_json, created_at
        ) VALUES (?,?,?,?,?,?,?,?)
        """,
        (
            call_id,
            "voice_call_json",
            str(path),
            data.get("agentId"),
            _ms_to_iso(data.get("startedAtMs")),
            _ms_to_iso(data.get("endedAtMs")),
            json.dumps(meta),
            _now(),
        ),
    )

    turns: List[Dict[str, Any]] = data.get("turns") or []
    inserted = 0
    for i, turn in enumerate(turns):
        mid = turn.get("id") or f"{call_id}-msg-{i}"
        text = turn.get("text") or ""
        if not text.strip():
            continue
        try:
            conn.execute(
                """
                INSERT INTO messages (
                  id, conversation_id, seq, role, speaker, text, at_ms, created_at
                ) VALUES (?,?,?,?,?,?,?,?)
                """,
                (
                    mid,
                    call_id,
                    i,
                    _role(turn.get("speaker") or "user"),
                    turn.get("speaker"),
                    text,
                    turn.get("atMs"),
                    _now(),
                ),
            )
            inserted += 1
        except Exception as e:
            # unique violation = already present; keep immutable
            if "UNIQUE" in str(e).upper():
                continue
            raise

    conn.execute(
        """
        INSERT OR REPLACE INTO processed_sources (
          source_id, source_kind, path, content_hash, processed_at, notes
        ) VALUES (?,?,?,?,?,?)
        """,
        (call_id, "voice_call_json", str(path), None, _now(), f"messages={inserted}"),
    )
    conn.execute(
        "INSERT INTO ingest_log (source_id, source_path, processed_at, action, notes) VALUES (?,?,?,?,?)",
        (call_id, str(path), _now(), "archive_import", f"messages={inserted}"),
    )
    conn.commit()
    conn.close()
    return {"call_id": call_id, "skipped": False, "messages": inserted}


def import_all_calls(
    directory: Path | None = None, *, db_path=None
) -> List[Dict[str, Any]]:
    d = directory or CEO_CALLS
    results = []
    if not d.exists():
        return [{"error": f"missing {d}"}]
    for path in sorted(d.glob("call-*.json")):
        results.append(import_call_file(path, db_path=db_path))
    return results
