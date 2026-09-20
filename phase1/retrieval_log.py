"""Log every retrieval / pack lookup."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .schema import connect, ensure_schema


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_retrieval(
    query: str,
    *,
    expanded_queries: Optional[List[str]] = None,
    candidate_ids: Optional[List[str]] = None,
    scores: Optional[Dict[str, float]] = None,
    injected_ids: Optional[List[str]] = None,
    approx_token_cost: int = 0,
    project_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    response_note: Optional[str] = None,
    correction_note: Optional[str] = None,
    db_path=None,
) -> int:
    conn = ensure_schema(connect(db_path) if db_path else None)
    cur = conn.execute(
        """
        INSERT INTO retrieval_log (
          ts, query, expanded_queries_json, candidate_ids_json, scores_json,
          injected_ids_json, approx_token_cost, project_id, context_json,
          response_note, correction_note
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            _now(),
            query,
            json.dumps(expanded_queries or []),
            json.dumps(candidate_ids or []),
            json.dumps(scores or {}),
            json.dumps(injected_ids or []),
            int(approx_token_cost),
            project_id,
            json.dumps(context or {}),
            response_note,
            correction_note,
        ),
    )
    conn.commit()
    rid = int(cur.lastrowid)
    conn.close()
    return rid
