"""Log every retrieval / pack lookup and attach later turn feedback."""
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
    request_id: Optional[str] = None,
    turn_id: Optional[str] = None,
    model_used: Optional[str] = None,
    user_accepted: Optional[bool] = None,
    user_corrected: Optional[bool] = None,
    helpfulness: Optional[str] = None,
    task_outcome: Optional[str] = None,
    feedback: Optional[Dict[str, Any]] = None,
    db_path=None,
) -> int:
    """
    Persist a retrieval event. request_id / turn_id let later feedback attach
    model used, injected memories, accept/correct, helpfulness, and task outcome.
    Full training pipeline is out of scope; the join keys and extensible fields exist now.
    """
    conn = ensure_schema(connect(db_path) if db_path else None)
    ctx = dict(context or {})
    if request_id:
        ctx.setdefault("request_id", request_id)
    if turn_id:
        ctx.setdefault("turn_id", turn_id)
    cur = conn.execute(
        """
        INSERT INTO retrieval_log (
          ts, query, expanded_queries_json, candidate_ids_json, scores_json,
          injected_ids_json, approx_token_cost, project_id, context_json,
          response_note, correction_note,
          request_id, turn_id, model_used, user_accepted, user_corrected,
          helpfulness, task_outcome, feedback_json
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
            json.dumps(ctx),
            response_note,
            correction_note,
            request_id,
            turn_id,
            model_used,
            None if user_accepted is None else int(bool(user_accepted)),
            None if user_corrected is None else int(bool(user_corrected)),
            helpfulness,
            task_outcome,
            json.dumps(feedback) if feedback is not None else None,
        ),
    )
    conn.commit()
    rid = int(cur.lastrowid)
    conn.close()
    return rid


def attach_retrieval_feedback(
    *,
    request_id: Optional[str] = None,
    turn_id: Optional[str] = None,
    log_id: Optional[int] = None,
    model_used: Optional[str] = None,
    user_accepted: Optional[bool] = None,
    user_corrected: Optional[bool] = None,
    helpfulness: Optional[str] = None,
    task_outcome: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    db_path=None,
) -> int:
    """Attach later-turn feedback to the retrieval_log row(s) for a request/turn."""
    if not any([request_id, turn_id, log_id]):
        raise ValueError("request_id, turn_id, or log_id is required")
    conn = ensure_schema(connect(db_path) if db_path else None)
    clauses = []
    args: list = []
    if log_id is not None:
        clauses.append("id=?")
        args.append(int(log_id))
    if request_id:
        clauses.append("request_id=?")
        args.append(request_id)
    if turn_id:
        clauses.append("turn_id=?")
        args.append(turn_id)
    where = " AND ".join(clauses)
    rows = conn.execute(f"SELECT id, feedback_json FROM retrieval_log WHERE {where}", args).fetchall()
    n = 0
    for row in rows:
        prev = {}
        if row["feedback_json"]:
            try:
                prev = json.loads(row["feedback_json"]) or {}
            except Exception:
                prev = {}
        if extra:
            prev.update(extra)
        conn.execute(
            """
            UPDATE retrieval_log SET
              model_used=COALESCE(?, model_used),
              user_accepted=COALESCE(?, user_accepted),
              user_corrected=COALESCE(?, user_corrected),
              helpfulness=COALESCE(?, helpfulness),
              task_outcome=COALESCE(?, task_outcome),
              feedback_json=?
            WHERE id=?
            """,
            (
                model_used,
                None if user_accepted is None else int(bool(user_accepted)),
                None if user_corrected is None else int(bool(user_corrected)),
                helpfulness,
                task_outcome,
                json.dumps(prev) if prev else row["feedback_json"],
                row["id"],
            ),
        )
        n += 1
    conn.commit()
    conn.close()
    return n
