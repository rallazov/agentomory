"""Generic aliases / synonyms / project vocabulary stored in SQLite."""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .schema import connect, ensure_schema


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_term(term: str) -> str:
    return re.sub(r"\s+", " ", (term or "").strip().lower())


def _vid(kind: str, term: str, project_id: Optional[str], entity_id: Optional[str]) -> str:
    raw = f"{kind}|{normalize_term(term)}|{project_id or ''}|{entity_id or ''}"
    return "voc-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def upsert_term(
    *,
    term: str,
    kind: str,
    canonical: Optional[str] = None,
    project_id: Optional[str] = None,
    entity_id: Optional[str] = None,
    weight: float = 1.0,
    db_path=None,
    conn=None,
) -> str:
    """Insert or update a vocabulary row. Engine-agnostic: no hardcoded products."""
    own = conn is None
    if own:
        conn = ensure_schema(connect(db_path) if db_path else None)
    vid = _vid(kind, term, project_id, entity_id)
    conn.execute(
        """
        INSERT INTO vocabulary (
          id, kind, term, normalized, canonical, project_id, entity_id, weight, created_at
        ) VALUES (?,?,?,?,?,?,?,?,?)
        ON CONFLICT(id) DO UPDATE SET
          term=excluded.term,
          canonical=excluded.canonical,
          weight=excluded.weight
        """,
        (
            vid,
            kind,
            term.strip(),
            normalize_term(term),
            canonical,
            project_id,
            entity_id,
            float(weight),
            _now(),
        ),
    )
    if own:
        conn.commit()
        conn.close()
    return vid


def load_vocabulary(*, kind: Optional[str] = None, db_path=None, conn=None) -> List[Dict[str, Any]]:
    own = conn is None
    if own:
        conn = ensure_schema(connect(db_path) if db_path else None)
    if kind:
        rows = conn.execute("SELECT * FROM vocabulary WHERE kind=?", (kind,)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM vocabulary").fetchall()
    out = [dict(r) for r in rows]
    if own:
        conn.close()
    return out


def terms_matching_query(query: str, *, db_path=None, conn=None) -> List[Dict[str, Any]]:
    """Return vocabulary rows whose term/canonical appears in the query (generic)."""
    own = conn is None
    if own:
        conn = ensure_schema(connect(db_path) if db_path else None)
    q = normalize_term(query)
    rows = conn.execute("SELECT * FROM vocabulary").fetchall()
    hits: List[Dict[str, Any]] = []
    for r in rows:
        term = r["normalized"] or ""
        canon = normalize_term(r["canonical"] or "")
        if term and (term == q or (len(term) >= 3 and term in q)):
            hits.append(dict(r))
        elif canon and (canon == q or (len(canon) >= 3 and canon in q)):
            hits.append(dict(r))
    hits.sort(key=lambda x: -len(x.get("normalized") or ""))
    if own:
        conn.close()
    return hits


def expansions_for_query(query: str, *, db_path=None, conn=None) -> List[str]:
    """Generic query expansions from stored vocabulary. No product-specific code."""
    extra: List[str] = []
    seen = {normalize_term(query)}
    matches = terms_matching_query(query, db_path=db_path, conn=conn)
    project_ids = {m["project_id"] for m in matches if m.get("project_id")}
    entity_ids = {m["entity_id"] for m in matches if m.get("entity_id")}

    def _add(text: Optional[str]) -> None:
        if not text:
            return
        key = normalize_term(text)
        if key and key not in seen:
            seen.add(key)
            extra.append(text)

    for m in matches:
        _add(m.get("canonical"))
        _add(m.get("term"))

    if project_ids or entity_ids:
        own = conn is None
        if own:
            conn = ensure_schema(connect(db_path) if db_path else None)
        if project_ids:
            qmarks = ",".join("?" * len(project_ids))
            sibs = conn.execute(
                f"SELECT term, canonical FROM vocabulary WHERE project_id IN ({qmarks})",
                tuple(project_ids),
            ).fetchall()
            for s in sibs:
                _add(s["canonical"])
                _add(s["term"])
            names = conn.execute(
                f"SELECT name FROM projects WHERE id IN ({qmarks})",
                tuple(project_ids),
            ).fetchall()
            for n in names:
                _add(n["name"])
        if entity_ids:
            qmarks = ",".join("?" * len(entity_ids))
            ents = conn.execute(
                f"SELECT name FROM entities WHERE id IN ({qmarks})",
                tuple(entity_ids),
            ).fetchall()
            for e in ents:
                _add(e["name"])
        if own:
            conn.close()
    return extra
