"""Rules-based extraction: useful candidates only; label origin/confidence."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import os

from .consolidate import apply_memory
from .schema import connect, ensure_schema


def _durable_md_paths() -> List[Path]:
    """Durable markdown paths for Layer B extraction.

    Set AGENTOMORY_DURABLE_MD to a colon-separated list of files, or place
    markdown under ~/.agentomory/durable/*.md. Defaults to empty (no personal paths).
    """
    env = os.environ.get("AGENTOMORY_DURABLE_MD")
    if env:
        return [Path(p) for p in env.split(":") if p.strip()]
    default_dir = Path.home() / ".agentomory" / "durable"
    if default_dir.is_dir():
        return sorted(default_dir.glob("*.md"))
    return []

DURABLE_MD: List[Path] = _durable_md_paths()

SKIP_MD_SUBSTRINGS = ("name-shortlist", "marketer-brief")

# User-turn patterns → high confidence
USER_RULES: List[Tuple[str, str, float, float]] = [
    # type, pattern, importance, confidence
    ("decision", r"\b(decided|decision|we(?:'| a)?re going with|lock(?:ed)? in|non-negotiable)\b", 0.85, 0.9),
    ("preference", r"\b(always|never|must|do not|don't|prefer|rule is|I want you to)\b", 0.8, 0.9),
    ("constraint", r"\b(must not|cannot|can't|out of scope|do not touch|no PR|don't commit)\b", 0.85, 0.9),
    ("goal", r"\b(goal is|by november|twenty (?:test )?users|priority)\b", 0.8, 0.85),
    ("task", r"\b(need to|please|open item|todo|follow up|rename|delete the)\b", 0.65, 0.75),
    ("open_question", r"\b(should we|what if|open question|still unclear|not sure)\b", 0.55, 0.7),
    ("correction", r"\b(that(?:'| i)?s wrong|actually|correction|not true|revert|supersede)\b", 0.9, 0.95),
    ("outcome", r"\b(it failed|that worked|lesson|we learned|hurt us)\b", 0.75, 0.8),
    ("lesson", r"\b(lesson|never again|don't repeat|what failed)\b", 0.8, 0.85),
    ("strategy", r"\b(approach|strategy|outer-loop|babysitter|story-bound)\b", 0.7, 0.75),
    ("fact", r"\b(path is|repo is|sqlite|Nylas|application-tracker)\b", 0.6, 0.8),
    ("project_state", r"\b(sprint 1|stage-?1|calendar strip|production (?:API )?blocker)\b", 0.7, 0.8),
]

# Assistant turns: only extract if strongly rule-like; lower confidence + agent_inferred
ASSISTANT_ALLOW = re.compile(
    r"\b(non-negotiable|standing rule|locked|must not|decision:|we agreed)\b",
    re.I,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _title_from(text: str) -> str:
    t = re.sub(r"\s+", " ", text.strip())
    return (t[:80] + "…") if len(t) > 80 else t


def classify_user_text(text: str) -> Optional[Tuple[str, float, float]]:
    for mtype, pat, imp, conf in USER_RULES:
        if re.search(pat, text, re.I):
            return mtype, imp, conf
    return None


def is_useful(text: str) -> bool:
    t = text.strip()
    if len(t) < 40:
        return False
    if len(t) > 1200:
        return False
    # skip pure greetings / ack
    if re.fullmatch(r"(ok|okay|thanks|thank you|yes|no|sure|got it)[.!]?", t, re.I):
        return False
    return True


def extract_from_messages(
    *, conversation_id: Optional[str] = None, db_path=None, limit: int = 500
) -> List[Dict[str, Any]]:
    conn = ensure_schema(connect(db_path) if db_path else None)
    if conversation_id:
        rows = conn.execute(
            "SELECT * FROM messages WHERE conversation_id=? ORDER BY seq",
            (conversation_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM messages ORDER BY conversation_id, seq LIMIT ?",
            (limit * 5,),
        ).fetchall()
    rows = [dict(r) for r in rows]
    conn.close()

    results = []
    for r in rows:
        text = r["text"]
        if not is_useful(text):
            continue
        role = r["role"]
        if role == "user":
            hit = classify_user_text(text)
            if not hit:
                continue
            mtype, imp, conf = hit
            origin = "user_stated"
        elif role == "assistant":
            if not ASSISTANT_ALLOW.search(text):
                continue
            hit = classify_user_text(text) or ("fact", 0.45, 0.4)
            mtype, imp, conf = hit
            conf = min(conf, 0.45)  # never treat agent as user-stated
            imp = min(imp, 0.6)
            origin = "agent_inferred"
        else:
            continue

        cand = {
            "id": f"ext-{_hash(r['id'] + mtype)}",
            "memory_type": mtype,
            "title": _title_from(text),
            "canonical_text": text.strip()[:800],
            "importance": imp,
            "confidence": conf,
            "origin": origin,
            "provenance_conversation_id": r["conversation_id"],
            "provenance_message_ids": json.dumps([r["id"]]),
            "provenance_note": f"extracted from {role} turn",
            "source_kind": "voice_call_message",
        }
        out = apply_memory(cand, db_path=db_path)
        results.append({**out, "type": mtype, "origin": origin})
        if len(results) >= limit:
            break
    return results


def register_artifact(path: Path, kind: str, *, db_path=None) -> str:
    aid = f"art-{_hash(str(path))}"
    now = _now()
    content = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    ch = _hash(content)
    conn = ensure_schema(connect(db_path) if db_path else None)
    try:
        conn.execute(
            """
            INSERT INTO artifacts (id, path, kind, title, content_hash, created_at, updated_at)
            VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(path) DO UPDATE SET
              content_hash=excluded.content_hash, updated_at=excluded.updated_at
            """,
            (aid, str(path), kind, path.name, ch, now, now),
        )
        conn.commit()
        row = conn.execute("SELECT id FROM artifacts WHERE path=?", (str(path),)).fetchone()
        return row["id"]
    finally:
        conn.close()



def _section_chunks(md: str) -> List[Tuple[str, str]]:
    """Split markdown into (heading, body) chunks."""
    parts = re.split(r"\n(?=##\s+)", md)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        lines = p.splitlines()
        if lines and lines[0].startswith("## "):
            title = lines[0][3:].strip()
            body = "\n".join(lines[1:]).strip()
        else:
            title = "intro"
            body = p
        if body:
            out.append((title, body))
    return out or [("document", md.strip()[:2000])]


def extract_from_durable_markdown(*, db_path=None) -> List[Dict[str, Any]]:
    results = []
    for path in _durable_md_paths():
        if not path.exists():
            continue
        if any(s in path.name for s in SKIP_MD_SUBSTRINGS):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        aid = register_artifact(path, "durable_md", db_path=db_path)
        for title, body in _section_chunks(text):
            # skip huge dumps; take useful paragraphs
            paras = [x.strip() for x in re.split(r"\n\n+", body) if x.strip()]
            for para in paras:
                if len(para) < 50:
                    continue
                if para.startswith("- [") or para.startswith("Updated:"):
                    continue
                hit = classify_user_text(para)
                # durable written record: treat as durable_record even if pattern weak
                if hit:
                    mtype, imp, conf = hit
                else:
                    # only keep if looks like a locked bullet / rule
                    if not re.search(r"^(?:- |\* |\d+\. )", para, re.M) and len(para) < 120:
                        continue
                    mtype, imp, conf = "fact", 0.55, 0.75
                conf = max(conf, 0.75)  # durable md is curated
                cand = {
                    "id": f"md-{_hash(str(path) + title + para[:80])}",
                    "memory_type": mtype,
                    "title": f"{path.stem}: {title}"[:80],
                    "canonical_text": para[:900],
                    "importance": imp,
                    "confidence": conf,
                    "origin": "durable_record",
                    "provenance_artifact_id": aid,
                    "provenance_conversation_id": None,
                    "provenance_message_ids": json.dumps([]),
                    "provenance_note": f"durable markdown {path.name} §{title}",
                    "source_kind": "durable_md",
                }
                out = apply_memory(cand, db_path=db_path)
                results.append({**out, "path": str(path), "type": mtype})
    return results
