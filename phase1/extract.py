"""Semantic/atomic extraction: regex is a cheap prefilter, not the final authority."""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .consolidate import apply_memory
from .schema import MEMORY_TYPES, connect, ensure_schema
from .vocabulary import terms_matching_query

# Cheap prefilter only — does this text *maybe* contain durable memory?
# Classification and canonicalization happen after splitting, not here.
_PREFILTER_CUES = re.compile(
    r"\b("
    r"decid(?:e|ed|ing|es)|decision|lock(?:ed)?\s+in|going with|"
    r"always|never|must(?:\s+not)?|do not|don't|prefer|rule|"
    r"cannot|can't|out of scope|constraint|"
    r"goal|priority|by\s+\w+|deadline|"
    r"need to|please|todo|follow up|"
    r"should we|open question|not sure|unclear|"
    r"wrong|actually|correction|not true|instead|no longer|revert|supersede|"
    r"failed|worked|lesson|learned|"
    r"approach|strategy|"
    r"we (?:will|are|use|chose|chose)|i want|i prefer"
    r")\b",
    re.I,
)

# Cue weights are *signals* for type scoring, never the sole authority.
_TYPE_CUES: List[Tuple[str, str, float]] = [
    ("correction", r"\b(that(?:'| i)?s wrong|actually|correction|not true|instead|no longer|revert|supersede|we changed)\b", 2.4),
    ("constraint", r"\b(must not|cannot|can't|out of scope|do not|don't|never|no longer allowed)\b", 2.0),
    ("preference", r"\b(always|prefer|standing rule|rule is|i want you to|please always)\b", 1.8),
    ("decision", r"\b(decid(?:e|ed|ing)|decision|locked in|going with|we(?:'| a)?re going with|chose|chosen)\b", 2.0),
    ("goal", r"\b(goal is|aim is|target is|by \w+|deadline|priority is)\b", 1.6),
    ("task", r"\b(need to|please|todo|follow up|open item)\b", 1.4),
    ("open_question", r"\b(should we|what if|open question|still unclear|not sure|unresolved)\b", 1.8),
    ("outcome", r"\b(it failed|that worked|we learned|hurt us|result was)\b", 1.6),
    ("lesson", r"\b(lesson|never again|don't repeat|what failed)\b", 1.7),
    ("strategy", r"\b(approach|strategy|playbook|we will handle)\b", 1.3),
    ("project_state", r"\b(currently|in progress|shipped|blocked|stage|status is)\b", 1.2),
    ("fact", r"\b(is|are|path|repo|uses|named|located)\b", 0.4),
]

_TYPE_PROTOTYPES: Dict[str, str] = {
    "decision": "A locked choice was made. We decided to use this option.",
    "preference": "A standing rule or preferred way of working. Always do this.",
    "constraint": "A hard restriction. Do not do this. This is out of scope.",
    "goal": "A target or deadline we are aiming for.",
    "task": "An action someone still needs to take.",
    "open_question": "An unresolved question that is still unclear.",
    "correction": "A previous belief was wrong and is being replaced.",
    "outcome": "Something we tried succeeded or failed.",
    "lesson": "A durable lesson from a failure so we do not repeat it.",
    "strategy": "An approach we will use for this class of problem.",
    "project_state": "Current status of a project or workstream.",
    "fact": "A stable factual statement about the system or world.",
}

_FILLER_LEAD = re.compile(
    r"^(?:(?:also|and also|and|please|oh|well|look|so|anyway|by the way|plus)[,:]?\s+)+",
    re.I,
)
_THINK_LEAD = re.compile(
    r"^(?:i (?:think|believe|feel|guess|suspect)|we think)(?: that)?\s+",
    re.I,
)
_DECISION_LEAD = re.compile(
    r"^(?:we (?:have )?(?:decided|locked in|chose)|decision(?: is|:)|"
    r"we(?:'re| are) going with|let's go with|we will use|locked choice(?: is|:)?)\s+",
    re.I,
)
_PREF_LEAD = re.compile(
    r"^(?:i (?:want you to|prefer(?: that)?|would like you to)|please always|the rule is)\s+",
    re.I,
)
_CONSTRAINT_LEAD = re.compile(
    r"^(?:you must not|we must not|do not|don't|never|please (?:do not|don't|never))\s+",
    re.I,
)

_TEMPORAL = [
    (r"\bby (?:end of )?(?:january|february|march|april|may|june|july|august|september|october|november|december|\d{1,2}(?:/\d{1,2})?(?:/\d{2,4})?|\d{4})\b", "deadline"),
    (r"\bas of (?:today|now|[A-Za-z]+ \d{1,2})\b", "as_of"),
    (r"\b(?:no longer|not anymore|formerly|used to)\b", "ended"),
    (r"\b(?:starting|from) (?:today|now|next \w+|this \w+)\b", "start"),
    (r"\b(?:yesterday|today|tomorrow|last week|next week|this week|next month)\b", "relative"),
]

_CORRECTION = re.compile(
    r"\b(that(?:'| i)?s wrong|actually|correction|not true|instead|no longer|"
    r"we (?:changed|reverted)|supersede|forget that|ignore that)\b",
    re.I,
)

SKIP_MD_SUBSTRINGS = ("name-shortlist", "marketer-brief")
ASSISTANT_ALLOW = re.compile(
    r"\b(non-negotiable|standing rule|locked|must not|decision:|we agreed)\b",
    re.I,
)


def _durable_md_paths() -> List[Path]:
    env = os.environ.get("AGENTOMORY_DURABLE_MD")
    if env:
        return [Path(p) for p in env.split(":") if p.strip()]
    default_dir = Path.home() / ".agentomory" / "durable"
    if default_dir.is_dir():
        return sorted(default_dir.glob("*.md"))
    return []


DURABLE_MD: List[Path] = _durable_md_paths()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _title_from(text: str) -> str:
    t = re.sub(r"\s+", " ", text.strip())
    return (t[:80] + "…") if len(t) > 80 else t


def looks_memory_worthy(text: str) -> bool:
    """Cheap prefilter. True means 'worth splitting', not 'this string is a memory'."""
    t = (text or "").strip()
    if len(t) < 12:
        return False
    if re.fullmatch(r"(ok|okay|thanks|thank you|yes|no|sure|got it|hi|hello)[.!]?", t, re.I):
        return False
    if len(t) > 8000:
        return False
    if _PREFILTER_CUES.search(t):
        return True
    # Longer statements can still be durable facts even without cue words.
    return len(t) >= 48 and bool(re.search(r"\b(is|are|will|should|use|uses)\b", t, re.I))


def _looks_like_claim(text: str) -> bool:
    t = text.strip()
    if len(t) < 12:
        return False
    return bool(re.search(r"\b([a-z]{3,})\b", t, re.I))


def split_propositions(text: str) -> List[str]:
    """Split one message into independent candidate units."""
    text = (text or "").strip()
    if not text:
        return []
    chunks: List[str] = []
    if re.search(r"(?m)^\s*(?:[-*]|\d+[.)])\s+", text):
        parts = re.split(r"(?m)^\s*(?:[-*]|\d+[.)])\s+", text)
        chunks = [p.strip() for p in parts if p.strip()]
    if not chunks:
        chunks = [text]

    out: List[str] = []
    sent_split = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])|(?:\n{2,})")
    for chunk in chunks:
        for sent in sent_split.split(chunk):
            sent = sent.strip()
            if not sent:
                continue
            bits = re.split(r"\s+(?:and also|additionally,|;)\s+", sent, flags=re.I)
            expanded: List[str] = []
            for bit in bits:
                and_bits = re.split(
                    r"\s+and\s+(?=(?:we|i|do|don't|never|always|must|please)\b)",
                    bit,
                    flags=re.I,
                )
                if len(and_bits) > 1 and all(_looks_like_claim(b) for b in and_bits):
                    expanded.extend(and_bits)
                else:
                    expanded.append(bit)
            out.extend(b.strip(" \t-") for b in expanded if b.strip())
    # Drop leftover greetings
    return [x for x in out if looks_memory_worthy(x) or (len(x) >= 16 and _looks_like_claim(x))]


def extract_temporal(text: str) -> Optional[str]:
    hits = []
    for pat, kind in _TEMPORAL:
        m = re.search(pat, text, re.I)
        if m:
            hits.append({"kind": kind, "text": m.group(0)})
    if not hits:
        return None
    return json.dumps(hits[0])


def appears_to_correct(text: str) -> bool:
    return bool(_CORRECTION.search(text or ""))


def canonicalize(text: str, memory_type: str) -> str:
    """Concise atomic statement — never a truncated dump of the original message."""
    t = re.sub(r"\s+", " ", (text or "").strip())
    t = t.strip("\"'`")
    t = _FILLER_LEAD.sub("", t)
    t = _THINK_LEAD.sub("", t)
    if memory_type == "decision":
        t = _DECISION_LEAD.sub("", t)
        t = re.sub(r"^to\s+", "", t, flags=re.I)
    elif memory_type == "preference":
        t = _PREF_LEAD.sub("", t)
    elif memory_type == "constraint":
        t = _CONSTRAINT_LEAD.sub("", t)
        if t and not re.match(r"^(?:do not|don't|never|must not)\b", t, re.I):
            t = "Do not " + t[0].lower() + t[1:] if t[:1].isupper() else "Do not " + t
    elif memory_type == "correction":
        t = re.sub(r"^(?:actually[,:]?\s+|that's wrong[,:]?\s+|correction[,:]?\s+)", "", t, flags=re.I)
    t = re.sub(r"\s*(?:thanks|thank you|ok|okay)[.!]?\s*$", "", t, flags=re.I)
    t = t.strip(" ,;")
    if not t:
        t = text.strip()
    # If still huge, keep the cue-bearing clause, not a raw slice of the user dump.
    if len(t) > 240:
        clauses = re.split(r"[,;] ", t)
        scored = sorted(clauses, key=lambda c: (1 if _PREFILTER_CUES.search(c) else 0, -len(c)), reverse=True)
        t = scored[0] if scored else t[:240]
    if t and t[0].islower():
        t = t[0].upper() + t[1:]
    if t and t[-1] not in ".!?":
        t += "."
    return t


def _cue_scores(text: str) -> Dict[str, float]:
    scores = {t: 0.0 for t in MEMORY_TYPES}
    for mtype, pat, weight in _TYPE_CUES:
        if re.search(pat, text, re.I):
            scores[mtype] += weight
    if text.strip().endswith("?"):
        scores["open_question"] += 1.5
    if appears_to_correct(text):
        scores["correction"] += 1.2
    return scores


def _prototype_scores(text: str) -> Dict[str, float]:
    """Embedding similarity to generic type prototypes when an embedder is available."""
    scores = {t: 0.0 for t in MEMORY_TYPES}
    try:
        from .embed import get_embedder
        import numpy as np

        emb = get_embedder()
        q = emb.embed_documents([text])[0]
        proto_texts = [_TYPE_PROTOTYPES[t] for t in MEMORY_TYPES]
        vecs = emb.embed_documents(proto_texts)
        qv = np.frombuffer(q, dtype=np.float32)
        qn = float(np.linalg.norm(qv)) or 1.0
        for t, blob in zip(MEMORY_TYPES, vecs):
            pv = np.frombuffer(blob, dtype=np.float32)
            pn = float(np.linalg.norm(pv)) or 1.0
            scores[t] = float(np.dot(qv, pv) / (qn * pn))
    except Exception:
        pass
    return scores


def classify_proposition(text: str) -> Tuple[str, float, float]:
    """Multi-signal type: cue weights + prototypes + structure. Regex is not final authority."""
    cues = _cue_scores(text)
    protos = _prototype_scores(text)
    combined: Dict[str, float] = {}
    for t in MEMORY_TYPES:
        combined[t] = 0.55 * cues[t] + 1.1 * max(0.0, protos[t] - 0.15)
    # Slight prior toward fact when nothing else fires.
    if max(cues.values()) == 0:
        combined["fact"] += 0.35
    mtype = max(combined, key=combined.get)
    strength = combined[mtype]
    if strength < 0.35:
        mtype = "fact"
        strength = 0.4
    importance = {
        "correction": 0.9,
        "constraint": 0.85,
        "decision": 0.85,
        "preference": 0.8,
        "goal": 0.8,
        "lesson": 0.8,
        "outcome": 0.75,
        "project_state": 0.7,
        "strategy": 0.7,
        "task": 0.65,
        "fact": 0.6,
        "open_question": 0.55,
    }.get(mtype, 0.55)
    if re.search(r"\b(non-negotiable|must not|never|hard rule)\b", text, re.I):
        importance = min(0.95, importance + 0.08)
    confidence = min(0.93, 0.62 + 0.12 * min(strength, 2.5))
    return mtype, importance, confidence


def associate_context(text: str, *, conn=None, db_path=None) -> Dict[str, Any]:
    project_id = None
    entity_ids: List[str] = []
    entity_names: List[str] = []
    if conn is None and db_path is None:
        return {"project_id": None, "entity_ids": [], "entity_names": []}
    try:
        hits = terms_matching_query(text, conn=conn, db_path=db_path)
    except Exception:
        hits = []
    for h in hits:
        if h.get("project_id") and not project_id:
            project_id = h["project_id"]
        if h.get("entity_id"):
            entity_ids.append(h["entity_id"])
            if h.get("term"):
                entity_names.append(h["term"])
    return {
        "project_id": project_id,
        "entity_ids": list(dict.fromkeys(entity_ids)),
        "entity_names": list(dict.fromkeys(entity_names)),
    }


@dataclass
class MemoryCandidate:
    memory_type: str
    canonical_text: str
    title: str
    importance: float
    confidence: float
    origin: str
    explicit: bool
    appears_to_correct: bool
    source_message_id: Optional[str] = None
    source_conversation_id: Optional[str] = None
    project_id: Optional[str] = None
    entity_ids: List[str] = field(default_factory=list)
    entity_names: List[str] = field(default_factory=list)
    temporal_meaning: Optional[str] = None
    provenance_note: Optional[str] = None
    source_kind: str = "voice_call_message"
    provenance_artifact_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": f"ext-{_hash((self.source_message_id or '') + self.memory_type + self.canonical_text)}",
            "memory_type": self.memory_type,
            "title": self.title,
            "canonical_text": self.canonical_text,
            "importance": self.importance,
            "confidence": self.confidence,
            "origin": self.origin,
            "explicit": self.explicit,
            "appears_to_correct": self.appears_to_correct,
            "source_message_id": self.source_message_id,
            "source_conversation_id": self.source_conversation_id,
            "provenance_conversation_id": self.source_conversation_id,
            "provenance_message_ids": json.dumps(
                [self.source_message_id] if self.source_message_id else []
            ),
            "project_id": self.project_id,
            "entity_ids": self.entity_ids,
            "entity_names": self.entity_names,
            "temporal_meaning": self.temporal_meaning,
            "provenance_note": self.provenance_note,
            "source_kind": self.source_kind,
            "provenance_artifact_id": self.provenance_artifact_id,
        }


def extract_candidates(
    text: str,
    *,
    role: str = "user",
    message_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    source_kind: str = "voice_call_message",
    artifact_id: Optional[str] = None,
    provenance_note: Optional[str] = None,
    conn=None,
    db_path=None,
) -> List[Dict[str, Any]]:
    """
    Produce structured atomic candidates from one message.
    One user message may yield multiple independent memories.
    """
    if not looks_memory_worthy(text):
        return []
    if role == "assistant" and not ASSISTANT_ALLOW.search(text):
        return []

    units = split_propositions(text)
    if not units:
        units = [text.strip()]

    origin = "user_stated" if role == "user" else "agent_inferred"
    if source_kind == "durable_md":
        origin = "durable_record"
    explicit = origin in ("user_stated", "durable_record")

    out: List[Dict[str, Any]] = []
    seen_canon = set()
    for unit in units:
        if len(unit) < 12:
            continue
        mtype, imp, conf = classify_proposition(unit)
        if role == "assistant":
            conf = min(conf, 0.45)
            imp = min(imp, 0.6)
            origin = "agent_inferred"
            explicit = False
        if source_kind == "durable_md":
            conf = max(conf, 0.75)
        canon = canonicalize(unit, mtype)
        # Reject "canonical == truncated original" style dumps.
        if canon.strip() == text.strip() and len(text) > 160 and len(units) == 1:
            # Try a tighter rewrite from the first sentence-like unit.
            canon = canonicalize(unit[:240], mtype)
        key = re.sub(r"\s+", " ", canon.lower())
        if key in seen_canon:
            continue
        seen_canon.add(key)
        assoc = associate_context(unit, conn=conn, db_path=db_path)
        cand = MemoryCandidate(
            memory_type=mtype,
            canonical_text=canon,
            title=_title_from(canon),
            importance=imp,
            confidence=conf,
            origin=origin,
            explicit=explicit,
            appears_to_correct=appears_to_correct(unit) or mtype == "correction",
            source_message_id=message_id,
            source_conversation_id=conversation_id,
            project_id=assoc.get("project_id"),
            entity_ids=assoc.get("entity_ids") or [],
            entity_names=assoc.get("entity_names") or [],
            temporal_meaning=extract_temporal(unit),
            provenance_note=provenance_note or f"extracted from {role} turn",
            source_kind=source_kind,
            provenance_artifact_id=artifact_id,
        )
        out.append(cand.to_dict())
    return out


def classify_user_text(text: str) -> Optional[Tuple[str, float, float]]:
    """Backward-compatible single-label helper. Prefer extract_candidates."""
    if not looks_memory_worthy(text):
        return None
    return classify_proposition(text)


def is_useful(text: str) -> bool:
    return looks_memory_worthy(text)


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
        cands = extract_candidates(
            r["text"],
            role=r["role"],
            message_id=r["id"],
            conversation_id=r["conversation_id"],
            db_path=db_path,
        )
        for cand in cands:
            out = apply_memory(cand, db_path=db_path)
            results.append({**out, "type": cand["memory_type"], "origin": cand["origin"], "candidate": cand})
            if len(results) >= limit:
                return results
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
            paras = [x.strip() for x in re.split(r"\n\n+", body) if x.strip()]
            for para in paras:
                if len(para) < 40:
                    continue
                if para.startswith("- [") or para.startswith("Updated:"):
                    continue
                cands = extract_candidates(
                    para,
                    role="user",
                    source_kind="durable_md",
                    artifact_id=aid,
                    provenance_note=f"durable markdown {path.name} §{title}",
                    db_path=db_path,
                )
                if not cands and looks_memory_worthy(para):
                    # curated durable record: keep a single atomic fact if split yielded nothing
                    mtype, imp, conf = classify_proposition(para)
                    cands = [
                        MemoryCandidate(
                            memory_type=mtype,
                            canonical_text=canonicalize(para, mtype),
                            title=f"{path.stem}: {title}"[:80],
                            importance=imp,
                            confidence=max(conf, 0.75),
                            origin="durable_record",
                            explicit=True,
                            appears_to_correct=appears_to_correct(para),
                            temporal_meaning=extract_temporal(para),
                            provenance_note=f"durable markdown {path.name} §{title}",
                            source_kind="durable_md",
                            provenance_artifact_id=aid,
                        ).to_dict()
                    ]
                for cand in cands:
                    cand["title"] = cand.get("title") or f"{path.stem}: {title}"[:80]
                    out = apply_memory(cand, db_path=db_path)
                    results.append({**out, "path": str(path), "type": cand["memory_type"]})
    return results
