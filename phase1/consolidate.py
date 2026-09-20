"""Semantic consolidation: vector + FTS + type/project before CREATE/MERGE/UPDATE/SUPERSEDE."""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .embed import get_embedder
from .privacy import normalize_flags
from .provenance import add_sources_from_candidate, link_memories
from .schema import connect, ensure_schema

ACTION_CREATE = "CREATE"
ACTION_MERGE = "MERGE"
ACTION_UPDATE = "UPDATE"
ACTION_SUPERSEDE = "SUPERSEDE"
ACTION_IGNORE = "IGNORE"
ACTION_CONFLICT = "CONFLICT"

# Conservative thresholds. SUPERSEDE is intentionally the strictest.
SEM_EXACT = 0.92
SEM_MERGE = 0.84
SEM_MERGE_SHARED = 0.72
SEM_SUPERSEDE = 0.86
SEM_CONFLICT_LO = 0.70
SEM_CONFLICT_HI = 0.84
SUPERSEDE_MIN_CONFIDENCE = 0.85


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _norm(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"\s+", " ", t)
    return t


def _token_set(text: str) -> set:
    return set(re.findall(r"[a-z0-9]{3,}", text.lower()))


def jaccard(a: str, b: str) -> float:
    """Cheap lexical overlap. Not the consolidation authority."""
    sa, sb = _token_set(a), _token_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _cosine_bytes(a: bytes, b: bytes) -> float:
    import numpy as np

    va = np.frombuffer(a, dtype=np.float32)
    vb = np.frombuffer(b, dtype=np.float32)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def _l2_to_cosine(dist: float) -> float:
    # Unit vectors: ||a-b||^2 = 2(1-cos)
    return max(0.0, 1.0 - (float(dist) ** 2) / 2.0)


_CONTENT_STOP = {
    "about", "after", "always", "because", "before", "being", "between",
    "local", "memory", "store", "using", "with", "that", "this", "from",
    "have", "will", "should", "would", "could", "their", "there", "where",
    "which", "while", "those", "these", "other", "into", "over", "under",
    "device", "system", "agent", "model", "external", "never", "please",
}


def _content_tokens(text: str) -> set:
    return {t for t in _token_set(text) if len(t) >= 5 and t not in _CONTENT_STOP}


def _mutual_exclusive_content(a: str, b: str) -> bool:
    """True when both statements carry distinct content words — likely different propositions."""
    sa, sb = _content_tokens(a), _content_tokens(b)
    return bool(sa - sb) and bool(sb - sa)


def _shared_content(a: str, b: str) -> bool:
    return bool(_content_tokens(a) & _content_tokens(b))


def _compatible_types(a: Optional[str], b: Optional[str]) -> bool:
    if not a or not b:
        return True
    if a == b:
        return True
    family = {
        "decision": {"decision", "correction", "preference", "constraint"},
        "correction": {"decision", "fact", "preference", "constraint", "correction"},
        "preference": {"preference", "constraint", "correction"},
        "constraint": {"constraint", "preference", "correction"},
        "fact": {"fact", "correction", "project_state"},
        "lesson": {"lesson", "outcome", "correction"},
        "outcome": {"outcome", "lesson"},
    }
    return b in family.get(a, {a})


def _same_project(cand: Dict[str, Any], row) -> bool:
    cp = cand.get("project_id")
    rp = row["project_id"] if row else None
    if not cp or not rp:
        return True
    return cp == rp


def _entity_overlap(cand: Dict[str, Any], row, conn) -> bool:
    ids = cand.get("entity_ids") or []
    if not ids:
        return True
    existing = {
        r["entity_id"]
        for r in conn.execute(
            "SELECT entity_id FROM memory_entities WHERE memory_id=?",
            (row["id"],),
        ).fetchall()
    }
    if not existing:
        return True
    return bool(set(ids) & existing)


def find_similar(
    conn,
    canonical_text: str,
    memory_type: Optional[str] = None,
    *,
    project_id: Optional[str] = None,
    entity_ids: Optional[List[str]] = None,
    limit: int = 8,
) -> List[Tuple[Any, Dict[str, float]]]:
    """Discover neighbors via vector + FTS; score primarily by semantic cosine."""
    cand_map: Dict[str, Any] = {}
    meta: Dict[str, Dict[str, float]] = {}

    def _touch(row, **scores):
        mid = row["id"]
        if mid not in cand_map:
            cand_map[mid] = row
            meta[mid] = {
                "semantic": 0.0,
                "fts": 0.0,
                "jaccard": 0.0,
                "type_match": 0.0,
                "project_match": 0.0,
                "combined": 0.0,
            }
        for k, v in scores.items():
            meta[mid][k] = max(meta[mid].get(k, 0.0), float(v))

    # Vector discovery
    try:
        emb = get_embedder().embed_query(canonical_text)
        fetch_n = max(limit * 4, 16)
        vec_rows = conn.execute(
            """
            SELECT v.memory_id AS memory_id, v.distance AS distance
            FROM memories_vec v
            WHERE v.embedding MATCH ? AND k = ?
            """,
            (emb, fetch_n),
        ).fetchall()
        for vr in vec_rows:
            row = conn.execute(
                "SELECT * FROM memories WHERE id=? AND status='active'",
                (vr["memory_id"],),
            ).fetchone()
            if not row:
                continue
            sem = _l2_to_cosine(float(vr["distance"]))
            if sem >= 0.28:
                _touch(row, semantic=sem)
    except Exception:
        pass

    # FTS discovery
    words = [w for w in re.findall(r"[A-Za-z0-9]{3,}", canonical_text)[:12]]
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
                (q, limit * 3),
            ).fetchall()
            for row in fts:
                _touch(row, fts=0.6)
        except Exception:
            pass

    # Re-embed neighbors we have for a precise cosine (handles FTS-only hits).
    try:
        embedder = get_embedder()
        qblob = embedder.embed_documents([canonical_text])[0]
        for mid, row in list(cand_map.items()):
            if meta[mid]["semantic"] > 0:
                continue
            tblob = embedder.embed_documents([row["canonical_text"]])[0]
            meta[mid]["semantic"] = _cosine_bytes(qblob, tblob)
    except Exception:
        pass

    scored: List[Tuple[float, Any, Dict[str, float]]] = []
    for mid, row in cand_map.items():
        jac = jaccard(canonical_text, row["canonical_text"])
        if _norm(canonical_text) == _norm(row["canonical_text"]):
            meta[mid]["semantic"] = max(meta[mid]["semantic"], 1.0)
            jac = 1.0
        meta[mid]["jaccard"] = jac
        meta[mid]["type_match"] = 1.0 if (not memory_type or row["memory_type"] == memory_type) else 0.0
        meta[mid]["project_match"] = 1.0 if (not project_id or row["project_id"] in (None, project_id)) else 0.0
        # Semantic is the authority; lexical/Jaccard are supporting signals only.
        combined = (
            0.72 * meta[mid]["semantic"]
            + 0.12 * max(meta[mid]["fts"], jac)
            + 0.08 * meta[mid]["type_match"]
            + 0.08 * meta[mid]["project_match"]
        )
        meta[mid]["combined"] = combined
        if combined >= 0.40 or meta[mid]["semantic"] >= 0.55:
            scored.append((combined, row, meta[mid]))
    scored.sort(key=lambda x: -x[0])
    return [(row, m) for _, row, m in scored[:limit]]


_POS_POLARITY = re.compile(r"\b(always|must|do)\b", re.I)
_NEG_POLARITY = re.compile(
    r"\b(never|don't|do not|must not|cannot|can't|no longer)\b", re.I
)
_CORRECTION_MARK = re.compile(
    r"\b(actually|that's wrong|that is wrong|correction|not true|instead|"
    r"no longer|supersede|forget that|ignore that)\b",
    re.I,
)


def _polarity(text: str) -> str:
    t = text or ""
    neg = bool(_NEG_POLARITY.search(t))
    pos = bool(_POS_POLARITY.search(t)) and not neg
    if pos and not neg:
        return "pos"
    if neg and not pos:
        return "neg"
    return "none"


def _polarity_conflict(a: str, b: str) -> bool:
    pa, pb = _polarity(a), _polarity(b)
    return {pa, pb} == {"pos", "neg"}


def _looks_like_correction(candidate: Dict[str, Any]) -> bool:
    if candidate.get("appears_to_correct"):
        return True
    if candidate.get("memory_type") == "correction":
        return True
    text = candidate.get("canonical_text") or ""
    return bool(_CORRECTION_MARK.search(text))


def _is_correction(candidate: Dict[str, Any]) -> bool:
    return _looks_like_correction(candidate)


def _same_topic(candidate: Dict[str, Any], row, meta: Dict[str, float]) -> bool:
    if meta.get("semantic", 0) >= 0.55:
        return True
    if meta.get("jaccard", 0) >= 0.35:
        return True
    return _shared_content(candidate.get("canonical_text") or "", row["canonical_text"] or "")


def _same_proposition(candidate: Dict[str, Any], row, meta: Dict[str, float], conn) -> bool:
    if meta.get("semantic", 0) < SEM_SUPERSEDE:
        return False
    if not _compatible_types(candidate.get("memory_type"), row["memory_type"]):
        return False
    if not _same_project(candidate, row):
        return False
    if not _entity_overlap(candidate, row, conn):
        return False
    return True


def decide_action(
    candidate: Dict[str, Any],
    similar: List[Tuple[Any, Dict[str, float]]],
    *,
    conn=None,
) -> Tuple[str, Optional[Any], Dict[str, float]]:
    """Return (action, target_row, scores). SUPERSEDE only when highly confident."""
    if not similar:
        return ACTION_CREATE, None, {}
    best, meta = similar[0]
    sem = meta.get("semantic", 0.0)
    jac = meta.get("jaccard", 0.0)

    polarity_flip = _polarity_conflict(candidate.get("canonical_text") or "", best["canonical_text"] or "")
    correction = _looks_like_correction(candidate) or polarity_flip
    # SUPERSEDE / CONFLICT always beat Jaccard-ish MERGE/IGNORE/UPDATE.
    if correction:
        conf = float(candidate.get("confidence", 0.5))
        same_topic = _same_topic(candidate, best, meta)
        if conn is not None:
            same_topic = same_topic or _same_proposition(candidate, best, meta, conn)
        if same_topic and (polarity_flip or _is_correction(candidate)):
            if conf >= SUPERSEDE_MIN_CONFIDENCE and (sem >= 0.55 or jac >= 0.35 or polarity_flip):
                return ACTION_SUPERSEDE, best, meta
            return ACTION_CONFLICT, best, meta
        if SEM_CONFLICT_LO <= sem < SEM_SUPERSEDE or (sem >= SEM_SUPERSEDE and conf < SUPERSEDE_MIN_CONFIDENCE):
            return ACTION_CONFLICT, best, meta
        return ACTION_CREATE, None, meta

    if sem >= SEM_EXACT or jac >= 0.97:
        if abs(float(candidate.get("importance", 0.5)) - float(best["importance"])) < 0.05:
            return ACTION_IGNORE, best, meta
        return ACTION_UPDATE, best, meta

    exclusive = _mutual_exclusive_content(candidate["canonical_text"], best["canonical_text"])
    shared = _shared_content(candidate["canonical_text"], best["canonical_text"])
    # High lexical overlap or shared content words + mid cosine = paraphrase,
    # even when wording differs enough that BGE scores ~0.74.
    paraphrase = (
        sem >= SEM_MERGE
        or (sem >= SEM_MERGE_SHARED and shared and not exclusive)
        or (jac >= 0.50 and sem >= 0.45 and not exclusive)
    )
    if paraphrase and _compatible_types(candidate.get("memory_type"), best["memory_type"]):
        if _same_project(candidate, best):
            if (
                candidate.get("memory_type") in ("decision", "preference", "constraint")
                and best["memory_type"] in ("decision", "preference", "constraint")
                and exclusive
            ):
                return ACTION_CONFLICT, best, meta
            return ACTION_MERGE, best, meta

    if (
        SEM_CONFLICT_LO <= sem < SEM_CONFLICT_HI
        and candidate.get("memory_type") in ("decision", "preference", "constraint")
        and best["memory_type"] in ("decision", "preference", "constraint")
    ):
        return ACTION_CONFLICT, best, meta

    return ACTION_CREATE, None, meta


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


def _attach_entities(conn, memory_id: str, entity_ids: Optional[List[str]]) -> None:
    for eid in entity_ids or []:
        try:
            conn.execute(
                "INSERT OR IGNORE INTO memory_entities (memory_id, entity_id) VALUES (?,?)",
                (memory_id, eid),
            )
        except Exception:
            continue


def _insert_memory(conn, new_id: str, candidate: Dict[str, Any], now: str, status: str = "active") -> None:
    flags = normalize_flags(candidate)
    conn.execute(
        """
        INSERT INTO memories (
          id, memory_type, status, title, canonical_text, importance, confidence,
          origin, project_id, supersedes_id, why_failed,
          provenance_conversation_id, provenance_message_ids, provenance_artifact_id,
          provenance_note, source_kind, phase0_entry_id, created_at, updated_at,
          remote_ok, local_only, sensitive, never_auto_retrieve, never_send_external_model,
          temporal_meaning, appears_to_correct
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            new_id,
            candidate["memory_type"],
            status,
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
            flags["remote_ok"],
            flags["local_only"],
            flags["sensitive"],
            flags["never_auto_retrieve"],
            flags["never_send_external_model"],
            candidate.get("temporal_meaning"),
            1 if candidate.get("appears_to_correct") else 0,
        ),
    )
    add_sources_from_candidate(conn, new_id, candidate, role="original")
    _attach_entities(conn, new_id, candidate.get("entity_ids"))
    if status == "active":
        _upsert_fts(conn, new_id, candidate["title"], candidate["canonical_text"], candidate["memory_type"])
        _upsert_vec(conn, new_id, candidate["canonical_text"])


def apply_memory(candidate: Dict[str, Any], *, db_path=None) -> Dict[str, Any]:
    """
    candidate keys: memory_type, title, canonical_text, importance, confidence,
    origin, project_id, provenance_*, supersedes_id, why_failed, source_kind,
    phase0_entry_id, privacy flags, appears_to_correct, entity_ids, temporal_meaning
    """
    conn = ensure_schema(connect(db_path) if db_path else None)
    try:
        similar = find_similar(
            conn,
            candidate["canonical_text"],
            candidate.get("memory_type"),
            project_id=candidate.get("project_id"),
            entity_ids=candidate.get("entity_ids"),
        )
        action, target, meta = decide_action(candidate, similar, conn=conn)
        now = _now()

        if action == ACTION_IGNORE and target is not None:
            # Reconfirm: keep the row, attach new provenance, do not drop old sources.
            add_sources_from_candidate(conn, target["id"], candidate, role="reconfirm")
            _attach_entities(conn, target["id"], candidate.get("entity_ids"))
            conn.execute(
                "UPDATE memories SET confidence=MAX(confidence, ?), updated_at=? WHERE id=?",
                (float(candidate.get("confidence", 0.5)), now, target["id"]),
            )
            conn.commit()
            return {"action": action, "id": target["id"], "reason": "near_duplicate", "semantic": meta.get("semantic")}

        if action == ACTION_UPDATE and target is not None:
            conn.execute(
                """
                UPDATE memories SET
                  title=?, canonical_text=?, importance=?, confidence=?,
                  origin=?, project_id=COALESCE(?, project_id),
                  provenance_conversation_id=COALESCE(?, provenance_conversation_id),
                  provenance_artifact_id=COALESCE(?, provenance_artifact_id),
                  provenance_note=COALESCE(?, provenance_note),
                  temporal_meaning=COALESCE(?, temporal_meaning),
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
                    candidate.get("provenance_artifact_id"),
                    candidate.get("provenance_note"),
                    candidate.get("temporal_meaning"),
                    now,
                    target["id"],
                ),
            )
            add_sources_from_candidate(conn, target["id"], candidate, role="update")
            _attach_entities(conn, target["id"], candidate.get("entity_ids"))
            _upsert_fts(
                conn,
                target["id"],
                candidate.get("title") or target["title"],
                candidate["canonical_text"],
                target["memory_type"],
            )
            _upsert_vec(conn, target["id"], candidate["canonical_text"])
            conn.commit()
            return {"action": action, "id": target["id"], "semantic": meta.get("semantic")}

        if action == ACTION_MERGE and target is not None:
            note = (target["provenance_note"] or "") + f" | merged:{candidate.get('title','')}"
            imp = max(float(target["importance"]), float(candidate.get("importance", 0.5)))
            conf = max(float(target["confidence"]), float(candidate.get("confidence", 0.5)))
            # Prefer the more concise atomic statement when both state the same thing.
            text_body = target["canonical_text"]
            cand_text = candidate["canonical_text"]
            if len(cand_text) + 8 < len(text_body) or (
                20 < len(cand_text) < len(text_body) and meta.get("semantic", 0) >= SEM_MERGE
            ):
                text_body = cand_text
            conn.execute(
                """
                UPDATE memories SET canonical_text=?, importance=?, confidence=?,
                  project_id=COALESCE(?, project_id),
                  provenance_note=?, updated_at=? WHERE id=?
                """,
                (text_body, imp, conf, candidate.get("project_id"), note[:2000], now, target["id"]),
            )
            add_sources_from_candidate(conn, target["id"], candidate, role="merge")
            _attach_entities(conn, target["id"], candidate.get("entity_ids"))
            _upsert_fts(conn, target["id"], target["title"], text_body, target["memory_type"])
            _upsert_vec(conn, target["id"], text_body)
            conn.commit()
            return {"action": action, "id": target["id"], "semantic": meta.get("semantic")}

        if action == ACTION_SUPERSEDE and target is not None:
            new_id = candidate.get("id") or f"mem-{uuid.uuid4().hex[:12]}"
            conn.execute(
                "UPDATE memories SET status='superseded', updated_at=? WHERE id=?",
                (now, target["id"]),
            )
            candidate = {**candidate, "supersedes_id": target["id"]}
            _insert_memory(conn, new_id, candidate, now, status="active")
            link_memories(conn, new_id, target["id"], "supersedes")
            conn.commit()
            return {"action": action, "id": new_id, "superseded": target["id"], "semantic": meta.get("semantic")}

        if action == ACTION_CONFLICT and target is not None:
            new_id = candidate.get("id") or f"mem-{uuid.uuid4().hex[:12]}"
            _insert_memory(conn, new_id, candidate, now, status="active")
            link_memories(conn, new_id, target["id"], "possible_conflict")
            conn.commit()
            return {
                "action": ACTION_CONFLICT,
                "id": new_id,
                "possible_conflict": target["id"],
                "semantic": meta.get("semantic"),
                "reason": "uncertain_preserve_both",
            }

        new_id = candidate.get("id") or f"mem-{uuid.uuid4().hex[:12]}"
        _insert_memory(conn, new_id, candidate, now, status=candidate.get("status", "active"))
        conn.commit()
        return {"action": ACTION_CREATE, "id": new_id}
    finally:
        conn.close()
