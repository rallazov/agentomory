"""Eval: Precision@K, Recall@K, top-1/3, rates, token cost, zero-result, adversarial cases."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .embed import get_embedder
from .memory_pack import build_pack
from .retrieve import DEFAULT_MIN_SCORE
from .schema import connect, ensure_schema
from .vocabulary import upsert_term

EVAL_PATH = Path(__file__).resolve().parent / "eval_set.json"
RESULTS_PATH = Path(__file__).resolve().parent / "eval_results.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_eval_document() -> Dict[str, Any]:
    if not EVAL_PATH.exists():
        return {"fixture": {}, "cases": []}
    raw = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return {"fixture": {}, "cases": raw}
    return raw


def ensure_eval_set() -> List[Dict[str, Any]]:
    return load_eval_document().get("cases") or []


def seed_eval_fixture(db_path, fixture: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
    """Insert known memories/projects/vocab so metrics can use expected IDs."""
    doc = fixture if fixture is not None else load_eval_document().get("fixture") or {}
    conn = ensure_schema(connect(db_path))
    now = _now()
    n_proj = n_ent = n_mem = n_voc = 0
    try:
        for p in doc.get("projects") or []:
            conn.execute(
                """
                INSERT OR REPLACE INTO projects (
                  id, name, purpose, current_goal, current_stage, key_decisions,
                  constraints, open_questions, next_actions, current_summary,
                  status, source_artifact_id, created_at, updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    p["id"],
                    p["name"],
                    p.get("purpose"),
                    p.get("current_goal"),
                    p.get("current_stage"),
                    p.get("key_decisions"),
                    p.get("constraints"),
                    p.get("open_questions"),
                    p.get("next_actions"),
                    p.get("current_summary"),
                    p.get("status", "active"),
                    None,
                    now,
                    now,
                ),
            )
            n_proj += 1
        for e in doc.get("entities") or []:
            conn.execute(
                """
                INSERT OR REPLACE INTO entities (id, name, entity_type, meta_json, created_at)
                VALUES (?,?,?,?,?)
                """,
                (e["id"], e["name"], e.get("entity_type", "thing"), e.get("meta_json"), now),
            )
            n_ent += 1
        conn.commit()
    finally:
        conn.close()

    for v in doc.get("vocabulary") or []:
        upsert_term(db_path=db_path, **v)
        n_voc += 1

    conn = ensure_schema(connect(db_path))
    try:
        for m in doc.get("memories") or []:
            status = m.get("status", "active")
            why = m.get("why_failed")
            if status == "reversed" and not why:
                why = "eval reversed"
            conn.execute(
                """
                INSERT OR REPLACE INTO memories (
                  id, memory_type, status, title, canonical_text, importance, confidence,
                  origin, project_id, supersedes_id, why_failed, provenance_note,
                  source_kind, created_at, updated_at,
                  remote_ok, local_only, sensitive, never_auto_retrieve,
                  never_send_external_model, temporal_meaning, appears_to_correct
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    m["id"],
                    m.get("memory_type", "fact"),
                    status,
                    m.get("title") or m["canonical_text"][:80],
                    m["canonical_text"],
                    float(m.get("importance", 0.7)),
                    float(m.get("confidence", 0.85)),
                    m.get("origin", "user_stated"),
                    m.get("project_id"),
                    m.get("supersedes_id"),
                    why,
                    m.get("provenance_note", "eval fixture"),
                    m.get("source_kind", "eval_fixture"),
                    now,
                    now,
                    int(m.get("remote_ok", 1)),
                    int(m.get("local_only", 0)),
                    int(m.get("sensitive", 0)),
                    int(m.get("never_auto_retrieve", 0)),
                    int(m.get("never_send_external_model", 0)),
                    m.get("temporal_meaning"),
                    1 if m.get("appears_to_correct") else 0,
                ),
            )
            if status == "active":
                conn.execute(
                    "INSERT INTO memories_fts (memory_id, title, canonical_text, memory_type) VALUES (?,?,?,?)",
                    (
                        m["id"],
                        m.get("title") or m["canonical_text"][:80],
                        m["canonical_text"],
                        m.get("memory_type", "fact"),
                    ),
                )
                blob = get_embedder().embed_documents([m["canonical_text"]])[0]
                conn.execute("DELETE FROM memories_vec WHERE memory_id=?", (m["id"],))
                conn.execute(
                    "INSERT INTO memories_vec (memory_id, embedding) VALUES (?, ?)",
                    (m["id"], blob),
                )
            n_mem += 1
        conn.commit()
    finally:
        conn.close()
    return {"projects": n_proj, "entities": n_ent, "memories": n_mem, "vocabulary": n_voc}


def _relevant_ids(case: Dict[str, Any], pack: Dict[str, Any]) -> Set[str]:
    ids: Set[str] = set(case.get("expect_ids") or [])
    props = [p.lower() for p in (case.get("expect_propositions") or [])]
    if props:
        for m in pack.get("memories") or []:
            blob = (m.get("title", "") + " " + m.get("canonical_text", "")).lower()
            if any(p in blob for p in props):
                ids.add(m["id"])
            else:
                # proposition tokens overlap
                for p in props:
                    pt = set(re.findall(r"[a-z0-9]{3,}", p))
                    bt = set(re.findall(r"[a-z0-9]{3,}", blob))
                    if pt and len(pt & bt) / len(pt) >= 0.7:
                        ids.add(m["id"])
    return ids


def _score_one(case: Dict[str, Any], pack: Dict[str, Any], conn) -> Dict[str, Any]:
    memories = pack.get("memories") or []
    retrieved = [m["id"] for m in memories]
    k = int(case.get("k") or max(len(retrieved), 3))
    expect_empty = bool(case.get("expect_empty"))
    relevant = set(case.get("expect_ids") or [])
    if not relevant and case.get("expect_propositions"):
        relevant = _relevant_ids(case, pack)

    topk = retrieved[:k]
    if expect_empty:
        precision = 1.0 if not topk else 0.0
        recall = 1.0 if not topk else 0.0
        top1 = not retrieved
        top3 = not retrieved
        zero_ok = not retrieved
    else:
        inter = set(topk) & relevant if relevant else set()
        precision = (len(inter) / len(topk)) if topk else 0.0
        recall = (len(inter) / len(relevant)) if relevant else (1.0 if not topk else 0.0)
        top1 = bool(retrieved) and retrieved[0] in relevant if relevant else False
        top3 = bool(set(retrieved[:3]) & relevant) if relevant else False
        zero_ok = None

        # substring fallback when no IDs (live personal DB)
        if not relevant and case.get("expect_any_substrings"):
            texts = " ".join(m["title"] + " " + m["canonical_text"] for m in memories)
            hit_subs = [s for s in case["expect_any_substrings"] if s.lower() in texts.lower()]
            right = len(hit_subs) > 0
            precision = 1.0 if right else 0.0
            recall = 1.0 if right else 0.0
            top1 = right
            top3 = right

    irrelevant_n = 0
    if relevant:
        irrelevant_n = sum(1 for i in retrieved if i not in relevant)
    elif case.get("expect_any_substrings"):
        texts = " ".join(m["title"] + " " + m["canonical_text"] for m in memories).lower()
        if memories and not any(s.lower() in texts for s in case["expect_any_substrings"]):
            irrelevant_n = len(memories)
    irrelevant_rate = (irrelevant_n / len(retrieved)) if retrieved else (0.0 if expect_empty else 0.0)

    expected_project = case.get("expect_project") or case.get("project_hint")
    wrong_project = 0
    if expected_project:
        for m in memories:
            row = conn.execute("SELECT project_id FROM memories WHERE id=?", (m["id"],)).fetchone()
            if row and row["project_id"] and row["project_id"] != expected_project:
                wrong_project += 1
    wrong_project_rate = (wrong_project / len(retrieved)) if retrieved else 0.0

    stale = 0
    for mid in retrieved:
        row = conn.execute("SELECT status FROM memories WHERE id=?", (mid,)).fetchone()
        if row and row["status"] in ("superseded", "reversed"):
            stale += 1
    stale_rate = (stale / len(retrieved)) if retrieved else 0.0

    contradictions = 0
    retrieved_set = set(retrieved)
    for mid in retrieved:
        links = conn.execute(
            """
            SELECT to_memory_id, relationship FROM memory_links
            WHERE from_memory_id=? AND relationship IN ('possible_conflict','contradicts','supersedes')
            """,
            (mid,),
        ).fetchall()
        for lk in links:
            if lk["to_memory_id"] in retrieved_set:
                contradictions += 1
    contradiction_rate = (contradictions / len(retrieved)) if retrieved else 0.0

    return {
        "id": case["id"],
        "query": case["query"],
        "adversarial": case.get("adversarial"),
        "precision_at_k": precision,
        "recall_at_k": recall,
        "top1_correct": bool(top1),
        "top3_correct": bool(top3),
        "irrelevant_rate": irrelevant_rate,
        "wrong_project_rate": wrong_project_rate,
        "stale_superseded_rate": stale_rate,
        "contradiction_rate": contradiction_rate,
        "approx_token_cost": pack.get("approx_token_cost", 0),
        "zero_result_correct": zero_ok,
        "pack_size": pack.get("pack_size", 0),
        "injected_ids": retrieved,
        "project": (pack.get("project") or {}).get("id") if pack.get("project") else None,
        "expect_empty": expect_empty,
    }


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(results) or 1
    zero_cases = [r for r in results if r.get("zero_result_correct") is not None]
    return {
        "n": len(results),
        "precision_at_k": _mean([r["precision_at_k"] for r in results]),
        "recall_at_k": _mean([r["recall_at_k"] for r in results]),
        "top1_correctness": _mean([1.0 if r["top1_correct"] else 0.0 for r in results]),
        "top3_correctness": _mean([1.0 if r["top3_correct"] else 0.0 for r in results]),
        "irrelevant_memory_rate": _mean([r["irrelevant_rate"] for r in results]),
        "wrong_project_rate": _mean([r["wrong_project_rate"] for r in results]),
        "stale_superseded_memory_rate": _mean([r["stale_superseded_rate"] for r in results]),
        "contradiction_rate": _mean([r["contradiction_rate"] for r in results]),
        "approx_token_cost": _mean([r["approx_token_cost"] for r in results]),
        "zero_result_correctness": _mean(
            [1.0 if r["zero_result_correct"] else 0.0 for r in zero_cases]
        )
        if zero_cases
        else None,
        "avg_pack_size": _mean([r["pack_size"] for r in results]),
        # keep older names so existing callers do not break
        "right_rate": _mean([1.0 if r["top1_correct"] or r["top3_correct"] else 0.0 for r in results]),
        "miss_rate": _mean([0.0 if (r["top1_correct"] or r["top3_correct"]) else 1.0 for r in results]),
        "irrelevant_rate": _mean([r["irrelevant_rate"] for r in results]),
        "superseded_used_total": sum(1 for r in results if r["stale_superseded_rate"] > 0),
        "avg_token_cost": _mean([r["approx_token_cost"] for r in results]),
    }


def run_eval(
    *,
    db_path=None,
    out_path: Path | None = None,
    fixture: bool = False,
) -> Dict[str, Any]:
    cases = ensure_eval_set()
    if fixture:
        from tempfile import TemporaryDirectory

        if db_path is None:
            tmp = TemporaryDirectory()
            db_path = str(Path(tmp.name) / "eval_fixture.sqlite")
            seed_eval_fixture(db_path)
            payload = _run_cases(cases, db_path)
            dest = out_path or RESULTS_PATH
            dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp.cleanup()
            return payload
        seed_eval_fixture(db_path)
    return _run_cases(cases, db_path, out_path=out_path)


def _run_cases(cases: List[Dict[str, Any]], db_path, out_path: Path | None = None) -> Dict[str, Any]:
    conn = ensure_schema(connect(db_path) if db_path else None)
    results = []
    for case in cases:
        pack = build_pack(
            case["query"],
            k=int(case.get("k") or 8),
            project_id=case.get("current_project_id") or case.get("project_hint"),
            db_path=db_path,
            log=True,
            min_score=float(case["min_score"]) if case.get("min_score") is not None else DEFAULT_MIN_SCORE,
        )
        results.append(_score_one(case, pack, conn))
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "summary": summarize(results),
        "results": results,
    }
    dest = out_path or RESULTS_PATH
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    conn.close()
    return payload
