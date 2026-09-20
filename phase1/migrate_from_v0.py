"""Import Phase 0 decisions.sqlite entries → typed memories with provenance."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .consolidate import apply_memory
from .schema import DB_PATH, connect, ensure_schema

V0_PATH = Path(__file__).resolve().parent.parent / "decisions.sqlite"

CATEGORY_MAP = {
    "decisions": "decision",
    "open_items": "task",
    "preferences_and_rules": "preference",
    "system_facts": "fact",
    "lessons_learned": "lesson",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def migrate(*, v0_path: Path | None = None, db_path=None) -> Dict[str, Any]:
    src = Path(v0_path) if v0_path else V0_PATH
    if not src.exists():
        return {"error": f"missing {src}"}

    c0 = ensure_schema(connect(db_path) if db_path else None)
    c0.close()
    v0 = sqlite3.connect(str(src))
    v0.row_factory = sqlite3.Row
    rows = v0.execute("SELECT * FROM entries").fetchall()
    stats = {"CREATE": 0, "MERGE": 0, "UPDATE": 0, "SUPERSEDE": 0, "IGNORE": 0, "reversed_imported": 0}
    details: List[Dict[str, Any]] = []

    for r in rows:
        mtype = CATEGORY_MAP.get(r["category"], "fact")
        status = r["status"]
        origin = "phase0_import"
        conf = 0.85 if status == "active" else 0.7
        # provenance: call id or source path — no message ids in v0
        prov_note = f"phase0 entry {r['id']} category={r['category']}"
        msg_ids = json.dumps([])
        conv_id = r["ingested_from_call_id"]
        art = None
        source = r["source"] or ""
        if source.startswith("/") and source.endswith(".md"):
            # register as artifact path note only
            art = None
            prov_note += f" source={source}"

        cand = {
            "id": f"p0-{r['id']}",
            "memory_type": mtype if status != "reversed" else "lesson",
            "title": r["title"],
            "canonical_text": r["body"],
            "importance": 0.8 if r["category"] in ("preferences_and_rules", "decisions") else 0.65,
            "confidence": conf,
            "origin": origin,
            "status": "active" if status == "active" else ("reversed" if status == "reversed" else "superseded"),
            "supersedes_id": f"p0-{r['supersedes_id']}" if r["supersedes_id"] else None,
            "why_failed": r["why_failed"],
            "provenance_conversation_id": conv_id,
            "provenance_message_ids": msg_ids,
            "provenance_artifact_id": art,
            "provenance_note": prov_note,
            "source_kind": r["source_kind"],
            "phase0_entry_id": r["id"],
        }
        if status == "reversed":
            # insert directly as reversed without consolidate active path
            from .schema import connect as c2
            conn = ensure_schema(c2(db_path) if db_path else None)
            now = _now()
            try:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO memories (
                      id, memory_type, status, title, canonical_text, importance, confidence,
                      origin, project_id, supersedes_id, why_failed,
                      provenance_conversation_id, provenance_message_ids, provenance_artifact_id,
                      provenance_note, source_kind, phase0_entry_id, created_at, updated_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        cand["id"], cand["memory_type"], "reversed", cand["title"],
                        cand["canonical_text"], cand["importance"], cand["confidence"],
                        origin, None, cand["supersedes_id"], cand["why_failed"] or "phase0 reversed",
                        conv_id, msg_ids, None, prov_note, r["source_kind"], r["id"], now, now,
                    ),
                )
                from .provenance import add_sources_from_candidate

                add_sources_from_candidate(conn, cand["id"], cand, role="original")
                conn.commit()
                conn.close()
                stats["reversed_imported"] += 1
                details.append({"id": cand["id"], "action": "reversed_import"})
            except Exception as e:
                try:
                    conn.close()
                except Exception:
                    pass
                details.append({"id": cand["id"], "error": str(e)})
            continue

        # active / superseded via consolidate for actives; superseded as status
        if status == "superseded":
            from .schema import connect as c2
            conn = ensure_schema(c2(db_path) if db_path else None)
            now = _now()
            conn.execute(
                """
                INSERT OR REPLACE INTO memories (
                  id, memory_type, status, title, canonical_text, importance, confidence,
                  origin, project_id, supersedes_id, why_failed,
                  provenance_conversation_id, provenance_message_ids, provenance_artifact_id,
                  provenance_note, source_kind, phase0_entry_id, created_at, updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    cand["id"], cand["memory_type"], "superseded", cand["title"],
                    cand["canonical_text"], cand["importance"], cand["confidence"],
                    origin, None, cand["supersedes_id"], None,
                    conv_id, msg_ids, None, prov_note, r["source_kind"], r["id"], now, now,
                ),
            )
            from .provenance import add_sources_from_candidate

            add_sources_from_candidate(conn, cand["id"], cand, role="original")
            conn.commit()
            conn.close()
            stats["SUPERSEDE"] += 1
            continue

        out = apply_memory(cand, db_path=db_path)
        stats[out.get("action", "CREATE")] = stats.get(out.get("action", "CREATE"), 0) + 1
        details.append(out)

    v0.close()
    return {"stats": stats, "count_entries": len(rows), "db": str(db_path or DB_PATH)}
