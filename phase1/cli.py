#!/usr/bin/env python3
"""Phase 1 CLI: archive-import, extract, consolidate, query/pack, eval, rebuild-embeddings."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow `python -m phase1.cli` and `python phase1/cli.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase1.schema import ensure_schema, connect, DB_PATH  # noqa: E402
from phase1.archive import import_all_calls  # noqa: E402
from phase1.extract import extract_from_messages, extract_from_durable_markdown  # noqa: E402
from phase1.migrate_from_v0 import migrate  # noqa: E402
from phase1.seed_projects import seed  # noqa: E402
from phase1.memory_pack import build_pack  # noqa: E402
from phase1.eval_harness import run_eval  # noqa: E402
from phase1.embed import get_embedder  # noqa: E402
from phase1.retrieval_log import attach_retrieval_feedback  # noqa: E402


def cmd_ensure(args):
    ensure_schema(connect(args.db))
    print(json.dumps({"ok": True, "db": str(args.db or DB_PATH)}))


def cmd_archive_import(args):
    directory = Path(args.dir) if getattr(args, "dir", None) else None
    res = import_all_calls(directory, db_path=args.db)
    print(json.dumps(res, indent=2))


def cmd_migrate(args):
    print(json.dumps(migrate(db_path=args.db), indent=2))


def cmd_seed_projects(args):
    print(json.dumps(seed(db_path=args.db), indent=2))


def cmd_extract(args):
    out = {
        "from_messages": extract_from_messages(db_path=args.db, limit=args.limit),
        "from_markdown": extract_from_durable_markdown(db_path=args.db),
    }
    # summarize
    summary = {
        "message_actions": len(out["from_messages"]),
        "markdown_actions": len(out["from_markdown"]),
    }
    print(json.dumps({"summary": summary, "sample": out["from_messages"][:5]}, indent=2))


def cmd_pack(args):
    pack = build_pack(
        args.query,
        k=args.k,
        project_id=args.project,
        current_project_id=getattr(args, "current_project", None),
        db_path=args.db,
        min_score=args.min_score,
        destination=args.destination,
        request_id=args.request_id,
        turn_id=args.turn_id,
        model_used=args.model,
    )
    print(json.dumps(pack, indent=2))


def cmd_query(args):
    cmd_pack(args)


def cmd_eval(args):
    print(json.dumps(run_eval(db_path=args.db, fixture=args.fixture).get("summary"), indent=2))


def cmd_feedback(args):
    n = attach_retrieval_feedback(
        request_id=args.request_id,
        turn_id=args.turn_id,
        log_id=args.log_id,
        model_used=args.model,
        user_accepted=args.accepted,
        user_corrected=args.corrected,
        helpfulness=args.helpfulness,
        task_outcome=args.outcome,
        db_path=args.db,
    )
    print(json.dumps({"updated": n}))


def cmd_rebuild_embeddings(args):
    conn = ensure_schema(connect(args.db))
    rows = conn.execute(
        "SELECT id, canonical_text FROM memories WHERE status='active'"
    ).fetchall()
    emb = get_embedder()
    n = 0
    for r in rows:
        blob = emb.embed_documents([r["canonical_text"]])[0]
        conn.execute("DELETE FROM memories_vec WHERE memory_id=?", (r["id"],))
        conn.execute(
            "INSERT INTO memories_vec (memory_id, embedding) VALUES (?, ?)",
            (r["id"], blob),
        )
        n += 1
    conn.commit()
    print(json.dumps({"rebuilt": n}))


def cmd_bootstrap(args):
    """Full Phase 1 bootstrap: schema → archive → migrate → seed → extract → eval."""
    ensure_schema(connect(args.db))
    archive = import_all_calls(db_path=args.db)
    mig = migrate(db_path=args.db)
    projects = seed(db_path=args.db)
    md = extract_from_durable_markdown(db_path=args.db)
    msgs = extract_from_messages(db_path=args.db, limit=args.limit)
    ev = run_eval(db_path=args.db)
    conn = connect(args.db)
    counts = {
        "conversations": conn.execute("SELECT COUNT(*) c FROM conversations").fetchone()["c"],
        "messages": conn.execute("SELECT COUNT(*) c FROM messages").fetchone()["c"],
        "memories_active": conn.execute(
            "SELECT COUNT(*) c FROM memories WHERE status='active'"
        ).fetchone()["c"],
        "memories_all": conn.execute("SELECT COUNT(*) c FROM memories").fetchone()["c"],
        "projects": conn.execute("SELECT COUNT(*) c FROM projects").fetchone()["c"],
        "artifacts": conn.execute("SELECT COUNT(*) c FROM artifacts").fetchone()["c"],
        "retrieval_log": conn.execute("SELECT COUNT(*) c FROM retrieval_log").fetchone()["c"],
    }
    conn.close()
    print(
        json.dumps(
            {
                "archive_files": len(archive),
                "migrate": mig.get("stats"),
                "projects": projects,
                "extract_md": len(md),
                "extract_msg": len(msgs),
                "counts": counts,
                "eval_summary": ev.get("summary"),
            },
            indent=2,
        )
    )


def main(argv=None):
    p = argparse.ArgumentParser(description="Agentomory Phase 1 — local agent memory CLI")
    p.add_argument("--db", default=None, help="path to agent_memory.sqlite")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ensure-schema").set_defaults(func=cmd_ensure)
    ai = sub.add_parser("archive-import")
    ai.add_argument("--dir", default=None, help="voice-call JSON dir (or set AGENTOMORY_CALLS_DIR)")
    ai.set_defaults(func=cmd_archive_import)
    sub.add_parser("migrate-v0").set_defaults(func=cmd_migrate)
    sub.add_parser("seed-projects").set_defaults(func=cmd_seed_projects)
    e = sub.add_parser("extract")
    e.add_argument("--limit", type=int, default=200)
    e.set_defaults(func=cmd_extract)
    def _add_pack_args(sp):
        sp.add_argument("query")
        sp.add_argument("-k", type=int, default=8, help="useful maximum pack size (may return fewer or zero)")
        sp.add_argument("--project", default=None, help="explicit current project id from the calling agent")
        sp.add_argument("--current-project", default=None, dest="current_project")
        sp.add_argument("--min-score", type=float, default=0.55)
        sp.add_argument("--destination", default="local", choices=("local", "external_model"))
        sp.add_argument("--request-id", default=None)
        sp.add_argument("--turn-id", default=None)
        sp.add_argument("--model", default=None)

    q = sub.add_parser("query")
    _add_pack_args(q)
    q.set_defaults(func=cmd_query)
    pk = sub.add_parser("pack")
    _add_pack_args(pk)
    pk.set_defaults(func=cmd_pack)
    ev = sub.add_parser("eval")
    ev.add_argument("--fixture", action="store_true", help="seed the built-in eval fixture into a temp DB")
    ev.set_defaults(func=cmd_eval)
    fb = sub.add_parser("feedback")
    fb.add_argument("--request-id", default=None)
    fb.add_argument("--turn-id", default=None)
    fb.add_argument("--log-id", type=int, default=None)
    fb.add_argument("--model", default=None)
    fb.add_argument("--accepted", action="store_true")
    fb.add_argument("--corrected", action="store_true")
    fb.add_argument("--helpfulness", default=None, choices=("helpful", "not_helpful", "mixed"))
    fb.add_argument("--outcome", default=None)
    fb.set_defaults(func=cmd_feedback)
    sub.add_parser("rebuild-embeddings").set_defaults(func=cmd_rebuild_embeddings)
    b = sub.add_parser("bootstrap")
    b.add_argument("--limit", type=int, default=300)
    b.set_defaults(func=cmd_bootstrap)

    # consolidate is applied inside extract; expose explicit no-op note
    c = sub.add_parser("consolidate")
    c.add_argument("--note", action="store_true")
    def _cons(args):
        print(json.dumps({"ok": True, "note": "consolidate runs inside extract via apply_memory"}))
    c.set_defaults(func=_cons)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
