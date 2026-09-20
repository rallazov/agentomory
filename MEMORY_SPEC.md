> **North star:** See [`ARCHITECTURE.md`](./ARCHITECTURE.md).  
> **Phase 0 status:** This file describes the **Phase 0 subset** (`decisions.sqlite`, five categories, vector-only retrieve). Phase 1 lives in `agent_memory.sqlite` + `phase1/` — see `PHASE1_PLAN.md` and `GAP_ANALYSIS.md`.

# Decision Memory Spec

Durable decision memory for agent CEO on the Grok Bot box.
Store: `decisions.sqlite` (Phase 0 legacy; local only, not shipped)
Embeddings: local `BAAI/bge-small-en-v1.5` via fastembed, 384 dims.
Vector index: sqlite-vec 0.1.9 (`vec0`).
Venv: project `.venv` (create locally)

This DB is **local call/decision memory on this computer**, not a product RAG feature.

---

## Categories (exactly one per entry)

| Category | Use for |
| --- | --- |
| `decisions` | Locked product/engineering/workflow choices that should guide future work |
| `open_items` | Agreed unresolved work that still needs a human or agent action |
| `preferences_and_rules` | Standing rules, gates, non-negotiables, ownership norms |
| `system_facts` | Stable facts about the stack, paths, roster, repo, templates |
| `lessons_learned` | What failed or hurt, with enough context to avoid repeating it |

## Status

| Status | Meaning |
| --- | --- |
| `active` | Current truth; default for retrieval |
| `superseded` | Replaced by a newer entry (`supersedes_id` on the newer entry points here) |
| `reversed` | Was wrong or harmful; **keep content** plus required `why_failed` note |

Never silently delete. Soft-retire via `superseded` or `reversed` only.

---

## Ingest test

Save **only** what would change a future decision.

**Include:** standing rules, locked scope, ownership, durable open blockers, system facts agents keep re-asking.

**Exclude:** small talk, one-off troubleshooting, transient noise, raw transcripts, speculative roadmap not written down, name brainstorming drafts.

Before write, ask: *Would retrieving this in six months help or hurt?*

- Help → save as `active` (correct category).
- Hurt / wrong → mark `reversed` with `why_failed`, or do not save.
- Uncertain / ephemeral → do not save.

Prefer short durable statements over long dumps.

---

## Post-conversation hook

There is **no native voice-call-ended event** in Grok Bot routines.

After each call ends, CEO (or a scheduled routine) must run:

```bash
`examples/run_call_hook.example.sh` (configure locally)
```

That script activates the venv and runs `python cli.py process-calls`, which:

1. Scans voice-call JSON under the CEO agent `voice-calls/` directories.
2. Skips `call_id`s already in `processed_calls`.
3. For completed/ended calls, ingests only an explicit `durable_summary`/`summary` field that passes the ingest test (never mines turns/transcripts).
4. Embeds and upserts into `decisions.sqlite`.
5. Logs skips when a call has no durable content.

See `HOOK.md` and `HOOK_WIRED.json`.

---

## Backfill rules

- Recoverable sources only (markdown decision notes, sprint docs, CEO call summary, story template, etc.).
- Extract only what passes the ingest test.
- No raw transcripts.
- Re-extract prior knowledge under the new categories rather than discarding it.
- Do not invent product roadmap items absent from the files.

---

## Retrieval

`retrieve.query(text, k=8)`:

1. Embed the query.
2. Vector-search all entries; join metadata.
3. Return top relevant **`active`** hits (up to `k`).
4. Also return **`reversed`** neighbors when the same query ranks them as related (distance within a relatedness band or among top reversed neighbors), flagged so the caller sees past failures on that topic.
5. `superseded` rows are not returned unless explicitly requested by a future API; use the active replacement instead.

---

## Schema (`decisions.sqlite`)

### Table `entries`

| Column | Type | Notes |
| --- | --- | --- |
| `id` | TEXT PRIMARY KEY | Stable slug |
| `category` | TEXT NOT NULL | CHECK IN five categories above |
| `status` | TEXT NOT NULL | CHECK IN (`active`, `superseded`, `reversed`) |
| `title` | TEXT NOT NULL | Short label |
| `body` | TEXT NOT NULL | Durable statement |
| `why_failed` | TEXT NULL | Required when `status='reversed'` |
| `supersedes_id` | TEXT NULL | FK-ish to older `entries.id` |
| `source` | TEXT NOT NULL | File path or call id |
| `source_kind` | TEXT NOT NULL | e.g. `ceo_call_summary`, `sprint_doc`, `voice_call`, `durable_md` |
| `created_at` | TEXT NOT NULL | ISO-8601 |
| `updated_at` | TEXT NOT NULL | ISO-8601 |
| `ingested_from_call_id` | TEXT NULL | Optional call id |

CHECK: if `status='reversed'` then `why_failed` IS NOT NULL.

### Virtual table `entries_vec` (sqlite-vec vec0)

```sql
CREATE VIRTUAL TABLE entries_vec USING vec0(
  entry_id TEXT PRIMARY KEY,
  embedding float[384]
);
```

`entry_id` matches `entries.id`. Embeddings are 384-dim float32. Inserts/updates go through `ingest.py` / `embed.py`.

### Table `ingest_log`

| Column | Type |
| --- | --- |
| `id` | INTEGER PRIMARY KEY AUTOINCREMENT |
| `source_id` | TEXT NOT NULL |
| `source_path` | TEXT |
| `processed_at` | TEXT NOT NULL |
| `entries_written` | INTEGER NOT NULL DEFAULT 0 |
| `notes` | TEXT |

### Table `processed_calls`

| Column | Type |
| --- | --- |
| `call_id` | TEXT PRIMARY KEY |
| `path` | TEXT NOT NULL |
| `processed_at` | TEXT NOT NULL |
| `summary_hash` | TEXT |
| `entries_written` | INTEGER NOT NULL DEFAULT 0 |
| `notes` | TEXT |

---

## Modules

| Module | Role |
| --- | --- |
| `db.py` | Connect, load sqlite_vec, ensure schema |
| `embed.py` | BAAI/bge-small-en-v1.5 via fastembed |
| `retrieve.py` | `query(text, k=8)` active + related reversed |
| `ingest.py` | Upsert, status transitions, never silent delete |
| `backfill.py` | Extract from durable markdown sources |
| `hook_after_call.py` | Process finished voice call JSON |
| `cli.py` | `ensure-schema`, `backfill`, `query`, `process-calls` |

## CLI

```bash
cd /path/to/agentomory
source .venv/bin/activate
python cli.py ensure-schema
python cli.py backfill
python cli.py query "Cloud Agent launch"
python cli.py process-calls
```
