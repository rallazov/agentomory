# Phase 1 Plan (this box only)

## DB path decision

**Create `agent_memory.sqlite`** (local) for the Phase 1 schema.

**Keep** a local `decisions.sqlite` as legacy Phase 0 / v0 (read-only for migrate) if you have one.

**Leave** any local `decisions.sqlite.bak-*` backups intact.

**Justify:** Safer than in-place alter of vec0 tables; Phase 0 hook can keep working until cutover; migrate imports `entries` → typed `memories` with provenance `source_kind=phase0_entries`.

## Phase 1 tables only

- `conversations`, `messages` (Layer A; messages immutable)
- `memories` + `memories_fts` (FTS5) + `memories_vec` (sqlite-vec)
- `projects` (compact state fields)
- `entities` (light)
- `memory_links` (**schema only**, no graph engine)
- `artifacts`
- `retrieval_log`
- `ingest_log` / `processed_sources` (ops)

Defer: Neo4j, IntentGraph, hierarchical summary engine, learned reranker, Domayn integration.

## Modules (`phase1/`)

| Module | Role |
| --- | --- |
| `schema.py` | ensure schema |
| `embed.py` | pluggable embedder (fastembed default) |
| `archive.py` | Layer A import (voice-call JSON → conversations/messages) |
| `extract.py` | rules-based candidates from messages + durable md |
| `consolidate.py` | search → CREATE/MERGE/UPDATE/SUPERSEDE/IGNORE |
| `retrieve.py` | hybrid vector + FTS + filters + weights |
| `memory_pack.py` | small pack builder |
| `retrieval_log.py` | log every lookup |
| `migrate_from_v0.py` | import decisions.sqlite entries |
| `eval_harness.py` | eval questions + metrics |
| `cli.py` | archive-import, extract, consolidate, query/pack, eval, rebuild-embeddings |
| `seed_projects.py` | projects from durable markdown only |

## Layer A + Layer B path

1. **Voice-call JSON** → Layer A: one conversation per `callId`; each turn → immutable message (`speaker` user|assistant, text, atMs). Never update message text.
2. **Durable markdown** (`ceo-call-summary.md`, sprint/decision notes that pass ingest test) → artifacts + extract → Layer B memories with high confidence when text is Ramin/CEO-locked written record; label `source_role`.
3. **Phase 0 entries** → memories with provenance to artifact/call where known; type mapped from category.
4. Extract from Layer A **by rules only** (user turns preferred for high confidence; assistant-inferred lower confidence + `origin=agent_inferred`).

## Memory pack (CEO turns)

Hybrid retrieve → score → top 3–8 + compact project state slice (purpose, current_goal, current_stage, key_decisions, constraints, open_questions, next_actions, current_summary).

## Non-goals (Phase 1)

- No Neo4j / graph engine / IntentGraph / agent routing
- No Domayn / application-tracker integration / auto-PR docs
- No learned reranker
- No shipping this tree into product repos

## Success check

Can retrieve a small memory pack for:
- `"Cloud Agent launch rule"`
- `"calendar sprint"`

Eval harness writes `phase1/eval_results.json` (local; not committed with personal hits).
