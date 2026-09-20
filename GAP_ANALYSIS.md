# Gap Analysis: ARCHITECTURE.md vs Phase 0

Compare north star (`ARCHITECTURE.md`) to Phase 0 as originally built on the box
(verified: `MEMORY_SPEC.md`, `db.py`, `decisions.sqlite`, `backfill_report.json`).

**Phase 0 verified facts (2026-09-19 PT):**
- DB: `decisions.sqlite` — tables `entries`, `entries_vec` (sqlite-vec vec0, 384-d), `ingest_log`, `processed_calls`
- Categories: `decisions`, `open_items`, `preferences_and_rules`, `system_facts`, `lessons_learned`
- Status: `active` / `superseded` / `reversed` (24 active, 7 reversed at last backfill report; live DB may differ slightly)
- Embeddings: local fastembed `BAAI/bge-small-en-v1.5` 384-d
- Hook: `run_call_hook.sh` + weekday routine marker; durable-summary only (no turn mining)
- **Missing:** conversations/messages archive, projects, entities, memory_links, FTS, hybrid rerank, memory packs as first-class, importance/confidence fields

| Architecture requirement | Phase 0 status | Phase 1 need |
| --- | --- | --- |
| BOUNDARY: local-only under `./` | Met (spec + practice) | Keep; document in new schema README |
| CHAT HISTORY ≠ MEMORY | Partial: no raw archive; only structured `entries` | Layer A `conversations`/`messages` + Layer B `memories` |
| Layer A raw archive | **Absent** | Create immutable conversations + messages; import voice-call JSON |
| Layer B structured memory | Partial: `entries` ≈ memories without full field set | `memories` with canonical_text, importance, confidence, provenance, supersedes |
| Memory types enum (expanded) | 5 categories only | Map + extend: facts, preferences, goals, project_state, decisions, constraints, tasks, open_questions, corrections, outcomes/lessons, strategies |
| `projects` first-class | **Absent** | `projects` table + compact evolving state fields |
| `entities` | **Absent** | Table ready; light extract/tag in Phase 1 |
| `memory_links` | Only `supersedes_id` on entries | Schema stub for links; **no graph engine** |
| `summaries` hierarchical | **Absent** | Defer full hierarchy; optional project `current_summary` field in Phase 1 |
| `artifacts` | Paths only in `source` text | `artifacts` table for durable markdown paths |
| `retrieval_log` | **Absent** | Log every pack/query |
| Key fields: importance, confidence | **Absent** | First-class columns; user statements high confidence; agent-inferred lower + labeled |
| Provenance to conversation_id + message_id(s) | Partial: source path / call id | Exact conversation_id + message_id list on every memory |
| Hybrid retrieval (vector + FTS + filters + weights) | Vector only (+ reversed heuristic) | FTS5 + vector + project/type/importance/confidence/recency/supersession |
| Memory packs (3–8) | `query(k=8)` flat list | `memory_pack` builder with compact project state |
| Query expansion | **Absent** | Light expansion before hybrid search |
| CREATE/MERGE/UPDATE/SUPERSEDE/IGNORE | Upsert + supersede/reverse helpers | `consolidate` before insert |
| Corrections override | `reversed` + why_failed | Supersede old; retrieval returns latest active |
| Extraction after meaningful turns | Hook on durable summary / backfill md | Rules-based extract from Layer A + md; no speculation as user fact |
| Dedup / consolidation | Manual / weak | Search-before-write consolidate module |
| Embeddings regeneratable / pluggable | Hard-wired fastembed in embed.py | Embedder interface; rebuild-embeddings CLI |
| Eval harness | sample_queries.txt only | Eval set + metrics JSON |
| Import Phase 0 entries | N/A | Migrate active (and reversed as labeled) into `agent_memory.sqlite` |
| Domayn / product integration | Correctly out of scope | Keep out of scope |

## 5-bullet gap summary

1. **No Layer A** — Phase 0 never stores immutable conversations/messages; provenance cannot point at message ids.
2. **No hybrid pack** — retrieval is vector-only over `entries`; no FTS, importance/confidence, projects, or retrieval_log.
3. **Narrow type model** — five categories vs architecture’s richer typed memories (corrections, strategies, tasks, etc.).
4. **No consolidate gate** — risk of near-duplicates without CREATE/MERGE/UPDATE/SUPERSEDE/IGNORE.
5. **No eval** — cannot score right/miss/irrelevant/superseded-used/token cost.

## Recommended next action

**Start Phase 1** (approved): new `agent_memory.sqlite` + `phase1/` package; leave `decisions.sqlite` + bak intact.
