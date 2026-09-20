# Persistent Local Agent Memory System

## BOUNDARY

This system is **local-first**: durable memory stays on the machine that runs the agent (default DB: `agent_memory.sqlite` next to the package / configurable).

It must **NOT** be shipped into Domayn / `application-tracker` or Dirijor Core. Agentomory is a separate reusable memory library — not product tenant RAG.

Local agent memory ≠ product tenant RAG. Keep that fence.

---

## Objective and core principle

**Objective:** Give the CEO (and later other box agents) a durable, queryable memory of decisions, rules, facts, lessons, projects, and corrections — so future turns retrieve *what still matters*, not a dump of chat.

**Core principle: CHAT HISTORY ≠ MEMORY**

| Chat history | Memory |
| --- | --- |
| Raw turns, noise, retries, small talk | Canonical statements that should change future behavior |
| Ephemeral troubleshooting | Standing rules, locked decisions, open blockers, lessons |
| Full transcripts as default context | Small **memory packs** (few high-signal items) |
| Model/session dependent | Local-first, model-independent store |

Raw conversation may be **archived** (Layer A). Only extracted, typed, scored items become **memory** (Layer B). Vectors and link graphs are **retrieval aids**, not the source of truth.

---

## Three layers

### Layer A — Raw archive

Immutable-enough record of conversations and messages (and pointers to durable markdown / call JSON).

- Tables: `conversations`, `messages` (and optionally pointers in `artifacts`).
- Purpose: provenance, re-extraction, audit. **Not** default prompt fuel.

### Layer B — Structured memory

Canonical, typed memories with status, importance, confidence, provenance, and supersession.

- Tables: `memories`, `projects`, `entities`, `summaries`, `artifacts`.
- Purpose: what retrieval should prefer for CEO turns.

### Retrieval / graph (not a separate “truth” layer)

- **Vectors:** discovery / similarity search over memory text (and optionally summaries).
- **Relationships:** `memory_links` (and entity links) for graph-ish navigation and rerank signals.
- Rule: **vector is discovery, not logic.** Scoring and business rules decide what enters a memory pack.

---

## Memory types (enum)

Typed memories (Layer B). Phase 0 categories map into this enum; architecture adds types needed for corrections and trajectories.

| Type | Role |
| --- | --- |
| `decision` | Locked product / engineering / workflow choice |
| `open_item` | Agreed unresolved work needing human or agent action |
| `preference_or_rule` | Standing rules, gates, non-negotiables, ownership norms |
| `system_fact` | Stable facts (paths, roster, repo, templates, stack) |
| `lesson_learned` | What failed or hurt; enough context to avoid repeat |
| `correction` | Explicit override of a prior memory (“that was wrong; do X”) |
| `outcome` | Result of a strategy or decision (trajectory endpoint) |
| `strategy` | Approach chosen for a project or class of problems |

Summaries, projects, entities, and artifacts are **first-class tables**, not only memory-type tags.

---

## Suggested SQLite tables

| Table | Layer | Purpose |
| --- | --- | --- |
| `conversations` | A | Thread / call / chat session metadata |
| `messages` | A | Ordered turns (role, text, timestamps, source ids) |
| `memories` | B | Canonical typed memories (see fields below) |
| `projects` | B | First-class projects / sprints / initiatives |
| `entities` | B | People, systems, repos, PRs, rooms, etc. |
| `memory_links` | graph | Typed edges between memories (and optionally entities) |
| `summaries` | B | Hierarchical rollups (thread → project → longer horizon) |
| `artifacts` | A/B | Paths to durable markdown, templates, reports |
| `retrieval_log` | ops | What was retrieved for which query (debug / tuning) |

Plus embedding virtual table(s) for memories (and optionally summaries), e.g. sqlite-vec `vec0`, consistent with Phase 0’s local embedding stack.

Operational tables from Phase 0 (`ingest_log`, `processed_calls`) remain useful for hooks and may live alongside or migrate into the new DB.

---

## Key memory fields

| Field | Intent |
| --- | --- |
| `canonical_text` | Short durable statement (prefer over long dumps) |
| `importance` | First-class score for pack selection / ranking |
| `confidence` | How sure we are this is still true |
| `status` | Lifecycle: e.g. `active` / `superseded` / `reversed` (align with Phase 0; never silent delete) |
| `provenance` | Source conversation/message/artifact/call id + kind |
| `supersedes` | Pointer to older memory this replaces |

Additional practical columns (implementation): `id`, `type`, `title`, `project_id`, `created_at`, `updated_at`, `why_failed` when reversed, embedding row in vec table.

---

## Projects as first-class

Projects are not tags bolted onto free text.

- `projects` table holds name, status, goal window, owner hints, links to summaries.
- Memories and summaries can reference `project_id`.
- Project-level summary and trajectories (outcome / strategy / correction) hang off the project.

Examples on this box (names only, from existing durable notes — not a product roadmap claim): calendar sprint work, Cloud Agent workflow rules, team ownership.

---

## `memory_links` relationships

Typed edges. Minimum set for the architecture:

| Relationship | Meaning |
| --- | --- |
| `supersedes` | Newer memory replaces older (also mirrored by `supersedes` field) |
| `corrects` | Correction memory overrides a prior claim |
| `contradicts` | Soft conflict; retrieval should surface both with care |
| `supports` | Evidence or rationale for another memory |
| `related_to` | Topical association without strong semantics |
| `part_of` | Memory belongs to a project or parent memory |
| `extracted_from` | Link back to Layer A message / artifact |
| `about_entity` | Memory concerns an `entities` row |
| `led_to` | Strategy / decision → outcome (trajectory) |
| `caused_by` | Outcome / lesson ← prior decision or event |

---

## Hybrid retrieval + configurable scoring

**Pipeline (conceptual):**

1. **Query expansion** — light rewrite / alias terms (e.g. story-bound launch ↔ Cloud Agent rule).
2. **Candidate discovery** — vector search (and later FTS) over `canonical_text` / titles / summaries.
3. **Hybrid merge** — combine dense + lexical candidates; dedupe by memory id.
4. **Configurable scoring** — weights for similarity, importance, confidence, recency, project match, status (prefer `active`; attach related `reversed` / corrections), link boosts.
5. **Memory pack** — emit a **small** pack for the turn (**about 3–8 memories**, not a chapter).

**Vector is discovery, not logic.** Embeddings find neighbors; rules and scores decide the pack.

---

## Query expansion and memory packs

- Expand queries with synonyms / known aliases from system facts and project names.
- Pack size: **3–8** high-signal memories by default (configurable `k`).
- Prefer: active rules + matching decisions + related reversed/corrections when the topic touches a past failure.
- Omit: superseded duplicates (follow `supersedes` to the live row), low-confidence noise, raw Layer A turns.

---

## Extraction after meaningful turns

- Run extraction **after meaningful turns** (or after a call / durable markdown update) — not on every keystroke.
- **Candidate rules** (align with Phase 0 ingest test):
  - **Include:** standing rules, locked scope, ownership, durable open blockers, system facts agents re-ask, explicit corrections, lessons with failure context.
  - **Exclude:** small talk, one-off troubleshooting, transient noise, raw transcripts as memories, speculative roadmap not written down, name brainstorm drafts.
  - Gate: *Would retrieving this in six months help or hurt?* Help → candidate; hurt → reverse or skip; uncertain → skip.
- Prefer short `canonical_text` over long dumps.
- Voice-call JSON: archive in Layer A; promote to Layer B only via durable summary fields or curated markdown — not silent mining of all turns (Phase 0 hook already avoids turn mining).

---

## Dedup / consolidation and corrections

- Near-duplicate candidates → consolidate or link (`related_to` / merge into one canonical row).
- Updates that replace truth → new row with `supersedes` + old `status=superseded`.
- Wrong / harmful past memory → `status=reversed` (keep content) + `why_failed`, or a `correction` type that `corrects` the old row.
- **Corrections override:** when a correction exists for a topic, pack builder must prefer the correction / active replacement over the reversed or superseded text.

---

## Summaries and trajectories

- **Hierarchical summaries:** message/thread → conversation → project → longer horizon.
- **Project summary:** living rollup of a project’s active decisions, open items, and recent outcomes.
- **Trajectories:** chains of `strategy` → `outcome` and `correction` links (`led_to` / `corrects`) so retrieval can answer “what did we try and what happened?”

Stored in `summaries` and/or typed memories; not a substitute for Layer A.

---

## Privacy and model independence

- **Local-first:** all durable memory on the user's machine; no sample personal DB is shipped.
- **Model independence:** schema and files outlive any one model or session; embeddings are swappable (Phase 0 uses local fastembed `BAAI/bge-small-en-v1.5`, 384-d).
- No shipping of this DB into product repos or cloud tenant stores by default (see BOUNDARY).

---

## Phases 1–4 MVP roadmap

### Phase 1 — Schema, Layer A/B, packs, migrate Phase 0

- New `agent_memory.sqlite` (keep `decisions.sqlite` as legacy/v0; leave bak intact).
- Create Phase 1 tables: `conversations`, `messages`, `memories` (+ vec), `projects`, `artifacts`, `retrieval_log` (and minimal `ingest_log` / `processed_calls` continuity).
- Import existing Phase 0 `entries` as typed `memories`.
- Memory pack builder for CEO turns; hybrid retrieval foundation (vector + status/importance filters).
- Success check: pack for “Cloud Agent launch rule” and “calendar sprint”.
- See `PHASE1_PLAN.md`.

### Phase 2 — Extraction quality

- Extraction after meaningful turns; candidate rules enforced in code.
- Dedup / consolidation; corrections override wiring.
- Stronger provenance from Layer A → Layer B.

### Phase 3 — Summaries and trajectories

- Hierarchical `summaries`; project summary.
- Outcome / strategy / correction trajectories via types + `memory_links`.

### Phase 4 — Graph and scoring depth

- First-class `entities` + rich `memory_links` usage in retrieval.
- FTS + hybrid rerank; configurable scoring weights; query expansion polish.
- Still local SQLite unless Ramin explicitly asks otherwise. **No** Neo4j / IntentGraph / Domayn product integration unless separately approved.

---

## Guiding rule

**Save only what should change a future decision. Retrieve a small pack of current truth (plus related failures/corrections), never the whole chat.**

Chat is archive. Memory is commitment.

---

## Success metrics

| Metric | Target |
| --- | --- |
| Pack usefulness | CEO turn gets 3–8 on-topic active memories without manual paste |
| Correction fidelity | Reversed / corrected items surface when the topic matches; wrong “active” claims do not |
| Provenance | Every memory traces to call id, markdown path, or message id |
| Boundary | Zero accidental commits of this system into `application-tracker` |
| Latency / locality | Retrieval stays on-box; no cloud memory dependency for CEO ops |
| Six-month test | Entries that fail “help in six months” are skipped or reversed |

---

## Document control

| Item | Value |
| --- | --- |
| Location | `ARCHITECTURE.md` (this repo) |
| Related | `MEMORY_SPEC.md` (Phase 0 subset), `GAP_ANALYSIS.md`, `PHASE1_PLAN.md` |
| Phase 0 store | `decisions.sqlite` (legacy; not shipped — migrate locally if you have one) |
| Captured | 2026-09-19 PT — structured north star for local agent memory on this box |

---

## Related direction: AIR / personal agent OS (Sep 19 2026)

Relevant to this architecture; **not Phase 1 scope**.

Goal: remove lossy ambiguity between human thought and model execution without “tricking” internal model vectors.

Layers we control:
1. External embeddings (SQLite / vectors) — controllable
2. Semantic IR (AIR) — controllable
3. Internal foundation-model activations — do **not** target

Direction:
- Natural language → Intent compiler → AIR → memory/tools → high-tier model → outcome extraction → memory
- Stable semantic addresses (`@decision.*`, `@project.*`) after fuzzy discovery
- Memory as tools (`memory.search/get/...`) so models like Astra reason hard on small packs
- Task router: local/small vs mid vs max-reasoning by difficulty (cheaper + smarter effective behavior)
- Models are replaceable; this OS owns memory, projects, strategies, outcomes

Phase gate: only after Phase 1 retrieve→pack loop is measured and reliable.
