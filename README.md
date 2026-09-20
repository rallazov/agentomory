# Agentomory

**Agent + memory.** Local, model-independent agent memory: infinite-ish durable store, finite context at query time.

Agentomory keeps standing decisions, preferences, project state, lessons, and corrections in a local SQLite database, then returns small **memory packs** for the next turn — not a dump of chat history.

> **CHAT HISTORY ≠ MEMORY.** Raw turns may be archived (Layer A). Only extracted, typed, scored items become memory (Layer B). Vectors help retrieval; SQLite text is the source of truth.

## What this is / is not

| Is | Is not |
| --- | --- |
| Reusable local memory for Cursor, Codex, Grok Bot (and similar) | **Not Domayn** / not `application-tracker` product RAG |
| Optional tool that Dirijor (or any orchestrator) can call later | **Not Dirijor Core** |
| SQLite + hybrid retrieval + small memory packs | Not Neo4j / IntentGraph / cloud memory SaaS |
| Data stays on your machine | No sample personal DB is shipped |

## Install

```bash
git clone https://github.com/rallazov/agentomory.git
cd agentomory
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# or: pip install -e .
```

Dependencies: `fastembed`, `sqlite-vec`, `numpy`.

## Quick start

```bash
# Create local schema (writes agent_memory.sqlite in the repo root by default)
python -m phase1.cli ensure-schema

# Build a memory pack for a query (after you have memories ingested)
python -m phase1.cli pack "Cloud Agent launch rule"
python -m phase1.cli pack "calendar sprint" -k 6

# Eval harness (needs local data; ships queries in phase1/eval_set.json)
python -m phase1.cli eval
```

Point the DB elsewhere with `--db /path/to/agent_memory.sqlite`.

### Ingest (local paths only)

```bash
# Voice-call JSON archive → Layer A
export AGENTOMORY_CALLS_DIR="$HOME/.agentomory/voice-calls"
python -m phase1.cli archive-import --dir "$AGENTOMORY_CALLS_DIR"

# Durable markdown → Layer B extraction
export AGENTOMORY_DURABLE_MD="$HOME/notes/decisions.md:$HOME/notes/rules.md"
# or: mkdir -p ~/.agentomory/durable && cp your.md ~/.agentomory/durable/
python -m phase1.cli extract

# Optional: migrate a legacy Phase 0 decisions.sqlite if you have one locally
# (place it next to the package as decisions.sqlite, then:)
python -m phase1.cli migrate-v0

python -m phase1.cli seed-projects   # optional project rows
python -m phase1.cli rebuild-embeddings
```

Full bootstrap (when local sources are configured):

```bash
python -m phase1.cli bootstrap
```

## Architecture (short)

See [ARCHITECTURE.md](ARCHITECTURE.md), [MEMORY_SPEC.md](MEMORY_SPEC.md), [PHASE1_PLAN.md](PHASE1_PLAN.md), [GAP_ANALYSIS.md](GAP_ANALYSIS.md).

- **Layer A** — immutable-ish conversations / messages (provenance).
- **Layer B** — typed memories (fact, preference, decision, …) with status, importance, confidence.
- **Retrieval** — hybrid (vector + lexical + recency/importance); output is a **memory pack**, not full history.
- **Phase 0** — earlier `decisions.sqlite` prototype. Not shipped here; `migrate-v0` can import a local copy. Phase 1 (`phase1/`) is the product.

## MCP / tools (planned)

Agentomory is meant to be callable from:

- **Cursor** / **Codex** (MCP or CLI wrappers)
- **Grok Bot** (local box tools)
- Optionally **Dirijor** as an external memory tool — without merging into Dirijor Core or Domayn

Packaging MCP servers and tool schemas is planned; this repo ships the Phase 1 Python library + CLI first.

## Privacy

- Runtime DBs (`*.sqlite`), venv, and personal dumps are **gitignored** and **not** in this repo.
- Configure `AGENTOMORY_CALLS_DIR` / `AGENTOMORY_DURABLE_MD` on your machine.
- Do not commit `eval_results.json` if it contains personal hit IDs or titles.

## License

MIT — Copyright (c) 2026 Ramin Allazov
