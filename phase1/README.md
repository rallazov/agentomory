# Phase 1 — Agentomory package

Local SQLite memory: Layer A (archive) + Layer B (typed memories) + hybrid packs.

## Setup

```bash
cd /path/to/agentomory
source .venv/bin/activate
python -m phase1.cli ensure-schema
# configure AGENTOMORY_CALLS_DIR / AGENTOMORY_DURABLE_MD, then:
python -m phase1.cli archive-import
python -m phase1.cli migrate-v0   # only if you have a local decisions.sqlite
python -m phase1.cli seed-projects
python -m phase1.cli extract
python -m phase1.cli eval
# or: python -m phase1.cli bootstrap
```

## Query → memory pack

```bash
python -m phase1.cli pack "your question here"
python -m phase1.cli pack "calendar sprint" -k 6
python -m phase1.cli query "local call memory not product RAG"
```

## Rebuild embeddings

```bash
python -m phase1.cli rebuild-embeddings
```

## Modules

`schema`, `embed` (pluggable), `archive`, `extract`, `consolidate`, `retrieve` (hybrid), `memory_pack`, `retrieval_log`, `migrate_from_v0`, `seed_projects`, `eval_harness`, `cli`

## Non-goals

No Neo4j / IntentGraph / Domayn product integration / learned reranker / graph engine (`memory_links` schema only).
