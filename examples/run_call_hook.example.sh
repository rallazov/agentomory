#!/usr/bin/env bash
# Example post-call hook. Configure AGENTOMORY_CALLS_DIR and run from the repo root.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate
python -m phase1.cli archive-import --dir "${AGENTOMORY_CALLS_DIR:-$HOME/.agentomory/voice-calls}"
python -m phase1.cli extract
