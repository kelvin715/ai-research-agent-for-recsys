#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_PARENT="$(dirname "$PROJECT_ROOT")"
DATA_ROOT="$PROJECT_PARENT/KuaiRand-Pure/data"
STARTER_ROOT="$PROJECT_PARENT/kuairand-starter-kit"
ORCHESTRATOR_PY="$PROJECT_ROOT/.orchestrator-venv/bin/python"
TRACK2_RUN_ID="${TRACK2_RUN_ID:-formal-reproduction-$(date -u +%Y%m%d-%H%M%S)}"
TRACK2_MODEL="gpt-5.4"

if [[ ! -x "$ORCHESTRATOR_PY" ]]; then
  echo "Missing $ORCHESTRATOR_PY. Complete the README setup first." >&2
  exit 2
fi

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "Missing official data directory: $DATA_ROOT" >&2
  exit 2
fi

if [[ ! -d "$STARTER_ROOT" ]]; then
  echo "Missing official starter kit: $STARTER_ROOT" >&2
  exit 2
fi

if [[ ! -d "$PROJECT_ROOT/views/agent" || ! -d "$PROJECT_ROOT/trusted_cache" ]]; then
  echo "Missing generated data views or trusted label cache. Complete the README setup first." >&2
  exit 2
fi

cd "$PROJECT_ROOT"
python3 trusted/manifest.py --verify

echo "Run ID: $TRACK2_RUN_ID"
echo "Live graph (second terminal):"
echo "  python3 scripts/serve_experiment_graph.py --run-dir runs/$TRACK2_RUN_ID --open"

exec "$ORCHESTRATOR_PY" orchestrator/agent.py \
  --run-id "$TRACK2_RUN_ID" \
  --role formal \
  --model "$TRACK2_MODEL" \
  --research-mode live \
  --research-policy prior-first \
  --empirical-prior-snapshot empirical_priors/track2-solutions-valid-only-v1.json
