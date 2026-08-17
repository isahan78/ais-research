#!/usr/bin/env bash
# Gate 1 in one command: 4 stages on 20 MATH-500 problems at k=50%.
#
# Stages run as SEPARATE processes on purpose: vLLM (stage 1) must tear down
# completely before HF (stage 3) loads — 2 x 15.26 GiB does not fit in 24 GiB.
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$EXPERIMENT_DIR")"
cd "$PROJECT_ROOT"

PY="${PYTHON:-python}"

run_stage() {
  local name="$1" module="$2"
  echo "=== stage: $name ==="
  local start=$SECONDS
  "$PY" -m "$module"
  echo "=== $name done in $((SECONDS - start))s ==="
  echo
}

total_start=$SECONDS

run_stage "1/4 generate_traces (vLLM)"      experiment.generate_traces
run_stage "2/4 truncate (k=50%)"            experiment.truncate
run_stage "3/4 harvest_activations (HF)"    experiment.harvest_activations
run_stage "4/4 train_probe"                 experiment.train_probe

echo "=== smoke test complete in $((SECONDS - total_start))s total ==="
echo "Final numbers (also in experiment/outputs/results.json):"
"$PY" - <<'EOF'
import json
from experiment.config import CONFIG
r = json.load(open(CONFIG.results_path))
for layer, d in r["per_layer"].items():
    print(f"  {layer}: AUC={d['auc']} CI95={d['auc_ci95']} "
          f"floor_mean={d['floor_mean']} floor_p95={d['floor_p95']} "
          f"frac_floor_seeds_beaten={d['floor_frac_below_auc']} "
          f"beats_floor_p95={d['beats_floor_p95']}")
EOF
