#!/usr/bin/env bash
# Gate 1 in one command: pytest gate + 4 stages + crude text floor on
# 20 level-4/5 MATH-500 problems at k=50%.
#
# Stages run as SEPARATE processes on purpose: vLLM (stage 1) must tear down
# completely before HF (stage 3) loads — 2 x 15.26 GiB does not fit in 24 GiB.
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$EXPERIMENT_DIR")"
cd "$PROJECT_ROOT"

PY="${PYTHON:-python}"

# Everything (stdout+stderr) is tee'd to a log that survives pod teardown —
# console scrollback dies with the pod.
mkdir -p "$EXPERIMENT_DIR/outputs"
LOG="$EXPERIMENT_DIR/outputs/smoke_test.log"
exec > >(tee "$LOG") 2>&1
echo "smoke_test.sh started $(date -u +%Y-%m-%dT%H:%M:%SZ); log: $LOG"
echo

run_stage() {
  local name="$1" module="$2"
  echo "=== stage: $name ==="
  local start=$SECONDS
  "$PY" -m "$module"
  echo "=== $name done in $((SECONDS - start))s ==="
  echo
}

total_start=$SECONDS

# Stage 0: the invariant suite MUST pass before any GPU time is spent.
echo "=== stage: 0/5 pytest (invariant suite) ==="
"$PY" -m pytest "$EXPERIMENT_DIR/tests/" -q
echo "=== pytest done in $((SECONDS - total_start))s ==="
echo

run_stage "1/5 generate_traces (vLLM)"      experiment.generate_traces
run_stage "2/5 truncate (k=50%)"            experiment.truncate
run_stage "3/5 harvest_activations (HF)"    experiment.harvest_activations
run_stage "4/5 train_probe"                 experiment.train_probe
run_stage "5/5 text_floor (crude text baseline)" experiment.text_floor

echo "=== smoke test complete in $((SECONDS - total_start))s total ==="
echo "Final numbers (also in experiment/outputs/results.json):"
"$PY" - <<'EOF'
import json
from experiment.config import CONFIG
r = json.load(open(CONFIG.results_path))
for layer, d in r["per_layer"].items():
    print(f"  {layer}: AUC={d['auc']} CI95={d['auc_ci95']}")
fl = r["shuffled_floor"]
tf = r["text_floor"]
best = r["per_layer"][r["best_layer"]]
print(f"  shuffled floor (max-across-layers, {fl['n_seeds']} seeds): "
      f"mean={fl['mean']} p95={fl['p95']}")
print(f"  text floor (prefix tokens + level): AUC={tf['auc']} CI95={tf['auc_ci95']}")
print(f"  verdict: {r['verdict']}")
print(f"GATE 1 FINAL: probe best {r['best_layer']} AUC={best['auc']} "
      f"vs shuffled floor p95={fl['p95']} vs text floor AUC={tf['auc']}")
EOF
