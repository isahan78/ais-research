#!/usr/bin/env bash
# Block 2: the full truncation grid on one pod session.
#
# Generation is the expensive part and is dataset-wide, so it runs ONCE.
# Truncation/harvest/probe are then re-run per k into their own output dir.
# The forced-answer baseline runs here too: it needs the model on the GPU,
# so doing it in a later session would mean paying for a second pod.
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$EXPERIMENT_DIR")"
cd "$PROJECT_ROOT"

PY="${PYTHON:-python}"
K_GRID="${K_GRID:-10 25 50 75 90}"
BASE_OUT="${BASE_OUT:-$EXPERIMENT_DIR/outputs/block2}"
mkdir -p "$BASE_OUT"
LOG="$BASE_OUT/run_grid.log"
exec > >(tee "$LOG") 2>&1
echo "run_grid.sh started $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "k grid: $K_GRID   base out: $BASE_OUT"
echo

step() { echo; echo "=== $* ==="; date -u +%Y-%m-%dT%H:%M:%SZ; }

step "0/N pytest (no GPU spend if this fails)"
$PY -m pytest experiment/tests/ -q

# --- generation: once, shared by every k -----------------------------------
GEN_OUT="$BASE_OUT/gen"
mkdir -p "$GEN_OUT"
if [ -f "$GEN_OUT/traces.jsonl" ]; then
  echo "traces.jsonl already present in $GEN_OUT — skipping generation (resumable)"
else
  step "1/N generate_traces (vLLM) -> $GEN_OUT"
  EXPERIMENT_OUTPUT_DIR="$GEN_OUT" $PY -m experiment.generate_traces
fi

# --- per-k: truncate -> harvest -> probe ------------------------------------
for K in $K_GRID; do
  KOUT="$BASE_OUT/k$K"
  mkdir -p "$KOUT"
  cp -f "$GEN_OUT/traces.jsonl" "$KOUT/traces.jsonl"
  step "k=$K truncate"
  EXPERIMENT_K="$K" EXPERIMENT_OUTPUT_DIR="$KOUT" $PY -m experiment.truncate
  step "k=$K harvest_activations"
  EXPERIMENT_K="$K" EXPERIMENT_OUTPUT_DIR="$KOUT" $PY -m experiment.harvest_activations
  step "k=$K train_probe"
  EXPERIMENT_K="$K" EXPERIMENT_OUTPUT_DIR="$KOUT" $PY -m experiment.train_probe || echo "WARNING: probe failed at k=$K; continuing grid"
  step "k=$K text_floor (crude)"
  EXPERIMENT_K="$K" EXPERIMENT_OUTPUT_DIR="$KOUT" $PY -m experiment.text_floor || echo "WARNING: text_floor failed at k=$K"
done

# --- forced-answer baseline: MUST be this session (needs the model) ---------
if [ -f "$EXPERIMENT_DIR/forced_answer.py" ]; then
  for K in $K_GRID; do
    step "k=$K forced_answer (on-policy text baseline)"
    EXPERIMENT_K="$K" EXPERIMENT_OUTPUT_DIR="$BASE_OUT/k$K" \
      $PY -m experiment.forced_answer || echo "WARNING: forced_answer failed at k=$K"
  done
else
  echo; echo "NOTE: forced_answer.py absent — skipping. It needs the GPU, so a"
  echo "later session would cost a second pod."
fi

echo; echo "=== GRID COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "Copy $BASE_OUT off the pod BEFORE terminating — scrollback dies with it."
