#!/usr/bin/env bash
# Run 008 / 009: the FIXED-LENGTH (absolute token) truncation grid, plus the
# gold-free forced-answer CONFIDENCE baseline, on one pod session.
#
# Why this exists rather than more flags on run_grid.sh: the two protocols cut
# differently (fraction vs token count) and must never be mixed in one output
# tree. EXPERIMENT_K and EXPERIMENT_ABS_N are mutually exclusive by config.
#
# Generation is the expensive part and is dataset-wide, so it runs ONCE and is
# shared with the k%-grid if that already ran (point GEN_OUT at it).
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$EXPERIMENT_DIR")"
cd "$PROJECT_ROOT"

PY="${PYTHON:-python}"
N_GRID="${N_GRID:-64 128 256 512 1024}"
BASE_OUT="${BASE_OUT:-$EXPERIMENT_DIR/outputs/block3_abs}"
GEN_OUT="${GEN_OUT:-$BASE_OUT/gen}"
mkdir -p "$BASE_OUT" "$GEN_OUT"
LOG="$BASE_OUT/run_abs_grid.log"
exec > >(tee "$LOG") 2>&1
echo "run_abs_grid.sh started $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "N grid: $N_GRID   base out: $BASE_OUT   generation: $GEN_OUT"
echo

step() { echo; echo "=== $* ==="; date -u +%Y-%m-%dT%H:%M:%SZ; }

step "0/N pytest (no GPU spend if this fails)"
$PY -m pytest experiment/tests/ -q

# --- generation: once, shared by every N ------------------------------------
if [ -f "$GEN_OUT/traces.jsonl" ]; then
  echo "traces.jsonl already present in $GEN_OUT — skipping generation (resumable)"
else
  step "1/N generate_traces (vLLM) -> $GEN_OUT"
  EXPERIMENT_OUTPUT_DIR="$GEN_OUT" $PY -m experiment.generate_traces
fi

# --- POWER CHECK FIRST: the fixed population is the whole experiment ---------
# If fewer than ~25 negatives survive the >=1024-thinking-token filter, every
# AUC below is noise and the grid is not worth the GPU minutes. This prints the
# survivor count and class balance and shouts if it is underpowered.
step "1b/N fixed-population report (CPU, seconds) — READ THIS BEFORE CONTINUING"
cp -f "$GEN_OUT/traces.jsonl" "$BASE_OUT/traces.jsonl"
EXPERIMENT_ABS_N=1024 EXPERIMENT_OUTPUT_DIR="$BASE_OUT" \
  $PY -m experiment.truncate_abs population

# --- per-N: truncate_abs -> forced_confidence -> harvest -> probe ------------
for N in $N_GRID; do
  NOUT="$BASE_OUT/abs$N"
  mkdir -p "$NOUT"
  cp -f "$GEN_OUT/traces.jsonl" "$NOUT/traces.jsonl"
  export EXPERIMENT_ABS_N="$N" EXPERIMENT_OUTPUT_DIR="$NOUT"

  step "N=$N truncate_abs (exactly $N thinking tokens; fixed population)"
  $PY -m experiment.truncate_abs

  step "N=$N forced_confidence generate (GPU — must be this session)"
  $PY -m experiment.forced_confidence generate \
    || echo "WARNING: forced_confidence generate failed at N=$N"

  step "N=$N harvest_activations"
  $PY -m experiment.harvest_activations

  step "N=$N train_probe"
  $PY -m experiment.train_probe || echo "WARNING: probe failed at N=$N; continuing grid"

  step "N=$N text_floor (crude)"
  # NOTE: under absolute cuts the prefix-length feature is constant + prompt
  # length, so this floor collapses to a prompt-length/difficulty reader. That
  # is the expected behaviour, not a bug — it is the length leak going away.
  $PY -m experiment.text_floor || echo "WARNING: text_floor failed at N=$N"

  unset EXPERIMENT_ABS_N EXPERIMENT_OUTPUT_DIR
done

# --- CPU-side scoring: safe to redo off-pod, but free to do here -------------
for N in $N_GRID; do
  NOUT="$BASE_OUT/abs$N"
  export EXPERIMENT_ABS_N="$N" EXPERIMENT_OUTPUT_DIR="$NOUT"

  step "N=$N forced_confidence score (gold-free; EXPERIMENT.md 12b)"
  $PY -m experiment.forced_confidence score \
    || echo "WARNING: forced_confidence score failed at N=$N"

  step "N=$N text_classifier (tuned TF-IDF)"
  # No forced_answer.jsonl in this tree, so this decodes prefixes with the
  # tokenizer (cheap, weights already cached) and prints a WARNING saying so.
  $PY -m experiment.text_classifier || echo "WARNING: text_classifier failed at N=$N"

  unset EXPERIMENT_ABS_N EXPERIMENT_OUTPUT_DIR
done

echo; echo "=== ABS GRID COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "Copy $BASE_OUT off the pod BEFORE terminating — scrollback dies with it."
