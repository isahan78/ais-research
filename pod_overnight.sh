#!/usr/bin/env bash
# Overnight expansion session (approved 2026-09-01, ~$5):
#   1,000 traces regenerated fresh -> ~180-200 negatives expected
#   k% grid (6 cuts) + fixed-length grid (5 cuts)
#   ALL 35 usable layers harvested (0-34; 35 is the post-norm trap)
#   gold-free forced-confidence at every cut (commitment curve comes free)
# Probes/TF-IDF/cross-fitting run LOCALLY afterwards - nothing here needs them.
# Fully self-driving: every stage nohup-safe, milestone markers for the monitor.
set -euo pipefail
EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$EXPERIMENT_DIR")"
PY="${PYTHON:-python}"
ALL_LAYERS="$(python3 - <<'P'
print(",".join(str(i) for i in range(35)))
P
)"
BASE="${BASE_OUT:-$EXPERIMENT_DIR/outputs/expansion}"
GEN="$BASE/gen"; mkdir -p "$GEN"
LOG="$BASE/pod_overnight.log"
exec > >(tee "$LOG") 2>&1
echo "OVERNIGHT SESSION START $(date -u +%FT%TZ)  base=$BASE"

echo "=== M0 pytest gate ==="
$PY -m pytest experiment/tests/ -q
echo "M0_TESTS_OK"

echo "=== M1 generate 1000 traces (the long pole, ~3h) ==="
if [ -f "$GEN/traces.jsonl" ]; then echo "traces exist - resuming past generation"; else
  EXPERIMENT_N_PROBLEMS=1000 EXPERIMENT_OUTPUT_DIR="$GEN" $PY -m experiment.generate_traces
fi
echo "M1_GENERATION_DONE"

for K in 1 10 25 50 75 90; do
  KOUT="$BASE/k$K"; mkdir -p "$KOUT"; cp -f "$GEN/traces.jsonl" "$KOUT/traces.jsonl"
  echo "=== M2 k=$K truncate ==="
  EXPERIMENT_N_PROBLEMS=1000 EXPERIMENT_K=$K EXPERIMENT_OUTPUT_DIR="$KOUT" $PY -m experiment.truncate
  echo "=== M2 k=$K forced_confidence (GPU) ==="
  EXPERIMENT_N_PROBLEMS=1000 EXPERIMENT_K=$K EXPERIMENT_OUTPUT_DIR="$KOUT" $PY -m experiment.forced_confidence || echo "WARN conf k=$K"
  echo "=== M2 k=$K harvest ALL LAYERS ==="
  EXPERIMENT_N_PROBLEMS=1000 EXPERIMENT_K=$K EXPERIMENT_LAYERS="$ALL_LAYERS" EXPERIMENT_OUTPUT_DIR="$KOUT" $PY -m experiment.harvest_activations || echo "WARN harvest k=$K"
  echo "M2_K${K}_DONE"
done

echo "=== M3 fixed-length population gate ==="
POPOUT="$BASE/popcheck"; mkdir -p "$POPOUT"; cp -f "$GEN/traces.jsonl" "$POPOUT/traces.jsonl"
EXPERIMENT_N_PROBLEMS=1000 EXPERIMENT_ABS_N=1024 EXPERIMENT_OUTPUT_DIR="$POPOUT" $PY -m experiment.truncate_abs population
for N in 64 128 256 512 1024; do
  NOUT="$BASE/abs$N"; mkdir -p "$NOUT"; cp -f "$GEN/traces.jsonl" "$NOUT/traces.jsonl"
  echo "=== M3 N=$N truncate_abs ==="
  EXPERIMENT_N_PROBLEMS=1000 EXPERIMENT_ABS_N=$N EXPERIMENT_OUTPUT_DIR="$NOUT" $PY -m experiment.truncate_abs
  echo "=== M3 N=$N forced_confidence (GPU) ==="
  EXPERIMENT_N_PROBLEMS=1000 EXPERIMENT_ABS_N=$N EXPERIMENT_OUTPUT_DIR="$NOUT" $PY -m experiment.forced_confidence || echo "WARN conf N=$N"
  echo "=== M3 N=$N harvest ALL LAYERS ==="
  EXPERIMENT_N_PROBLEMS=1000 EXPERIMENT_ABS_N=$N EXPERIMENT_LAYERS="$ALL_LAYERS" EXPERIMENT_OUTPUT_DIR="$NOUT" $PY -m experiment.harvest_activations || echo "WARN harvest N=$N"
  echo "M3_N${N}_DONE"
done
echo "OVERNIGHT_SESSION_COMPLETE $(date -u +%FT%TZ)"
