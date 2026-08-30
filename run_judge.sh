#!/usr/bin/env bash
# Run the LLM-judge baseline with the API key scoped to THIS process only.
#
# Why not export ANTHROPIC_API_KEY globally: a global key can push Claude Code
# onto metered API billing instead of the Max subscription (see the note in
# ~/.zshrc). And after a key was once leaked into a terminal, keeping the
# secret in the macOS Keychain rather than a dotfile is the cheap upgrade.
#
# One-time setup (run in YOUR terminal, never through an agent):
#   security add-generic-password -a "$USER" -s MATS_ANTHROPIC_KEY -w
#     ...then paste the key at the silent prompt and press return.
#
# Usage:
#   bash experiment/run_judge.sh                      # default k from config
#   EXPERIMENT_K=50 EXPERIMENT_OUTPUT_DIR=... bash experiment/run_judge.sh
set -euo pipefail

KEYCHAIN_SERVICE="${KEYCHAIN_SERVICE:-MATS_ANTHROPIC_KEY}"

if ! KEY=$(security find-generic-password -a "$USER" -s "$KEYCHAIN_SERVICE" -w 2>/dev/null); then
  echo "No key in Keychain under service '$KEYCHAIN_SERVICE'." >&2
  echo "Store one with:" >&2
  echo "  security add-generic-password -a \"\$USER\" -s $KEYCHAIN_SERVICE -w" >&2
  echo "(omit the value; it prompts silently so the key never enters shell history)" >&2
  exit 1
fi

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$EXPERIMENT_DIR")"

# Scoped to this one command: not exported to the shell, not inherited by
# anything else, never written to disk.
ANTHROPIC_API_KEY="$KEY" "${PYTHON:-python}" -m experiment.llm_judge "$@"
