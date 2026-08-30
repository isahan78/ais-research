"""Single source of truth for the Gate 1 smoke pipeline.

Every stage imports CONFIG from here and stamps CONFIG.config_hash() into its
output so any reported number is traceable to the exact settings that
produced it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

# Token ids verified against Qwen/Qwen3-8B tokenizer (agent-checked 2026-08-16).
# Both are special:false, so skip_special_tokens=True preserves them in decodes.
THINK_START_ID = 151667  # <think>
THINK_END_ID = 151668    # </think>

EXPERIMENT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Config:
    # --- model / data -------------------------------------------------------
    model_id: str = "Qwen/Qwen3-8B"
    # dataset_kind selects the row->record adapter in dataset_adapters.py.
    # "mmlu_pro" adopted 2026-08-26 on a MEASURED base rate (RESULTS.md Run 004:
    # 23% error, 0 ungradeable, 12,032 items) after MATH-500 was exhausted
    # (11% error, only 134 level-5 items => ~15 negatives). "math500" is kept
    # selectable so Runs 001-003 reproduce.
    dataset_kind: str = "mmlu_pro"     # "mmlu_pro" | "math500"
    dataset_id: str = "TIGER-Lab/MMLU-Pro"
    dataset_split: str = "test"
    n_problems: int = 300
    levels: Tuple[int, ...] = (4, 5)   # decision A; applies to math500 ONLY (MMLU-Pro has no level)

    # --- generation (stage 1, vLLM) ----------------------------------------
    max_model_len: int = 24576   # 16384 thinking + 8192 prompt headroom: MMLU-Pro items with 10 long options can exceed 4k, and vLLM would silently shorten that item's thinking budget — reintroducing the label-correlated truncation decision B removed
    max_new_tokens: int = 16384        # decision B, validated Run 002: 8k truncated 30% of traces, 16k truncates 8%
    temperature: float = 0.6           # Qwen3 thinking-mode recommended sampling
    top_p: float = 0.95
    gpu_memory_utilization: float = 0.90

    # --- truncation (stage 2) ----------------------------------------------
    truncation_k_percent: int = 50     # single point for Gate 1; the full grid is deferred
    min_thinking_tokens: int = 4       # thinking blocks shorter than this are degenerate

    # --- harvest (stage 3, HF prefill-only) ---------------------------------
    layers: Tuple[int, ...] = (9, 18, 27)  # ~25/50/75% depth of 36; all below the [-1] post-norm trap
    num_decoder_layers: int = 36
    hidden_size: int = 4096
    harvest_batch_size: int = 1        # 20 short prefills; batching buys nothing and risks padding bugs

    # --- probe (stage 4) -----------------------------------------------------
    seed: int = 0
    test_fraction: float = 0.35
    probe_C: float = 1.0               # logistic regularization; in the hash so a sweep is traceable
    n_bootstrap: int = 1000
    n_shuffle_seeds: int = 500  # floor seeds are cheap; 500 gives the p95 real resolution
    max_split_retries: int = 50        # retry GroupShuffleSplit seeds until both classes land in both splits

    # --- quality gates -------------------------------------------------------
    max_ungradeable_fraction: float = 0.30  # HALT above this (I/O matrix row 4)
    max_incomplete_fraction: float = 0.34   # HALT above this: label-correlated truncation bias (row 6)
    min_included_rows: int = 12             # HALT below this: refuse to fit a probe on degenerate data

    # --- paths ----------------------------------------------------------------
    output_dir: str = str(EXPERIMENT_DIR / "outputs")
    traces_path: str = str(EXPERIMENT_DIR / "outputs" / "traces.jsonl")
    prefixes_path: str = str(EXPERIMENT_DIR / "outputs" / "prefixes.jsonl")
    acts_path: str = str(EXPERIMENT_DIR / "outputs" / "acts.npz")
    results_path: str = str(EXPERIMENT_DIR / "outputs" / "results.json")

    def config_hash(self) -> str:
        """Deterministic 12-hex-char hash of every field except the paths.

        Paths are machine-specific and excluded so the same experimental
        settings hash identically on any box.
        """
        d = dataclasses.asdict(self)
        for k in ("output_dir", "traces_path", "prefixes_path", "acts_path", "results_path"):
            d.pop(k)
        blob = json.dumps(d, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:12]


CONFIG = Config()


def lineage(input_file: str | None) -> dict:
    """Standard lineage stamp every stage writes into its output."""
    return {
        "config_hash": CONFIG.config_hash(),
        "input_file": input_file,
    }
