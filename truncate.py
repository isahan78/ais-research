"""Stage 2: truncate each trace at k% of its thinking tokens.

Reads traces.jsonl, writes prefixes.jsonl. Pure stdlib: the core function
`build_prefix` is unit-tested on machines with no GPU or model weights.

This is the single most bug-prone step (token-boundary correctness), so the
core logic is a pure function over token ids with explicit exclusion reasons,
and it guarantees the emitted prefix ends strictly INSIDE the thinking span:
after `<think>`, before `</think>`, with `</think>` never present.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from typing import List, Optional

try:
    from experiment.config import CONFIG, THINK_END_ID, THINK_START_ID, lineage
except ImportError:  # run as a plain script from inside experiment/
    from config import CONFIG, THINK_END_ID, THINK_START_ID, lineage


@dataclass
class PrefixResult:
    problem_id: str
    included: bool
    exclusion_reason: Optional[str]          # None | "truncated_incomplete" | "thinking_too_short" | "ungradeable"
    label: Optional[bool]
    prefix_token_ids: Optional[List[int]]    # prompt + <think> + first n_keep thinking tokens
    prompt_token_ids: Optional[List[int]]    # carried through for the stage-3 prefix-match assertion
    n_thinking_tokens: Optional[int]
    n_kept_thinking_tokens: Optional[int]


def build_prefix(
    problem_id: str,
    prompt_token_ids: List[int],
    trace_token_ids: List[int],
    label: Optional[bool],
    k_percent: int = CONFIG.truncation_k_percent,
    min_thinking_tokens: int = CONFIG.min_thinking_tokens,
) -> PrefixResult:
    """Truncate one trace at k% of thinking tokens. Never raises on malformed input.

    I/O matrix:
      - no </think> in trace       -> excluded, reason "truncated_incomplete"
      - thinking span < min tokens -> excluded, reason "thinking_too_short"
      - label is None              -> excluded, reason "ungradeable"
    """

    def excluded(reason: str) -> PrefixResult:
        return PrefixResult(
            problem_id=problem_id, included=False, exclusion_reason=reason,
            label=label, prefix_token_ids=None, prompt_token_ids=None,
            n_thinking_tokens=None, n_kept_thinking_tokens=None,
        )

    # 1) Locate the thinking span in the COMPLETION token ids.
    if THINK_END_ID not in trace_token_ids:
        return excluded("truncated_incomplete")  # generation hit max_tokens mid-thinking
    end = trace_token_ids.index(THINK_END_ID)

    if THINK_START_ID in trace_token_ids[:end]:
        start = trace_token_ids.index(THINK_START_ID)
        head = trace_token_ids[: start + 1]      # completion tokens up to and incl. <think>
    else:
        # Chat template already rendered `<think>` into the prompt; completion
        # begins directly with thinking tokens.
        start = -1
        head = []

    thinking = trace_token_ids[start + 1 : end]
    n_think = len(thinking)

    # 2) Degenerate / ungradeable exclusions.
    if n_think < min_thinking_tokens:
        return excluded("thinking_too_short")
    if label is None:
        return excluded("ungradeable")

    # 3) Keep k% of thinking tokens, clamped strictly inside the span so the
    #    prefix always ends mid-thinking (>=1 kept, <= n_think - 1).
    n_keep = int(n_think * k_percent / 100)
    n_keep = max(1, min(n_keep, n_think - 1))

    prefix = list(prompt_token_ids) + head + thinking[:n_keep]

    assert THINK_END_ID not in head + thinking[:n_keep], (
        f"{problem_id}: </think> leaked into prefix — truncation bug"
    )

    return PrefixResult(
        problem_id=problem_id, included=True, exclusion_reason=None,
        label=label, prefix_token_ids=prefix, prompt_token_ids=list(prompt_token_ids),
        n_thinking_tokens=n_think, n_kept_thinking_tokens=n_keep,
    )


def main() -> None:
    # Absolute-cut mode sets truncation_k_percent to the TOKEN COUNT N as a cut
    # label (see config.py), which this fraction-based stage would silently
    # misread as "N percent" and clamp to nearly the whole trace. Refuse.
    if getattr(CONFIG, "truncation_abs_n", None) is not None:
        raise SystemExit(
            f"HALT: EXPERIMENT_ABS_N={CONFIG.truncation_abs_n} is set, but this is the "
            f"fixed-FRACTION truncation stage. Run `python -m experiment.truncate_abs` "
            f"instead (or unset EXPERIMENT_ABS_N)."
        )

    traces_path = CONFIG.traces_path
    results: List[PrefixResult] = []
    meta_in = None

    with open(traces_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("record_type") == "meta":
                meta_in = rec
                continue
            results.append(
                build_prefix(
                    problem_id=rec["problem_id"],
                    prompt_token_ids=rec["prompt_token_ids"],
                    trace_token_ids=rec["trace_token_ids"],
                    label=rec["correct"],
                )
            )

    if meta_in is not None and meta_in.get("config_hash") != CONFIG.config_hash():
        print(
            f"WARNING: traces.jsonl was produced under config {meta_in.get('config_hash')} "
            f"but current config is {CONFIG.config_hash()}",
            file=sys.stderr,
        )

    n_inc = sum(r.included for r in results)
    counts = {}
    for r in results:
        if not r.included:
            counts[r.exclusion_reason] = counts.get(r.exclusion_reason, 0) + 1
            print(f"WARNING: excluding {r.problem_id}: {r.exclusion_reason}", file=sys.stderr)

    with open(CONFIG.prefixes_path, "w") as f:
        meta = {
            "record_type": "meta",
            "stage": "truncate",
            "k_percent": CONFIG.truncation_k_percent,
            "n_input": len(results),
            "n_included": n_inc,
            "exclusion_counts": counts,
            **lineage(traces_path),
        }
        f.write(json.dumps(meta) + "\n")
        for r in results:
            row = asdict(r)
            row["config_hash"] = CONFIG.config_hash()
            f.write(json.dumps(row) + "\n")

    print(f"truncate: {n_inc}/{len(results)} traces included at k={CONFIG.truncation_k_percent}% "
          f"(exclusions: {counts or 'none'}) -> {CONFIG.prefixes_path}")


if __name__ == "__main__":
    main()
