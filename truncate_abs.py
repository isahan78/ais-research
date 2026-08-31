"""Stage 2b: truncate each trace at a FIXED NUMBER of thinking tokens.

Why this module exists (RESULTS.md Run 007, EXPERIMENT.md §12b)
---------------------------------------------------------------
Cutting at k% of a trace makes the prefix length a fixed multiple of the FULL
trace length: `corr(prefix thinking tokens at k=50, full trace thinking tokens)
= 0.99999999`. Every reader — probe, TF-IDF, judge — is therefore handed the
trace's eventual length, which a real-time monitor cannot know, and prefix
length alone already scores ~0.70. Cutting at a fixed TOKEN COUNT removes the
leak structurally: within a cut every prefix is exactly the same length, so
length carries zero information by construction.

THE DESIGN TRAP THIS MODULE EXISTS TO AVOID
-------------------------------------------
The obvious implementation ("skip traces shorter than N") is wrong. Short
traces are disproportionately CORRECT, so a per-cut length exclusion is
LABEL-CORRELATED — it reintroduces exactly the survivor bias decision B
removed, and it makes the cuts incomparable because each N would be scored on
a different, differently-balanced set of problems.

So the population is FIXED ONCE, BEFORE ANY CUT:

    population = traces that are complete, gradeable, and have
                 >= CONFIG.abs_population_min_thinking (= max(N) = 1024)
                 thinking tokens

and the SAME problems are used at every N. `population_status()` — the only
function that decides inclusion — does not take N as an argument. That is not
a stylistic choice; it is the guarantee, and `tests/test_abs_and_confidence.py`
asserts it by signature.

Consequence to watch: restricting to long traces removes the easy/short items,
so the surviving class balance is NOT the dataset's. The meta line records the
survivor count and the balance, and `main()` prints a LOUD warning when the
surviving negative count is under 25 — below that the whole grid is
underpowered and the Δ curve should not be read as evidence.

Schema
------
The emitted prefixes.jsonl is a superset of the one `truncate.py` writes, so
`harvest_activations.py`, `train_probe.py`, `text_classifier.py`,
`forced_answer.py` and `forced_confidence.py` consume it unchanged. Extra
per-row fields (`k_percent`, `abs_n`, `cut_mode`) are additive; `k_percent`
carries the cut LABEL (= N) so `forced_answer.row_key` produces keys that are
unique across the grid and match the probe's own `pid@k<label>` keys.

Usage
-----
    EXPERIMENT_ABS_N=256 EXPERIMENT_OUTPUT_DIR=.../abs256 \
        python -m experiment.truncate_abs

    python -m experiment.truncate_abs population   # counts only, writes nothing

Pure stdlib: no GPU, no tokenizer, no network. The core functions are unit
tested on a laptop.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from experiment.config import CONFIG, THINK_END_ID, THINK_START_ID, lineage
except ImportError:  # run as a plain script from inside experiment/
    from config import CONFIG, THINK_END_ID, THINK_START_ID, lineage


CUT_MODE = "absolute"

# Exclusion reasons, all of them POPULATION-level (N-independent). There is
# deliberately no per-cut reason: if one ever appears here, the population is
# no longer fixed and the cuts are no longer comparable.
POPULATION_EXCLUSION_REASONS: Tuple[str, ...] = (
    "truncated_incomplete",   # generation hit max_tokens; no </think>
    "ungradeable",            # grader could not extract a final answer
    "thinking_too_short",     # degenerate thinking block
    "below_population_min",   # shorter than max(N) -> cannot be cut at every N
)

# Below this many surviving NEGATIVES the grid is underpowered; say so loudly.
MIN_NEGATIVES_FOR_POWER = 25


@dataclass
class AbsPrefixResult:
    """Same field set as `truncate.PrefixResult`, plus the absolute-cut fields."""

    problem_id: str
    included: bool
    exclusion_reason: Optional[str]
    label: Optional[bool]
    prefix_token_ids: Optional[List[int]]
    prompt_token_ids: Optional[List[int]]
    n_thinking_tokens: Optional[int]
    n_kept_thinking_tokens: Optional[int]
    # --- additive, absolute-cut only ---------------------------------------
    k_percent: int          # the cut LABEL (= abs_n). See module docstring.
    abs_n: int
    cut_mode: str = CUT_MODE


# ---------------------------------------------------------------------------
# Thinking-span extraction — mirrors truncate.build_prefix exactly
# ---------------------------------------------------------------------------

def thinking_span(trace_token_ids: Sequence[int]) -> Optional[Tuple[List[int], List[int]]]:
    """`(head, thinking)` for one completion, or None if it never closed `</think>`.

    `head` is the completion prefix up to and including `<think>` (empty when
    the chat template already rendered `<think>` into the prompt); `thinking`
    is everything between `<think>` and `</think>`.

    This is `truncate.build_prefix`'s span logic, lifted verbatim so the two
    stages cut the same span. A test asserts the two agree on shared inputs.
    """
    ids = list(trace_token_ids)
    if THINK_END_ID not in ids:
        return None
    end = ids.index(THINK_END_ID)
    if THINK_START_ID in ids[:end]:
        start = ids.index(THINK_START_ID)
        head = ids[: start + 1]
    else:
        start = -1
        head = []
    return head, ids[start + 1 : end]


# ---------------------------------------------------------------------------
# THE POPULATION FILTER — note the absence of an `n` parameter
# ---------------------------------------------------------------------------

def population_status(
    trace_token_ids: Sequence[int],
    label: Optional[bool],
    population_min_thinking: int = CONFIG.abs_population_min_thinking,
    min_thinking_tokens: int = CONFIG.min_thinking_tokens,
) -> Tuple[Optional[str], Optional[int]]:
    """`(exclusion_reason_or_None, n_thinking_or_None)` for one trace.

    ADMISSIBILITY INVARIANT: this function takes NO cut size. Membership of the
    absolute-cut population therefore cannot depend on N, which is what makes
    the same problems appear at every cut and keeps the exclusion from being
    label-correlated. Do not add an `n` parameter here.
    """
    span = thinking_span(trace_token_ids)
    if span is None:
        return "truncated_incomplete", None
    _head, thinking = span
    n_think = len(thinking)
    if n_think < min_thinking_tokens:
        return "thinking_too_short", n_think
    if label is None:
        return "ungradeable", n_think
    if n_think < population_min_thinking:
        return "below_population_min", n_think
    return None, n_think


def build_abs_prefix(
    problem_id: str,
    prompt_token_ids: Sequence[int],
    trace_token_ids: Sequence[int],
    label: Optional[bool],
    abs_n: int = CONFIG.truncation_abs_n or 0,
    population_min_thinking: int = CONFIG.abs_population_min_thinking,
    min_thinking_tokens: int = CONFIG.min_thinking_tokens,
) -> AbsPrefixResult:
    """Cut one trace at EXACTLY `abs_n` thinking tokens. Never raises on data.

    Raises only on a *configuration* error (`abs_n` outside 1..population_min),
    because an `abs_n` above the population minimum would silently turn into a
    per-cut, length-correlated exclusion.
    """
    if abs_n < 1:
        raise ValueError(f"abs_n must be >= 1, got {abs_n}")
    if abs_n > population_min_thinking:
        raise ValueError(
            f"abs_n={abs_n} > population_min_thinking={population_min_thinking}: the "
            f"population is fixed at >= max(N) thinking tokens precisely so that no "
            f"cut can exclude a trace. Raise the population minimum instead."
        )

    reason, n_think = population_status(
        trace_token_ids, label, population_min_thinking, min_thinking_tokens
    )
    if reason is not None:
        return AbsPrefixResult(
            problem_id=problem_id, included=False, exclusion_reason=reason,
            label=label, prefix_token_ids=None, prompt_token_ids=None,
            n_thinking_tokens=n_think, n_kept_thinking_tokens=None,
            k_percent=abs_n, abs_n=abs_n,
        )

    head, thinking = thinking_span(trace_token_ids)  # type: ignore[misc]
    n_keep = abs_n  # EXACT, never clamped: the population guarantees n_think >= abs_n
    kept = thinking[:n_keep]
    assert len(kept) == abs_n, (
        f"{problem_id}: kept {len(kept)} thinking tokens, expected exactly {abs_n} — "
        f"the population filter did not do its job"
    )
    prefix = list(prompt_token_ids) + head + kept
    assert THINK_END_ID not in head + kept, (
        f"{problem_id}: </think> leaked into prefix — truncation bug"
    )

    return AbsPrefixResult(
        problem_id=problem_id, included=True, exclusion_reason=None,
        label=label, prefix_token_ids=prefix, prompt_token_ids=list(prompt_token_ids),
        n_thinking_tokens=n_think, n_kept_thinking_tokens=n_keep,
        k_percent=abs_n, abs_n=abs_n,
    )


# ---------------------------------------------------------------------------
# Population report — cheap, GPU-free, run it BEFORE spending money
# ---------------------------------------------------------------------------

def read_traces(traces_path: str) -> Tuple[Optional[dict], List[dict]]:
    meta: Optional[dict] = None
    rows: List[dict] = []
    with open(traces_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("record_type") == "meta":
                meta = rec
            else:
                rows.append(rec)
    return meta, rows


def population_report(
    trace_rows: Sequence[dict],
    population_min_thinking: int = CONFIG.abs_population_min_thinking,
    min_thinking_tokens: int = CONFIG.min_thinking_tokens,
) -> dict:
    """Who survives the fixed population filter, and with what class balance."""
    counts: Dict[str, int] = {}
    survivors: List[str] = []
    n_pos = n_neg = 0
    for rec in trace_rows:
        reason, _n = population_status(
            rec["trace_token_ids"], rec.get("correct"),
            population_min_thinking, min_thinking_tokens,
        )
        if reason is None:
            survivors.append(rec["problem_id"])
            if rec["correct"]:
                n_pos += 1
            else:
                n_neg += 1
        else:
            counts[reason] = counts.get(reason, 0) + 1
    return {
        "population_min_thinking": population_min_thinking,
        "n_input": len(trace_rows),
        "n_population": len(survivors),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "neg_fraction": round(n_neg / len(survivors), 4) if survivors else None,
        "exclusion_counts": counts,
        "population_problem_ids": sorted(survivors),
        "underpowered": n_neg < MIN_NEGATIVES_FOR_POWER,
        "min_negatives_for_power": MIN_NEGATIVES_FOR_POWER,
    }


def _print_report(rep: dict, abs_n: Optional[int] = None) -> None:
    where = f" at N={abs_n}" if abs_n is not None else ""
    print(
        f"truncate_abs population{where}: {rep['n_population']}/{rep['n_input']} traces "
        f"survive the fixed filter (>= {rep['population_min_thinking']} thinking tokens), "
        f"class balance {rep['n_pos']} correct / {rep['n_neg']} incorrect "
        f"({rep['neg_fraction']} negative)"
    )
    print(f"  population exclusions: {rep['exclusion_counts'] or 'none'}")
    if rep["underpowered"]:
        print(
            "\n"
            "!!! " + "=" * 72 + "\n"
            f"!!! UNDERPOWERED: only {rep['n_neg']} negatives survive "
            f"(< {rep['min_negatives_for_power']}).\n"
            "!!! Every AUC on this population will have a CI wide enough to swallow the\n"
            "!!! effect, and Delta(N) must NOT be read as evidence either way. Fix by\n"
            "!!! generating more traces, or by lowering the population minimum AND the\n"
            "!!! top of the N grid together (EXPERIMENT_ABS_POP_MIN) — never by dropping\n"
            "!!! short traces per cut, which is label-correlated.\n"
            "!!! " + "=" * 72 + "\n",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------

def main() -> None:
    abs_n = CONFIG.truncation_abs_n
    if abs_n is None:
        raise SystemExit(
            "HALT: EXPERIMENT_ABS_N is not set. truncate_abs cuts at a fixed TOKEN "
            "COUNT; set e.g. EXPERIMENT_ABS_N=256 (and EXPERIMENT_OUTPUT_DIR), or run "
            "`python -m experiment.truncate` for the fixed-fraction protocol."
        )

    traces_path = CONFIG.traces_path
    meta_in, trace_rows = read_traces(traces_path)

    rep = population_report(trace_rows)
    _print_report(rep, abs_n)

    results = [
        build_abs_prefix(
            problem_id=rec["problem_id"],
            prompt_token_ids=rec["prompt_token_ids"],
            trace_token_ids=rec["trace_token_ids"],
            label=rec.get("correct"),
            abs_n=abs_n,
        )
        for rec in trace_rows
    ]

    if meta_in is not None and meta_in.get("config_hash") != CONFIG.config_hash():
        print(
            f"WARNING: traces.jsonl was produced under config {meta_in.get('config_hash')} "
            f"but current config is {CONFIG.config_hash()}",
            file=sys.stderr,
        )

    included = [r for r in results if r.included]
    counts: Dict[str, int] = {}
    for r in results:
        if not r.included:
            counts[r.exclusion_reason] = counts.get(r.exclusion_reason, 0) + 1

    # The two routes to "who is in" must agree, or the file lies about itself.
    assert {r.problem_id for r in included} == set(rep["population_problem_ids"]), (
        "population_report and build_abs_prefix disagree about the population"
    )
    assert all(r.n_kept_thinking_tokens == abs_n for r in included)

    with open(CONFIG.prefixes_path, "w") as f:
        meta = {
            "record_type": "meta",
            "stage": "truncate_abs",
            "cut_mode": CUT_MODE,
            "abs_n": abs_n,
            # cut LABEL, read by forced_answer.load_included_prefix_rows and
            # every text baseline as the default k for row keys.
            "k_percent": abs_n,
            "population_min_thinking": CONFIG.abs_population_min_thinking,
            "abs_n_grid": list(CONFIG.abs_n_grid),
            "n_input": len(results),
            "n_included": len(included),
            "exclusion_counts": counts,
            "population": {k: v for k, v in rep.items() if k != "population_problem_ids"},
            "population_problem_ids": rep["population_problem_ids"],
            **lineage(traces_path),
        }
        f.write(json.dumps(meta) + "\n")
        for r in results:
            row = asdict(r)
            row["config_hash"] = CONFIG.config_hash()
            f.write(json.dumps(row) + "\n")

    print(
        f"truncate_abs: {len(included)}/{len(results)} traces cut at EXACTLY {abs_n} "
        f"thinking tokens (exclusions: {counts or 'none'}) -> {CONFIG.prefixes_path}"
    )


def population_main() -> None:
    """`population` subcommand: report the fixed population, write nothing."""
    _meta, trace_rows = read_traces(CONFIG.traces_path)
    rep = population_report(trace_rows)
    _print_report(rep)
    print(json.dumps({k: v for k, v in rep.items() if k != "population_problem_ids"}, indent=2))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "truncate"
    if cmd == "truncate":
        main()
    elif cmd == "population":
        population_main()
    else:
        raise SystemExit("usage: python -m experiment.truncate_abs [truncate|population]")
