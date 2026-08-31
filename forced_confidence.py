"""Text baseline: FORCED-ANSWER CONFIDENCE — the gold-free replacement.

Why this module exists (RESULTS.md Run 007, EXPERIMENT.md §12b)
---------------------------------------------------------------
`forced_answer.py` was WITHDRAWN as a correctness predictor on 2026-08-30: its
score was `grade(forced_answer, GOLD)`, so it read the answer key at test time.
Whenever the interrupted answer equalled the trace's final answer the score WAS
the label by construction; on the rows where they differed its AUC was 0.000.

This module keeps the good half of that idea — interrupt the model mid-thought
and make it commit — and throws away the part that cheated. Instead of asking
"was the forced answer RIGHT?" (needs gold) it asks "how CONFIDENT was the
model in the answer it blurted out?" (needs only the model's own logprobs). A
real monitor can compute this: it has the prefix and the model, and nothing
else.

ADMISSIBILITY (EXPERIMENT.md §12b, the rule that withdrew the predecessor)
--------------------------------------------------------------------------
    ASSERTION: the score of every row is a function of the PREFIX ALONE.

    The forced prompt is `prefix + </think> + FORCED_ANSWER_SUFFIX`, all of
    which derive from the prefix. The score is a transform of the model's
    next-token distribution over that prompt. The gold answer is never read,
    never passed, and never imported: this module does not import
    `experiment.grading`, calls no grader, and no function in the scoring path
    takes a gold/answer-key parameter. Gold enters the pipeline only as the
    LABEL that the AUC is measured against — which is evaluation, not scoring,
    and is the same affordance the probe gets.

    `tests/test_abs_and_confidence.py` enforces this by inspection (no
    grading import, no gold-shaped parameter anywhere in the scoring path) and
    empirically (rescoring with deliberately corrupted gold answers leaves
    every score and every AUC bit-identical).

What is recorded
----------------
The top-k next-token distribution at the answer position is written to
`forced_confidence.jsonl`, not just the winning probability, so entropy /
margin / renormalised-over-letters variants can be computed later on a laptop
WITHOUT re-renting a GPU. `confidence_variants()` computes all of them from the
stored record.

RUN ORDER — the generate half must run in the SAME pod session as generation:

    python -m experiment.generate_traces                # stage 1 (vLLM)
    python -m experiment.truncate_abs                   # stage 2b (CPU)
    python -m experiment.forced_confidence              # THIS, on the GPU box
    ...                                                 # harvest / train_probe
    python -m experiment.forced_confidence score        # CPU, after train_probe

Heavy imports (vllm, transformers) live inside `main()`, so importing this
module on a laptop is free — the offline test gate depends on that.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

try:
    from experiment.config import CONFIG, lineage
    from experiment.forced_answer import (
        FORCED_ANSWER_MAX_TOKENS,
        FORCED_ANSWER_SUFFIX,
        FORCED_ANSWER_TEMPERATURE,
        FORCED_ANSWER_TOP_P,
        build_forced_prompt_ids,
        group_rows_by_k,
        load_included_prefix_rows,
        read_jsonl,
        row_k,
        row_key,
    )
except ImportError:  # run as a plain script from inside experiment/
    from config import CONFIG, lineage  # type: ignore
    from forced_answer import (  # type: ignore
        FORCED_ANSWER_MAX_TOKENS,
        FORCED_ANSWER_SUFFIX,
        FORCED_ANSWER_TEMPERATURE,
        FORCED_ANSWER_TOP_P,
        build_forced_prompt_ids,
        group_rows_by_k,
        load_included_prefix_rows,
        read_jsonl,
        row_k,
        row_key,
    )

# NOTE (admissibility): `experiment.grading` is deliberately NOT imported here.
# Nothing in this file may grade anything.

BASELINE_NAME = "forced_confidence"
FORCED_CONFIDENCE_PATH = os.path.join(CONFIG.output_dir, "forced_confidence.jsonl")

# MMLU-Pro answers are single letters A..J (dataset_adapters caps options at 10).
ANSWER_LETTERS: Tuple[str, ...] = tuple("ABCDEFGHIJ")

# How many top alternatives to ask vLLM for at each generated position. 20 is
# comfortably more than the 10 answer letters, so the full letter distribution
# is nearly always captured, and it costs nothing at generation time.
CONFIDENCE_TOP_LOGPROBS = 20

# The forcing suffix opens `\boxed{`, so the answer letter is normally the very
# first generated token; allow a few positions of slack for a stray space or
# newline before giving up and falling back to position 0.
ANSWER_SCAN_POSITIONS = 4

# Variants computable from the stored distribution. The primary one is
# pre-registered (EXPERIMENT.md §12b: "AUC of the answer-token logprob against
# the true label"); the rest are recorded so a post-hoc variant needs no GPU,
# and any switch away from the default must be labelled post-hoc.
DEFAULT_VARIANT = "p_top"
CONFIDENCE_VARIANTS: Tuple[str, ...] = (
    "p_top",              # P(the token the model actually emitted) -- pre-registered
    "logprob_top",        # its logprob (a monotone transform, so identical AUC)
    "p_top_letter",       # P(the most likely ANSWER LETTER)
    "p_letter_norm",      # that, renormalised over the answer letters only
    "letter_margin",      # top letter minus runner-up, renormalised
    "neg_letter_entropy", # negative entropy of the renormalised letter distribution
)


def selected_variant() -> str:
    """Which variant `score` uses. Overridable, and recorded in the artifact."""
    v = os.environ.get("EXPERIMENT_CONFIDENCE_VARIANT", DEFAULT_VARIANT)
    if v not in CONFIDENCE_VARIANTS:
        raise SystemExit(
            f"HALT: EXPERIMENT_CONFIDENCE_VARIANT={v!r} is not one of {CONFIDENCE_VARIANTS}"
        )
    return v


# ---------------------------------------------------------------------------
# Pure logprob handling — unit-tested with fakes, no GPU, no tokenizer
# ---------------------------------------------------------------------------

def normalize_answer_token(text: Optional[str]) -> Optional[str]:
    """`" c"` -> `"C"`; anything that is not a single answer letter -> None."""
    t = (text or "").strip()
    if len(t) == 1 and t.upper() in ANSWER_LETTERS:
        return t.upper()
    return None


def _logprob_value(entry) -> float:
    """vLLM hands back `Logprob` objects; tests hand back bare floats."""
    lp = getattr(entry, "logprob", entry)
    return float(lp)


def _logprob_text(entry, token_id: int, decode: Optional[Callable[[Sequence[int]], str]]) -> str:
    text = getattr(entry, "decoded_token", None)
    if text is None and decode is not None:
        text = decode([token_id])
    return text if text is not None else ""


def token_distribution(
    logprob_map: Dict[int, object],
    decode: Optional[Callable[[Sequence[int]], str]] = None,
) -> List[dict]:
    """vLLM's `{token_id: Logprob}` at one position -> JSON rows, best first."""
    rows = [
        {
            "token_id": int(tid),
            "token": _logprob_text(entry, int(tid), decode),
            "logprob": _logprob_value(entry),
        }
        for tid, entry in logprob_map.items()
    ]
    rows.sort(key=lambda r: r["logprob"], reverse=True)
    return rows


def letter_logprobs(topk: Sequence[dict]) -> Dict[str, float]:
    """Best logprob seen for each answer letter in the recorded top-k."""
    out: Dict[str, float] = {}
    for r in topk:
        letter = normalize_answer_token(r.get("token"))
        if letter is None:
            continue
        lp = float(r["logprob"])
        if letter not in out or lp > out[letter]:
            out[letter] = lp
    return out


def extract_answer_distribution(
    token_ids: Sequence[int],
    logprobs_per_pos: Optional[Sequence[Dict[int, object]]],
    decode: Optional[Callable[[Sequence[int]], str]] = None,
    max_scan: int = ANSWER_SCAN_POSITIONS,
) -> Optional[dict]:
    """The recorded confidence payload for one forced generation.

    Finds the first generated position whose emitted token is an answer letter
    (falling back to position 0), and records the emitted token plus the whole
    top-k distribution there. No gold, no grading — just the model's own
    next-token distribution.
    """
    if not token_ids or not logprobs_per_pos:
        return None

    n = min(len(token_ids), len(logprobs_per_pos))
    if n == 0:
        return None

    pos = 0
    for i in range(min(n, max_scan)):
        entry = logprobs_per_pos[i].get(int(token_ids[i])) if logprobs_per_pos[i] else None
        text = _logprob_text(entry, int(token_ids[i]), decode) if entry is not None else (
            decode([int(token_ids[i])]) if decode is not None else ""
        )
        if normalize_answer_token(text) is not None:
            pos = i
            break

    lp_map = logprobs_per_pos[pos] or {}
    topk = token_distribution(lp_map, decode)
    chosen_id = int(token_ids[pos])
    chosen_entry = lp_map.get(chosen_id)
    chosen_token = (
        _logprob_text(chosen_entry, chosen_id, decode)
        if chosen_entry is not None
        else (decode([chosen_id]) if decode is not None else "")
    )
    if chosen_entry is not None:
        chosen_logprob = _logprob_value(chosen_entry)
    elif topk:
        # The sampled token is normally in its own top-k; if a backend omits it,
        # fall back to the best alternative rather than inventing a number.
        chosen_logprob = topk[0]["logprob"]
    else:
        return None

    return {
        "answer_position": pos,
        "chosen_token_id": chosen_id,
        "chosen_token": chosen_token,
        "chosen_logprob": float(chosen_logprob),
        "chosen_letter": normalize_answer_token(chosen_token),
        "topk": topk,
        "letter_logprobs": letter_logprobs(topk),
    }


def confidence_variants(rec: Optional[dict]) -> Dict[str, Optional[float]]:
    """Every confidence statistic derivable from one stored record.

    Gold-free by construction: the only input is the model's own distribution.
    Missing statistics are None (the caller drops those rows and counts them)
    rather than being imputed, which would quietly move an AUC.
    """
    out: Dict[str, Optional[float]] = {v: None for v in CONFIDENCE_VARIANTS}
    if not rec:
        return out

    lp_top = rec.get("chosen_logprob")
    if lp_top is not None:
        out["logprob_top"] = float(lp_top)
        out["p_top"] = float(math.exp(float(lp_top)))

    letters = rec.get("letter_logprobs") or {}
    if letters:
        probs = {l: math.exp(float(lp)) for l, lp in letters.items()}
        ordered = sorted(probs.values(), reverse=True)
        total = sum(probs.values())
        out["p_top_letter"] = float(ordered[0])
        if total > 0:
            norm = [p / total for p in ordered]
            out["p_letter_norm"] = float(norm[0])
            if len(norm) >= 2:
                out["letter_margin"] = float(norm[0] - norm[1])
            out["neg_letter_entropy"] = float(
                sum(p * math.log(p) for p in norm if p > 0)  # == -H
            )
    return out


# ---------------------------------------------------------------------------
# Stage A: generate (GPU, vLLM, same pod session as generate_traces)
# ---------------------------------------------------------------------------

def main() -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    t0 = time.time()
    os.makedirs(CONFIG.output_dir, exist_ok=True)

    rows, default_k = load_included_prefix_rows()
    if not rows:
        raise SystemExit(
            "HALT: no included rows in prefixes.jsonl — run truncate/truncate_abs first."
        )
    # NOTE: unlike forced_answer.main this stage does NOT load gold answers.
    # There is nothing here to look them up for.

    tokenizer = AutoTokenizer.from_pretrained(CONFIG.model_id)
    suffix_ids = tokenizer(FORCED_ANSWER_SUFFIX, add_special_tokens=False)["input_ids"]

    prompts = [
        {"prompt_token_ids": build_forced_prompt_ids(r["prefix_token_ids"], suffix_ids)}
        for r in rows
    ]

    llm = LLM(
        model=CONFIG.model_id,
        max_model_len=CONFIG.max_model_len,
        gpu_memory_utilization=CONFIG.gpu_memory_utilization,
        dtype="bfloat16",
    )
    params = SamplingParams(
        temperature=FORCED_ANSWER_TEMPERATURE,   # greedy: the modal immediate answer
        top_p=FORCED_ANSWER_TOP_P,
        max_tokens=FORCED_ANSWER_MAX_TOKENS,
        logprobs=CONFIDENCE_TOP_LOGPROBS,        # <- the whole point of this stage
        seed=CONFIG.seed,
    )
    outputs = llm.generate(prompts, params)

    def decode(ids: Sequence[int]) -> str:
        return tokenizer.decode(list(ids), skip_special_tokens=True)

    records = []
    n_missing = 0
    n_not_a_letter = 0
    for r, out in zip(rows, outputs):
        comp = out.outputs[0]
        dist = extract_answer_distribution(
            list(comp.token_ids), getattr(comp, "logprobs", None), decode
        )
        if dist is None:
            n_missing += 1
        elif dist["chosen_letter"] is None:
            n_not_a_letter += 1
        rec = {
            "row_key": row_key(r, default_k),
            "problem_id": r["problem_id"],
            "k_percent": row_k(r, default_k),
            "label": bool(r["label"]),        # the TARGET, never an input to the score
            "generated_text": comp.text,
            "confidence": dist,
            "variants": confidence_variants(dist),
            # decoded here where the tokenizer is loaded, so laptop-side readers
            # never need one (mirrors forced_answer.jsonl's prefix_text)
            "prefix_text": decode(r["prefix_token_ids"]),
            "config_hash": CONFIG.config_hash(),
        }
        records.append(rec)

    with open(FORCED_CONFIDENCE_PATH, "w") as f:
        meta = {
            "record_type": "meta",
            "stage": "forced_confidence",
            "forcing_suffix": FORCED_ANSWER_SUFFIX,
            "max_tokens": FORCED_ANSWER_MAX_TOKENS,
            "temperature": FORCED_ANSWER_TEMPERATURE,
            "top_logprobs": CONFIDENCE_TOP_LOGPROBS,
            "answer_letters": list(ANSWER_LETTERS),
            "variants_recorded": list(CONFIDENCE_VARIANTS),
            "default_variant": DEFAULT_VARIANT,
            "gold_free": True,
            "n_rows": len(records),
            "n_missing_logprobs": n_missing,
            "n_answer_token_not_a_letter": n_not_a_letter,
            "elapsed_s": round(time.time() - t0, 1),
            **lineage(CONFIG.prefixes_path),
        }
        f.write(json.dumps(meta) + "\n")
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(
        f"forced_confidence: {len(records)} forced answers with top-"
        f"{CONFIDENCE_TOP_LOGPROBS} logprobs ({n_missing} missing, "
        f"{n_not_a_letter} whose answer token was not a letter) "
        f"-> {FORCED_CONFIDENCE_PATH} ({time.time() - t0:.1f}s)"
    )
    print("forced_confidence: now run `python -m experiment.forced_confidence score` "
          "AFTER train_probe has written results.json.")


# ---------------------------------------------------------------------------
# Stage B: score (CPU, after train_probe)
#
# ADMISSIBILITY: read this function and check it for yourself. It opens exactly
# two files — forced_confidence.jsonl (prefix-derived scores) and results.json
# (the probe's split) — and uses `label` only as the AUC target. traces.jsonl,
# which is where gold_answer lives, is never opened.
# ---------------------------------------------------------------------------

def score_rows(rows: Sequence[dict], variant: str) -> Tuple[List[dict], List[float], int]:
    """(kept rows, their scores, n dropped for a missing score) — pure, gold-free."""
    kept: List[dict] = []
    scores: List[float] = []
    for r in rows:
        v = (r.get("variants") or confidence_variants(r.get("confidence"))).get(variant)
        if v is None:
            continue
        kept.append(r)
        scores.append(float(v))
    return kept, scores, len(rows) - len(kept)


def score() -> None:
    """AUC of the model's confidence in its own forced answer, on the probe's split."""
    import numpy as np

    try:
        from experiment import analysis
        from experiment.text_floor import split_indices_from_results
    except ImportError:  # pragma: no cover - script-mode fallback
        import analysis  # type: ignore
        from text_floor import split_indices_from_results  # type: ignore

    variant = selected_variant()

    if not os.path.exists(FORCED_CONFIDENCE_PATH):
        raise SystemExit(
            f"HALT: {FORCED_CONFIDENCE_PATH} not found — run "
            f"`python -m experiment.forced_confidence` on the GPU box first."
        )
    try:
        with open(CONFIG.results_path) as f:
            results = json.load(f)
    except OSError:
        raise SystemExit(
            "HALT: results.json not found — run train_probe first; every text "
            "baseline must reuse the probe's exact train/test split."
        )

    _meta, rows = read_jsonl(FORCED_CONFIDENCE_PATH)
    per_k: Dict[str, dict] = {}
    for k, krows in sorted(group_rows_by_k(rows, CONFIG.truncation_k_percent).items()):
        kept, kept_scores, n_dropped = score_rows(krows, variant)
        if not kept:
            continue
        pids = [r["problem_id"] for r in kept]
        _train_idx, test_idx = split_indices_from_results(pids, results)
        if len(test_idx) == 0:
            continue
        y = np.array([bool(kept[i]["label"]) for i in test_idx])
        s = np.array([kept_scores[i] for i in test_idx], dtype=float)
        point = analysis.score_at_k(
            y, s, row_keys=[kept[i]["row_key"] for i in test_idx]
        )

        # Every other variant's AUC on the SAME test rows, so entropy/margin
        # readings need no second GPU run. Reported, never silently selected:
        # the headline is `variant` above.
        other: Dict[str, Optional[float]] = {}
        for v in CONFIDENCE_VARIANTS:
            v_kept, v_scores, _ = score_rows(krows, v)
            v_pids = [r["problem_id"] for r in v_kept]
            if not v_pids:
                other[v] = None
                continue
            _t, v_test = split_indices_from_results(v_pids, results)
            vy = np.array([bool(v_kept[i]["label"]) for i in v_test])
            vs = np.array([v_scores[i] for i in v_test], dtype=float)
            other[v] = (
                round(analysis.roc_auc(vy, vs), 4)
                if len(vy) and len(set(vy.tolist())) == 2
                else None
            )

        point["notes"] = {
            "predictor": (
                "probability the subject model assigned to the answer token it "
                "emitted when interrupted mid-thought and forced to commit"
            ),
            "variant": variant,
            "variant_aucs": other,
            "n_dropped_no_score": n_dropped,
            "n_answer_token_not_a_letter": sum(
                1 for r in krows
                if (r.get("confidence") or {}).get("chosen_letter") is None
            ),
            "forcing_suffix": FORCED_ANSWER_SUFFIX,
            "gold_free": True,
        }
        per_k[str(k)] = point

    if not per_k:
        raise SystemExit("HALT: no test rows survived — nothing to score.")

    path = analysis.write_baseline_json(
        BASELINE_NAME,
        per_k,
        notes={
            "description": (
                "subject model interrupted mid-thought and forced to answer; scored by "
                "its CONFIDENCE in that answer, never by whether the answer was right"
            ),
            "on_policy": True,
            "gold_free": True,
            "admissible_under": "EXPERIMENT.md 12b — score is a function of the prefix alone",
            "variant": variant,
            "variants_available": list(CONFIDENCE_VARIANTS),
            **lineage(FORCED_CONFIDENCE_PATH),
        },
    )
    for k, p in sorted(per_k.items(), key=lambda kv: int(kv[0])):
        print(f"forced_confidence[cut={k}]: AUC={p['auc']} CI95={p['auc_ci95']} "
              f"(n_test={p['n_test']}, variant={variant}) -> {path}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "generate"
    if cmd == "generate":
        main()
    elif cmd == "score":
        score()
    else:
        raise SystemExit("usage: python -m experiment.forced_confidence [generate|score]")
