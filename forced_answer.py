"""Text baseline 1 of 3: FORCED ANSWER — the on-policy reader.

The idea (EXPERIMENT.md §5.3, "our most novel component"): take the exact
prefix the probe reads — prompt + `<think>` + the first k% of thinking tokens —
close the thinking block with `</think>`, append a short forcing string, and
make the subject model commit to a final answer IMMEDIATELY. Grade that answer
with `experiment.grading.grade`. Its correctness is a prediction of whether the
full trace will end correct.

Why this is the strongest kind of text baseline: it reads exactly the same
information horizon as the probe, it is on-policy (same model, same tokens, no
distribution shift), and it costs one short generation per row. If it matches
the probe, the headline sharpens from "internals know" to *"you can just ask."*

RUN ORDER — this stage must run in the SAME pod session as generation:

    python -m experiment.generate_traces      # stage 1 (vLLM)
    python -m experiment.truncate             # stage 2 (CPU, seconds)
    python -m experiment.forced_answer        # THIS, still on the GPU box
    ...                                       # harvest / train_probe later
    python -m experiment.forced_answer score  # CPU, after train_probe

Two subcommands because the two halves need different things:
  * `generate` (default) needs the GPU and vLLM, and only prefixes/traces.
  * `score` needs results.json (the probe's train/test split), which does not
    exist until stage 4 has run. It is pure CPU and re-runnable.

Heavy imports (vllm, transformers) live inside `main()`, so importing this
module on a laptop with no GPU is free — the invariant tests depend on that.

This module also owns the SHARED PREFIX I/O helpers that `llm_judge.py` and
`text_classifier.py` import (see the section marked below): reading
prefixes.jsonl, grouping rows by truncation k, and turning stored
`prefix_token_ids` back into the decoded text the text-side readers see.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

try:
    from experiment.config import CONFIG, THINK_END_ID, lineage
    from experiment.grading import extract_boxed, grade
except ImportError:  # run as a plain script from inside experiment/
    from config import CONFIG, THINK_END_ID, lineage
    from grading import extract_boxed, grade


# --- tunables (module-level on purpose: config.py is owned by another stage) --

# The forcing string, appended AFTER `</think>`. Design constraints:
#   * no new information — it must not hint at the answer or re-state the
#     question, or the baseline stops being a read of the prefix;
#   * it opens `\boxed{` so the model's very first tokens ARE the answer, which
#     both keeps the generation short and makes it gradeable by the existing
#     grader with no new parser;
#   * it acknowledges the reasoning is unfinished, so the model commits to a
#     best guess instead of trying to resume thinking (which it will do if
#     simply cut off — measured behaviour of Qwen3 thinking mode).
FORCED_ANSWER_SUFFIX = (
    "\n\nI have to stop reasoning here and commit. "
    "Based only on the reasoning above, my single best answer is \\boxed{"
)
FORCED_ANSWER_MAX_TOKENS = 32      # `\boxed{C}` needs ~3; 32 leaves room for LaTeX
FORCED_ANSWER_TEMPERATURE = 0.0    # greedy: we want the model's modal immediate answer,
                                   # not a sample — this predictor must be deterministic
FORCED_ANSWER_TOP_P = 1.0

FORCED_ANSWER_PATH = os.path.join(CONFIG.output_dir, "forced_answer.jsonl")

BASELINE_NAME = "forced_answer"


# ---------------------------------------------------------------------------
# Pure prompt construction / grading — unit-tested with no GPU
# ---------------------------------------------------------------------------

def build_forced_prompt_ids(
    prefix_token_ids: Sequence[int],
    suffix_token_ids: Sequence[int],
    think_end_id: int = THINK_END_ID,
) -> List[int]:
    """prefix + `</think>` + forcing string, as token ids.

    `truncate.build_prefix` guarantees the prefix ends strictly INSIDE the
    thinking span (no `</think>` anywhere in it). We re-check rather than
    trust it: if a `</think>` ever leaked in, the model would be answering
    after its own finished reasoning and this baseline would silently become a
    different, much easier task.
    """
    prefix = list(prefix_token_ids)
    if think_end_id in prefix:
        raise RuntimeError(
            "forced_answer: `</think>` already present in the prefix — the prefix "
            "does not end mid-thinking, so a forced answer would not be an "
            "interruption. Rerun truncate."
        )
    if not prefix:
        raise RuntimeError("forced_answer: empty prefix_token_ids")
    return prefix + [think_end_id] + list(suffix_token_ids)


def close_boxed(text: str) -> str:
    """Append a `}` when the last `\\boxed{` in `text` is left unclosed.

    The forcing string opens the brace, so a generation that runs out of tokens
    mid-answer leaves it dangling and `extract_boxed` (correctly) returns None.
    Closing it recovers the common short case instead of throwing the row away
    as ungradeable — a discarded row is a silently weakened baseline.
    """
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return text
    depth = 0
    for ch in text[idx + len("\\boxed") :]:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text  # already balanced
    return text + "}" * max(depth, 1)


def assemble_forced_response(suffix_text: str, generated_text: str) -> str:
    """The text handed to `grading.grade` — the forcing string plus what the
    model wrote after it. The suffix must be included because it carries the
    opening `\\boxed{`."""
    return close_boxed(suffix_text + generated_text)


def grade_forced_answer(
    suffix_text: str, generated_text: str, gold_answer: str
) -> Optional[bool]:
    """True / False / None(ungradeable) for one forced answer."""
    return grade(assemble_forced_response(suffix_text, generated_text), gold_answer)


def forced_answer_text(suffix_text: str, generated_text: str) -> Optional[str]:
    """What the model actually committed to, for eyeballing 10 rows by hand."""
    return extract_boxed(assemble_forced_response(suffix_text, generated_text))


# ---------------------------------------------------------------------------
# SHARED PREFIX I/O — imported by llm_judge.py and text_classifier.py
# ---------------------------------------------------------------------------

def read_jsonl(path: str) -> Tuple[Optional[dict], List[dict]]:
    """Split a pipeline .jsonl into its `record_type: meta` line and its rows."""
    meta: Optional[dict] = None
    rows: List[dict] = []
    with open(path) as f:
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


def load_included_prefix_rows(
    prefixes_path: str = CONFIG.prefixes_path,
) -> Tuple[List[dict], int]:
    """Included prefix rows plus the default truncation k for the file.

    Every text baseline must read EXACTLY the rows the probe reads, so the
    `included` filter here is the same one text_floor/harvest apply.
    """
    meta, rows = read_jsonl(prefixes_path)
    default_k = int((meta or {}).get("k_percent", CONFIG.truncation_k_percent))
    return [r for r in rows if r.get("included")], default_k


def row_k(row: dict, default_k: int) -> int:
    """Truncation k for one prefix row.

    Today truncate.py writes one row per problem at a single k, recorded in the
    file's meta line. If the k-grid lands (EXPERIMENT.md §4 layer 3) rows will
    carry their own `k_percent`; honour it when present so the baselines and
    the Δ(k) curve need no change.
    """
    k = row.get("k_percent")
    return int(default_k if k is None else k)


def row_key(row: dict, default_k: int) -> str:
    """Stable identity for one (problem, cut) row, unique across the k grid."""
    return f"{row['problem_id']}@k{row_k(row, default_k)}"


def group_rows_by_k(rows: Sequence[dict], default_k: int) -> Dict[int, List[dict]]:
    out: Dict[int, List[dict]] = {}
    for r in rows:
        out.setdefault(row_k(r, default_k), []).append(r)
    return out


def gold_answers_by_problem(traces_path: str = CONFIG.traces_path) -> Dict[str, str]:
    _meta, rows = read_jsonl(traces_path)
    return {r["problem_id"]: r["gold_answer"] for r in rows}


def _hf_decoder(model_id: str = CONFIG.model_id) -> Callable[[Sequence[int]], str]:
    """Lazy tokenizer decode. Imported inside so this module stays GPU-free."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)

    def decode(ids: Sequence[int]) -> str:
        # <think>/</think> are special:false for Qwen3, so they survive this
        # decode — the judge and the TF-IDF classifier see the same markers the
        # model wrote.
        return tok.decode(list(ids), skip_special_tokens=True)

    return decode


def decode_prefix_texts(
    rows: Sequence[dict], decode: Optional[Callable[[Sequence[int]], str]] = None
) -> List[str]:
    """Decode `prefix_token_ids` -> text for each row, in order.

    `decode` is injectable so tests can run with no tokenizer and no network.
    """
    if decode is None:
        decode = _hf_decoder()
    return [decode(r["prefix_token_ids"]) for r in rows]


def load_prefix_texts(
    rows: Sequence[dict],
    default_k: int,
    cache_path: str = FORCED_ANSWER_PATH,
    decode: Optional[Callable[[Sequence[int]], str]] = None,
) -> List[str]:
    """Prefix text per row, reusing what the forced-answer stage already wrote.

    `forced_answer.jsonl` stores the decoded `prefix_text` because that stage
    has the tokenizer loaded anyway. Reusing it means the LLM-judge and the
    TF-IDF classifier — both of which run on a laptop — need no tokenizer, no
    model download, and cannot possibly decode the prefix differently from the
    text the subject model was actually shown.
    """
    cached: Dict[str, str] = {}
    if cache_path and os.path.exists(cache_path):
        _meta, crows = read_jsonl(cache_path)
        for r in crows:
            if r.get("prefix_text") is not None and r.get("row_key"):
                cached[r["row_key"]] = r["prefix_text"]

    keys = [row_key(r, default_k) for r in rows]
    if all(k in cached for k in keys):
        return [cached[k] for k in keys]

    missing = sum(1 for k in keys if k not in cached)
    if cached:
        print(
            f"WARNING: {missing}/{len(keys)} prefixes are not in {cache_path}; "
            f"decoding those with the tokenizer instead.",
            file=sys.stderr,
        )
    texts = decode_prefix_texts(rows, decode)
    return [cached.get(k, t) for k, t in zip(keys, texts)]


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
            "HALT: no included rows in prefixes.jsonl — run truncate first."
        )
    gold = gold_answers_by_problem()
    missing_gold = sorted({r["problem_id"] for r in rows if r["problem_id"] not in gold})
    if missing_gold:
        raise SystemExit(
            f"HALT: no gold answer in traces.jsonl for {missing_gold[:5]}"
            f"{' ...' if len(missing_gold) > 5 else ''} — prefixes.jsonl and "
            f"traces.jsonl are out of sync."
        )

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
        temperature=FORCED_ANSWER_TEMPERATURE,
        top_p=FORCED_ANSWER_TOP_P,
        max_tokens=FORCED_ANSWER_MAX_TOKENS,
        seed=CONFIG.seed,
    )
    outputs = llm.generate(prompts, params)

    records = []
    n_ungradeable = 0
    for r, out in zip(rows, outputs):
        gen = out.outputs[0].text
        forced_correct = grade_forced_answer(FORCED_ANSWER_SUFFIX, gen, gold[r["problem_id"]])
        if forced_correct is None:
            n_ungradeable += 1
        records.append(
            {
                "row_key": row_key(r, default_k),
                "problem_id": r["problem_id"],
                "k_percent": row_k(r, default_k),
                "label": bool(r["label"]),          # did the FULL trace end correct?
                "forced_correct": forced_correct,   # did the INTERRUPTED model get it right?
                "forced_answer": forced_answer_text(FORCED_ANSWER_SUFFIX, gen),
                "generated_text": gen,
                # decoded once here, where the tokenizer is already loaded, so the
                # laptop-side text baselines never need one (see load_prefix_texts)
                "prefix_text": tokenizer.decode(
                    r["prefix_token_ids"], skip_special_tokens=True
                ),
                "config_hash": CONFIG.config_hash(),
            }
        )

    with open(FORCED_ANSWER_PATH, "w") as f:
        meta = {
            "record_type": "meta",
            "stage": "forced_answer",
            "forcing_suffix": FORCED_ANSWER_SUFFIX,
            "max_tokens": FORCED_ANSWER_MAX_TOKENS,
            "temperature": FORCED_ANSWER_TEMPERATURE,
            "n_rows": len(records),
            "n_ungradeable": n_ungradeable,
            "elapsed_s": round(time.time() - t0, 1),
            **lineage(CONFIG.prefixes_path),
        }
        f.write(json.dumps(meta) + "\n")
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    agree = sum(1 for r in records if r["forced_correct"] is r["label"])
    print(
        f"forced_answer: {len(records)} forced answers "
        f"({n_ungradeable} ungradeable, {agree}/{len(records)} agree with the final label) "
        f"-> {FORCED_ANSWER_PATH} ({time.time() - t0:.1f}s)"
    )
    print("forced_answer: now run `python -m experiment.forced_answer score` "
          "AFTER train_probe has written results.json.")


# ---------------------------------------------------------------------------
# Stage B: score (CPU, after train_probe)
# ---------------------------------------------------------------------------

def score() -> None:
    """AUC of forced-answer correctness against the true label, on the probe's
    test split.

    The score is BINARY (the forced answer is right or wrong), so its ROC curve
    has a single interior operating point and its AUC equals balanced accuracy.
    That is the honest reading of this predictor — it commits, it does not
    rank — and it is still directly comparable to the probe's AUC.
    """
    import numpy as np

    try:
        from experiment import analysis
        from experiment.text_floor import split_indices_from_results
    except ImportError:
        import analysis  # type: ignore
        from text_floor import split_indices_from_results  # type: ignore

    if not os.path.exists(FORCED_ANSWER_PATH):
        raise SystemExit(
            f"HALT: {FORCED_ANSWER_PATH} not found — run "
            f"`python -m experiment.forced_answer` on the GPU box first."
        )
    try:
        with open(CONFIG.results_path) as f:
            results = json.load(f)
    except OSError:
        raise SystemExit(
            "HALT: results.json not found — run train_probe first; every text "
            "baseline must reuse the probe's exact train/test split."
        )

    _meta, rows = read_jsonl(FORCED_ANSWER_PATH)
    per_k: Dict[str, dict] = {}
    for k, krows in sorted(group_rows_by_k(rows, CONFIG.truncation_k_percent).items()):
        gradeable = [r for r in krows if r["forced_correct"] is not None]
        n_dropped = len(krows) - len(gradeable)
        pids = [r["problem_id"] for r in gradeable]
        _train_idx, test_idx = split_indices_from_results(pids, results)
        if len(test_idx) == 0:
            continue
        y = np.array([bool(gradeable[i]["label"]) for i in test_idx])
        s = np.array([1.0 if gradeable[i]["forced_correct"] else 0.0 for i in test_idx])
        point = analysis.score_at_k(
            y, s, row_keys=[gradeable[i]["row_key"] for i in test_idx]
        )
        point["notes"] = {
            "predictor": "binary correctness of the interrupted model's immediate answer",
            "n_ungradeable_dropped": n_dropped,
            "forcing_suffix": FORCED_ANSWER_SUFFIX,
            "agreement_rate_test": round(float((s == y.astype(float)).mean()), 4),
        }
        per_k[str(k)] = point

    if not per_k:
        raise SystemExit("HALT: no test rows survived — nothing to score.")

    path = analysis.write_baseline_json(
        BASELINE_NAME,
        per_k,
        notes={
            "description": "subject model interrupted mid-thought and forced to answer",
            "on_policy": True,
            **lineage(FORCED_ANSWER_PATH),
        },
    )
    for k, p in sorted(per_k.items(), key=lambda kv: int(kv[0])):
        print(f"forced_answer[k={k}%]: AUC={p['auc']} CI95={p['auc_ci95']} "
              f"(n_test={p['n_test']}) -> {path}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "generate"
    if cmd == "generate":
        main()
    elif cmd == "score":
        score()
    else:
        raise SystemExit(f"usage: python -m experiment.forced_answer [generate|score]")
