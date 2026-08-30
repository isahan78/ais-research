"""Stage 1: generate thinking traces with vLLM and grade them.

Dataset-agnostic: CONFIG.dataset_kind picks a pure adapter in
dataset_adapters.py that maps a raw HF row to (problem_id, prompt_text,
gold_answer, meta). Nothing below that mapping knows which benchmark it is.

Writes traces.jsonl. Runs as its own process and exits fully before stage 3
starts — vLLM and HF cannot be co-resident on a 24 GB card (2 x 15.26 GiB).

Correctness lynchpin: the prompt is rendered ONCE here via
tokenizer.apply_chat_template(..., tokenize=False) and the rendered string is
handed to vLLM as `prompt_token_ids` (tokenized once, stored verbatim), so
stage 3 replays the exact same context. vLLM never re-renders.

Heavy imports (vllm, transformers, datasets) live inside main() so this
module is importable on a no-GPU machine.
"""

from __future__ import annotations

import json
import os
import sys
import time

try:
    from experiment.config import CONFIG, THINK_END_ID, lineage
    from experiment.dataset_adapters import get_adapter, parse_level
    from experiment.grading import grade
except ImportError:
    from config import CONFIG, THINK_END_ID, lineage
    from dataset_adapters import get_adapter, parse_level
    from grading import grade

__all__ = [
    "check_incomplete_fraction",
    "make_record",
    "check_ungradeable_fraction",
    "get_adapter",
    "main",
    "parse_level",
    "render_prompt",
]


def check_ungradeable_fraction(n_ungradeable: int, n_gradeable_pool: int) -> None:
    """HALT (I/O matrix row 4) when too many completed traces are ungradeable.

    Called only AFTER traces.jsonl is on disk — the expensive artifact must
    never be destroyed by its own quality gate.
    """
    if n_gradeable_pool > 0 and n_ungradeable / n_gradeable_pool > CONFIG.max_ungradeable_fraction:
        raise SystemExit(
            f"HALT: {n_ungradeable}/{n_gradeable_pool} completed traces are ungradeable "
            f"(> {CONFIG.max_ungradeable_fraction:.0%}). The answer parser is likely broken "
            f"for this data — inspect trace_text in {CONFIG.traces_path} before proceeding. "
            f"traces.jsonl has already been written; no GPU time is lost."
        )


def check_incomplete_fraction(n_incomplete: int, n_total: int) -> None:
    """HALT (I/O matrix row 6) when too many traces hit max_tokens mid-thinking.

    Incomplete traces are not random: they are the long, struggling — i.e.
    disproportionately incorrect — ones. Excluding many of them is
    label-correlated survivor bias and would fabricate the probe's job.
    Called only AFTER traces.jsonl is on disk.
    """
    if n_total > 0 and n_incomplete / n_total > CONFIG.max_incomplete_fraction:
        raise SystemExit(
            f"HALT: {n_incomplete}/{n_total} traces hit max_tokens mid-thinking "
            f"(> {CONFIG.max_incomplete_fraction:.0%}). Label-correlated truncation biases "
            f"the sample — raise max_new_tokens (currently {CONFIG.max_new_tokens}) and "
            f"max_model_len, then regenerate. traces.jsonl has already been written."
        )


def render_prompt(tokenizer, prompt_text: str) -> str:
    """Render the chat prompt exactly once. Note: `enable_thinking` is
    deliberately NOT passed — thinking mode is on by default for Qwen3, and
    passing enable_thinking=False would pre-fill an empty think block."""
    messages = [{"role": "user", "content": prompt_text}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def make_record(item, rendered_prompt: str, prompt_token_ids, trace_text: str,
                trace_token_ids, correct, truncated_incomplete: bool) -> dict:
    """One traces.jsonl row. Pure, so the schema is testable without a GPU.

    Schema note: `level` and `subject` stay TOP-LEVEL keys even where the
    dataset has no such field (level is None for MMLU-Pro, subject is its
    category) — downstream stages (truncate, harvest, train_probe, text_floor)
    and previously written traces.jsonl files must keep reading the same keys.
    Dataset-specific extras are ADDED under `meta`, never swapped in.
    """
    return {
        "problem_id": item.problem_id,
        "level": item.meta.get("level"),
        "subject": item.meta.get("subject"),
        "gold_answer": item.gold_answer,
        "meta": dict(item.meta),
        "rendered_prompt": rendered_prompt,
        "prompt_token_ids": prompt_token_ids,
        "trace_text": trace_text,
        "trace_token_ids": trace_token_ids,
        "correct": correct,
        "truncated_incomplete": truncated_incomplete,
        "config_hash": CONFIG.config_hash(),
    }


def main() -> None:
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    os.makedirs(CONFIG.output_dir, exist_ok=True)
    t0 = time.time()

    adapter = get_adapter(CONFIG.dataset_kind)
    ds = load_dataset(CONFIG.dataset_id, split=CONFIG.dataset_split)
    # The level filter is a MATH-500 concept (decision A). The adapter owns it,
    # so MMLU-Pro — which has no `level` field — is simply unfiltered.
    ds = ds.filter(lambda row: adapter.keep_row(row, CONFIG.levels))
    if len(ds) < CONFIG.n_problems:
        raise SystemExit(
            f"HALT: only {len(ds)} problems survive the {CONFIG.dataset_kind} filter in "
            f"{CONFIG.dataset_id}:{CONFIG.dataset_split}; need {CONFIG.n_problems}."
        )
    ds = ds.shuffle(seed=CONFIG.seed).select(range(CONFIG.n_problems))

    # One pure mapping from raw rows to the schema the rest of the pipeline
    # knows. Everything below this line is dataset-agnostic.
    items = [adapter.adapt(row) for row in ds]

    tokenizer = AutoTokenizer.from_pretrained(CONFIG.model_id)

    rendered = [render_prompt(tokenizer, item.prompt_text) for item in items]
    prompt_ids = [tokenizer(p, add_special_tokens=False)["input_ids"] for p in rendered]

    llm = LLM(
        model=CONFIG.model_id,
        max_model_len=CONFIG.max_model_len,
        gpu_memory_utilization=CONFIG.gpu_memory_utilization,
        dtype="bfloat16",
    )
    params = SamplingParams(
        temperature=CONFIG.temperature,
        top_p=CONFIG.top_p,
        max_tokens=CONFIG.max_new_tokens,
        seed=CONFIG.seed,
    )

    # Feed token ids (not text) so vLLM cannot re-tokenize differently.
    outputs = llm.generate(
        [{"prompt_token_ids": ids} for ids in prompt_ids], params
    )

    n_ungradeable = 0
    n_incomplete = 0
    records = []
    for item, prompt, ids, out in zip(items, rendered, prompt_ids, outputs):
        trace_ids = list(out.outputs[0].token_ids)
        # <think>/<\think> are special:false, so they survive this decode.
        trace_text = tokenizer.decode(trace_ids, skip_special_tokens=True)

        truncated_incomplete = THINK_END_ID not in trace_ids
        if truncated_incomplete:
            n_incomplete += 1
            correct = None  # no post-thinking answer exists to grade
            print(f"WARNING: {item.problem_id}: no </think> (hit max_tokens mid-thinking)",
                  file=sys.stderr)
        else:
            end = trace_ids.index(THINK_END_ID)
            response_text = tokenizer.decode(trace_ids[end + 1:], skip_special_tokens=True)
            correct = grade(response_text, item.gold_answer)
            if correct is None:
                n_ungradeable += 1
                print(f"WARNING: {item.problem_id}: ungradeable answer", file=sys.stderr)

        records.append(make_record(
            item, prompt, ids, trace_text, trace_ids, correct, truncated_incomplete
        ))

    # Write FIRST, gate after: the traces are the expensive artifact and must
    # survive their own quality gates (rebuild task, review loop 1).
    with open(CONFIG.traces_path, "w") as f:
        meta = {
            "record_type": "meta",
            "stage": "generate_traces",
            "dataset_kind": CONFIG.dataset_kind,
            "n_problems": len(records),
            "n_truncated_incomplete": n_incomplete,
            "n_ungradeable": n_ungradeable,
            "elapsed_s": round(time.time() - t0, 1),
            **lineage(f"{CONFIG.dataset_id}:{CONFIG.dataset_split}"),
        }
        f.write(json.dumps(meta) + "\n")
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    n_correct = sum(1 for r in records if r["correct"] is True)
    print(f"generate_traces: {len(records)} traces "
          f"({n_correct} correct, {n_incomplete} incomplete, {n_ungradeable} ungradeable) "
          f"-> {CONFIG.traces_path}")

    # Quality gates run AFTER the artifact is safely on disk.
    check_incomplete_fraction(n_incomplete, len(records))
    check_ungradeable_fraction(n_ungradeable, len(records) - n_incomplete)


if __name__ == "__main__":
    main()
