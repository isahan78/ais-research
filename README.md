# Gate 1 smoke pipeline

Go/no-go for the MATS probe-audit project: on MMLU-Pro questions (dataset
selectable in `config.py` — `dataset_kind` is `mmlu_pro` or `math500`), does a
logistic probe on mid-trace Qwen3-8B activations beat a shuffled-label floor
at k=50% thinking-token truncation?

MMLU-Pro was adopted on a measured base rate (RESULTS.md Run 004: 23% error, 0
ungradeable, 12,032 items); MATH-500 stays selectable and its level filter
applies to that path only. The row->record mapping is the only
dataset-specific code and lives in `dataset_adapters.py` (pure, no torch).

Four stages, plain scripts, JSONL between them, one config
(`config.py`). Every output is stamped with the config hash and the input
file that produced it.

```
generate_traces.py  (vLLM)  -> outputs/traces.jsonl
truncate.py                 -> outputs/prefixes.jsonl
harvest_activations.py (HF) -> outputs/acts.npz (+ .lineage.jsonl)
train_probe.py              -> outputs/results.json   <- the go/no-go number
text_floor.py               -> adds the crude text baseline into results.json
```

**Target: Linux + CUDA, one 24 GB card (4090-class).** This does not run on
a Mac and has no Mac fallbacks — only `tests/` runs anywhere.

## Fresh Runpod / Vast.ai setup

1. Rent a 24 GB GPU box (RTX 4090 / A5000 class), template: CUDA 12.x +
   PyTorch, ≥60 GB disk (weights ~16 GB + HF cache).
2. Clone / copy this repo to the box, then:

   ```bash
   cd <project-root>
   python -m venv .venv && source .venv/bin/activate
   pip install -r experiment/requirements.txt
   # optional but recommended: pre-download weights while you read
   huggingface-cli download Qwen/Qwen3-8B
   ```

   Note: vLLM pins its own torch; if pip reports a torch conflict, delete the
   `torch==` line from requirements.txt and let vLLM's pin win.
3. Sanity check the environment:

   ```bash
   python -c "import torch; print(torch.cuda.get_device_name(0))"
   python -c "import experiment.config as c; print(c.CONFIG)"
   pytest experiment/tests/ -v        # must pass BEFORE burning GPU time
   ```

## Run order

One command:

```bash
bash experiment/smoke_test.sh
```

It runs the invariant suite first (stage 0 — no GPU time is spent if it
fails), then the stages as **separate processes**, tees everything to
`outputs/smoke_test.log`, and ends with one line comparing probe AUC vs
shuffled-label floor vs text floor. Do not run stage 1 and stage 3
concurrently or from one process: vLLM and HF each need the 15.26 GiB bf16
weights and cannot be co-resident on 24 GB.

To run stages individually (same order, from the project root):

```bash
pytest experiment/tests/ -q
python -m experiment.generate_traces
python -m experiment.truncate
python -m experiment.harvest_activations
python -m experiment.train_probe
python -m experiment.text_floor
```

## Cost estimate

| Item | Estimate |
|---|---|
| Instance | ~$0.35–0.60/hr (4090 on Vast/Runpod, 2026 prices) |
| Setup + weight download | 15–25 min |
| Stage 1 (20 traces, ≤8192 new tokens) | 10–30 min |
| Stages 2–5 | < 10 min combined (500 floor seeds × 3 layers dominates) |
| **Total Gate 1** | **well under 1 GPU-hour ≈ $1; budget 90 min wall-clock** |

If the smoke test does not finish in 90 min, that is itself a NO-GO signal —
see the project spec's fallback (drop to R1-Distill-1.5B for prototyping).

## Hand-verification checklist (agent-sanity rule)

Every reported number must be traceable and spot-checked. After each stage,
verify 10 random examples by hand:

**Stage 1 — traces.jsonl**
- [ ] Pick 10 random records. `rendered_prompt` contains the problem text and
      ends with the assistant generation header — no empty `<think></think>`
      pre-fill (thinking mode must be ON).
- [ ] `trace_text` starts a thinking block and, unless
      `truncated_incomplete`, contains `</think>` followed by a final answer.
- [ ] Re-grade 10 answers by eye against `gold_answer`; agree with `correct`.
      If you disagree with >1/10, stop and fix `grading.py` first.
- [ ] Meta line (first line) records `config_hash` and the dataset as
      `input_file`; count of `truncated_incomplete` matches the warnings.

**Stage 2 — prefixes.jsonl**
- [ ] Decode 10 random `prefix_token_ids`
      (`tokenizer.decode(ids)`) — each must end **mid-thinking**: an open
      `<think>` present, **no `</think>` anywhere**, text stops mid-reasoning.
- [ ] `n_kept_thinking_tokens ≈ 0.5 × n_thinking_tokens` for each.
- [ ] Every excluded record has an `exclusion_reason` from
      {truncated_incomplete, thinking_too_short, ungradeable}; counts in the
      meta line match.

**Stage 3 — acts.npz**
- [ ] `problem_ids`, `labels`, and each `acts_layer{9,18,27}` array have the
      same length = number of included prefixes.
- [ ] Vectors are shape 4096, finite, not all-zero; the same
      row index across layers belongs to the same problem_id.
- [ ] `config_hash` inside the npz matches the one in prefixes.jsonl.

**Stage 4 — results.json**
- [ ] `n_train + n_test` = included prefixes; `class_balance` matches stage 1
      correct/incorrect counts (minus exclusions).
- [ ] `lineage.config_hash` matches the hash in traces.jsonl (manual check
      from the spec's verification section).
- [ ] The printed GO/NO-GO line agrees with the per-layer numbers.

## Interpreting the result

The floor is the distribution of the **max-across-layers** AUC under shuffled
train labels (500 seeds) — we report the best of 3 layers, so the floor must
apply the same selection or it flatters the probe.

- **GO:** best-layer AUC > the shuffled-label floor's p95.
- **MARGINAL:** AUC above the floor mean but not its p95. With n_test this
  small that is genuinely ambiguous — the printed line reports what fraction
  of floor seeds the probe beats; a human decides (usually: regenerate with
  more problems before calling it either way).
- **NO-GO:** AUC at or below the floor mean → either infrastructure issue
  or the effect doesn't reproduce at this scale; both paths are in the project
  spec (`_bmad/memory/agent-tyler/mats-project-spec.md`).

**Text-floor comparison (the real question).** `results.json.text_floor` is a
logistic regression on just (prefix token count, a per-problem difficulty
scalar: MATH-500's level, or prompt length where the dataset has none) — no GPU, no
internals, same train/test problems as the probe. Beating shuffled noise only
proves the pipeline works; Gate 1 is interesting to the extent the probe also
clears this crude text-side predictor. If the probe cannot beat two scalars a
spectator reads off the page, the "internals know early" story has no legs at
this scale — which is itself a reportable outcome (the project's question is
exactly how much of the probe's signal the text already gives away). The
three serious text baselines (LLM judge, trained text classifier,
forced-answer) remain deferred work.

## What is deliberately NOT here (deferred; do not add)

Full truncation grid, second dataset (GPQA/MMLU-Pro), the three text-only
baselines, Δ(k) analysis. See `_bmad-output/implementation-artifacts/deferred-work.md`.
Also: no quantization (perturbs the residual stream being measured), no
generation under forward hooks, never `hidden_states[-1]`.
