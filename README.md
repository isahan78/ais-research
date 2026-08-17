# Gate 1 smoke pipeline

Go/no-go for the MATS probe-audit project: on 20 MATH-500 problems, does a
logistic probe on mid-trace Qwen3-8B activations beat a shuffled-label floor
at k=50% thinking-token truncation?

Four stages, plain scripts, JSONL between them, one config
(`config.py`). Every output is stamped with the config hash and the input
file that produced it.

```
generate_traces.py  (vLLM)  -> outputs/traces.jsonl
truncate.py                 -> outputs/prefixes.jsonl
harvest_activations.py (HF) -> outputs/acts.npz (+ .lineage.jsonl)
train_probe.py              -> outputs/results.json   <- the go/no-go number
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

It runs the four stages as **separate processes** and prints elapsed time per
stage plus the final AUC vs floor. Do not run stage 1 and stage 3 concurrently
or from one process: vLLM and HF each need the 15.26 GiB bf16 weights and
cannot be co-resident on 24 GB.

To run stages individually (same order, from the project root):

```bash
python -m experiment.generate_traces
python -m experiment.truncate
python -m experiment.harvest_activations
python -m experiment.train_probe
```

## Cost estimate

| Item | Estimate |
|---|---|
| Instance | ~$0.35–0.60/hr (4090 on Vast/Runpod, 2026 prices) |
| Setup + weight download | 15–25 min |
| Stage 1 (20 traces, ≤3072 new tokens) | 5–15 min |
| Stages 2–4 | < 5 min combined |
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

- **GO:** best-layer AUC > that layer's shuffled-label floor p95.
- **NO-GO:** AUC inside the floor distribution → either infrastructure issue
  or the effect doesn't reproduce at this scale; both paths are in the project
  spec (`_bmad/memory/agent-tyler/mats-project-spec.md`).

## What is deliberately NOT here (deferred; do not add)

Full truncation grid, second dataset (GPQA/MMLU-Pro), the three text-only
baselines, Δ(k) analysis. See `_bmad-output/implementation-artifacts/deferred-work.md`.
Also: no quantization (perturbs the residual stream being measured), no
generation under forward hooks, never `hidden_states[-1]`.
