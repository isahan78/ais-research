# Run Log

**Append-only.** Every run that produced a number, newest last. One line of provenance per entry so any claim in the write-up can be traced to the run that produced it. Design rationale lives in [EXPERIMENT.md](EXPERIMENT.md); this file is what happened.

---

## Run 001 — Gate 1 smoke test
**2026-08-26 05:21 UTC · RTX 4090 (Runpod, secure) · commit `8f74ff6` · config hash `395afb8a19e5`**
Qwen3-8B · MATH-500 levels 4–5 · n=20 · thinking budget 8,192 · k=50%

| Metric | Value |
|---|---|
| Traces generated | 20 (420 s) |
| Hit the token cap mid-thinking | **6 (30%)** |
| Of the 14 that finished | **13 correct, 1 incorrect** |
| Activations harvested | 14 × layers 9/18/27 (16 s) |
| Probe | **HALTed** |

**Outcome: infrastructure GO, science NO-GO.**
`train_probe` refused to fit: *"after 50 split seeds, could not build a train/test split with both classes on both sides (13 True / 1 False of 14)."* This guard was added because of the 2026-08-23 adversarial review; without it the run would have produced a number.

Two failures found before this run succeeded, neither reproducible on macOS:
1. `transformers==4.57.1` + `torch==2.9.0` vs `vllm==0.27.1` → `ResolutionImpossible`. Fixed in `8f74ff6`.
2. Missing `ninja` → vLLM FlashInfer JIT crash at warmup. Fixed with `pip install ninja` + `VLLM_USE_FLASHINFER_SAMPLER=0`.
Also: Runpod **Community** 4090s advertise stock but refuse to deploy; Secure Cloud works.

Artifacts: `outputs/` (local — traces, prefixes, acts.npz, smoke_test.log).

---

## Run 002 — Calibration: does a bigger budget + harder levels fix the balance?
**2026-08-26 05:42 UTC · same pod · Qwen3-8B · MATH-500 level 5 only · n=40 · budget 16,384**

| Metric | Value | vs Run 001 |
|---|---|---|
| Truncated mid-thinking | **3 (8%)** | 30% → 8% ✅ |
| Correct / incorrect | **32 / 4** | error 11% on gradeable |
| Ungradeable | 4 (only **1** genuine — 3 were truncated, hand-checked) | grader vindicated |

**Decision B confirmed** (budget fix removes label-correlated survivor bias). **Decision A refuted** — level-5-only did not produce a usable negative class.
Artifact: `outputs/calibration/traces_8B_math500_L5.jsonl`

---

## Run 003 — Calibration: is a weaker recent model the answer?
**2026-08-26 06:15 UTC · same pod · Qwen3-4B · MATH-500 level 5 · n=40 · budget 16,384**

| Model | Error on gradeable | Projected negatives from all 134 L5 items |
|---|---|---|
| Qwen3-8B | 4/36 = **11%** | ~15 |
| Qwen3-4B | 5/36 = **14%** | ~19 |

**Halving the model moved the error rate 3 points.** Conclusion: the *dataset*, not the model, is the binding constraint — MATH-500 level 5 does not discriminate between these models. MATH-500 is exhausted for a correctness target (134 level-5 items ⇒ ~15 negatives, far too few for a 4096-dim probe).
Artifact: `outputs/calibration/traces_4B_math500_L5.jsonl`

---

## Run 004 — Calibration: MMLU-Pro base rate (measured before adoption)
**2026-08-26 06:35 UTC · same pod · Qwen3-8B · MMLU-Pro · n=40 · budget 16,384**

| Metric | Value |
|---|---|
| Correct / incorrect | **30 / 9** |
| **Error rate on gradeable** | **9/39 = 23%** |
| Ungradeable | **0** |
| Truncated | 1 (2.5%) |

**Adopted.** With 12,032 available items the negative-class ceiling disappears; ~300 questions yields ~70 negatives. Binary correctness target, AUC, probe, floors and all 190 tests are unchanged — a dataset swap, not a redesign.
Artifact: `outputs/calibration/mmlu_pro_8B_probe.log`

---

## Session summary — 2026-08-26

**Spend:** ~$1.15 (one RTX 4090, 95 min, terminated and verified: zero pods running).

**Produced:** a pipeline that runs end-to-end on real hardware (190 tests green *on the box*, from a real clone, before any GPU spend); three hardware-only bugs found and fixed; and a dataset decision made by measurement with the evidence committed.

**Not produced:** a scientific result. Δ(k) has not been measured yet.

**Process lesson, recorded against Tyler:** model and dataset were each chosen on sound but independent criteria, and the base rate of the predicted variable was never measured. The adversarial review caught the symptom (class imbalance); a plausible fix (harder levels) was applied and *not verified*. New rule: **no dataset is adopted without a measured base rate.** Runs 002–004 are that rule being applied.

**Next:** MMLU-Pro loader in `generate_traces.py` (dataset swap + build prompts from the 10 options; `\boxed{}` grading already works — 0 ungradeable). Then fill the pre-registration in EXPERIMENT.md §12 **before** Block 2 runs.
