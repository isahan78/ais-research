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

---

## Run 005 — Block 2: the actual experiment (IN PROGRESS)
**2026-08-30 05:40 UTC · RTX 4090 (Runpod secure, EU-RO-1) · commit `0677f09` · gen config hash `3f151bfa10e2`**
Qwen3-8B · **MMLU-Pro** · n=300 · thinking budget 16,384 · truncation grid k ∈ {1, 10, 25, 50, 75, 90}

### Generation (complete)
| Metric | Value | Pre-registered |
|---|---|---|
| Traces | 300 (420 s… actually 76 min) | — |
| Correct / **incorrect** | 237 / **52** | — |
| **Error rate on gradeable** | **18.0%** (52/289) | Owner 40% · Tyler 23% → **Tyler closer, both over** |
| Truncated mid-thinking | 8 (2.7%) | Decision B fully vindicated (was 30% at 8k) |
| Ungradeable | 11 (3.7%) | — |

**52 negatives** — a real minority class. MATH-500's hard ceiling was ~15. The dataset change was correct.

### k=1 — the near-zero control (prompt alone, one thinking token)
Added at Tyler's request while the pod was warm; **the control that decides whether the whole curve is meaningful.**

| Layer | AUC |
|---|---|
| 9 | 0.608 |
| 18 | 0.589 |
| **27** | **0.621** (CI95 0.503–0.741) |

Floor p95 = 0.669 → **MARGINAL** (below the floor's 95th percentile).

**The confound is ruled out.** Reading the question alone does *not* reliably predict correctness (0.621, not clearing noise), while 10% of thinking jumps to 0.761. So the probe reads something about the reasoning, not merely how hard the question looks — the failure mode [No Answer Needed](https://arxiv.org/abs/2509.10625) demonstrates is real for others but not operating here. Measured, not asserted.

### k=10 — first Δ
| Reader | AUC |
|---|---|
| **Tuned TF-IDF text classifier** | **0.7735** ← best text reader |
| Probe (layer 27) | 0.7605 |
| Forced-answer (on-policy) | 0.7057 |
| Crude text floor (prefix len + prompt len) | 0.6177 |

**Δ = −0.013, CI95 [−0.151, +0.117]** (paired bootstrap). n_train=187 / n_test=102.

**The probe does not beat the text.** CI comfortably contains zero ⇒ *no detectable difference*, not "text wins".

**Why the strong baseline mattered, concretely:** against the crude floor alone, Δ would have read **+0.14** — a clean, publishable-looking win. The tuned classifier (48-config grid, StratifiedGroupKFold inside the training split only) is what removed it. This is the project's own thesis demonstrated on its own result before publication.

Δ here is an **upper bound**: the LLM judge has not run, and adding any reader can only raise S_text and lower Δ. Recorded in `analysis.json`.

### Incidents
- **CUDA OOM at k=1 forced-answer.** Killing the grid by PID left a vLLM child holding 11 GiB; the new loop's engine collided with it. k=1 forced-answer must be re-run. Lesson: verify the GPU is clear before relaunching, not just that the parent died.
- **Pod CPU fitting was ~55× slower than the laptop** (55 min vs ~1 min per probe). Switched the pod to GPU-only work (truncate → forced-answer → harvest); all probe/floor/classifier/analysis now runs locally. Saved ~4 h of pod time (~$3).

### Status
k=1 ✅ · k=10 ✅ · k=25/50/75/90 in progress. Pod ~$2.70 so far.
