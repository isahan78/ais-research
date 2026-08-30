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

## Run 005 — Block 2 COMPLETE: the full Δ(k) curve
**2026-08-30 05:40–09:09 UTC · RTX 4090 (Runpod secure, EU-RO-1) · commit `0677f09` · terminated and verified**
Qwen3-8B · MMLU-Pro · n=300 · thinking budget 16,384 · k ∈ {1, 10, 25, 50, 75, 90} · n_train=187 / n_test=102 · **total cost ≈ $3.30**

### Generation
300 traces · **237 correct / 52 incorrect** · **18.0% error** on 289 gradeable · truncated 2.7% · ungradeable 3.7%.
Pre-registered error rate: owner 40%, Tyler 23% → **both over; Tyler closer**. 52 negatives is a real minority class (MATH-500's ceiling was ~15).

### The headline table (AUC, higher = better predictor of final correctness)

| k | probe (best layer) | forced-answer | tuned TF-IDF | crude floor | **Δ = S_probe − max(S_text)** | CI excl. 0 |
|---:|---:|---:|---:|---:|---:|:--:|
| **1** *(control)* | 0.621 | — | 0.732 | 0.641 | **−0.111** [−0.235, +0.012] | no |
| 10 | 0.761 | 0.706 | **0.774** | 0.618 | **−0.013** [−0.151, +0.117] | no |
| 25 | 0.647 | 0.745 | **0.781** | 0.613 | **−0.135** [−0.244, −0.036] | **yes** |
| 50 | 0.681 | **0.842** | 0.777 | 0.613 | **−0.162** [−0.325, +0.011] | no (grazes) |
| 75 | 0.677 | **0.848** | 0.776 | 0.613 | **−0.171** [−0.308, −0.042] | **yes** |
| 90 | 0.701 | **0.961** | 0.776 | 0.611 | **−0.260** [−0.375, −0.156] | **yes** |

### Pre-registration: BOTH FORECASTS WRONG, SAME DIRECTION

| | Predicted Δ at k=50 | Actual |
|---|---|---|
| Owner (Isahan) | **+0.10** (Camp Internals) | **−0.162** |
| Tyler | **+0.02** (Camp Text) | **−0.162** |

Both predicted the probe would win. It came **third**. The shared blind spot was the *text* side: owner predicted S_text ≈ 0.60, Tyler ≈ 0.70; it measured **0.842**. Tyler was directionally right and still badly miscalibrated on magnitude.

### Findings

**1. Δ(k) is negative and widens monotonically after k=10.** The probe plateaus at ~0.68 while the best text reader climbs from 0.77 to 0.96. Three of six cuts have CIs excluding zero, all negative. On this evidence, in this setting, **a linear probe on internals does not beat the strongest cheap text reader — and the gap grows as the model reasons.**

**2. The winner is the baseline with no precedent in the audited papers.** Forced-answer — close the think block, make the model commit — rises 0.706 → **0.961**. For predicting where a reasoning model is heading, *asking it* beats *reading its internals*, and costs one short generation instead of a trained probe.

**3. The k=1 control does its job, and complicates the story usefully.** The probe on the prompt alone scores 0.621 and does **not** clear the noise floor (p95 ≈ 0.669) — so probe signal at k≥10 is not mere question difficulty. **But TF-IDF at k=1 scores 0.732**, and stays ~0.78 from k=25 onward: the text classifier is substantially a *question-difficulty detector*, nearly flat in k. Forced-answer is the only reader that tracks the reasoning itself. Any honest write-up must separate these.

**4. The model commits early.** Forced-answer agreement with the final answer: 72% (k=25) → 88% (k=50) → 94% (k=75) → 97% (k=90). The commitment-boundary effect, measured directly.

**5. Strong baselines were load-bearing.** Against the crude floor alone, Δ at k=50 would have read **+0.07** and looked like a clean win. The tuned classifier and forced-answer removed it. The project's own thesis, demonstrated on its own result before publication.

### Caveats
n_test = 102 · single model, single dataset · **Δ is an upper bound** (LLM judge not yet run; adding a reader can only lower Δ) · k=1 forced-answer missing (OOM casualty, pod terminated before retry) · probe non-monotonic across k (0.761 → 0.647 → 0.681) is likely small-n noise and should not be read as a trend.

### Incidents
- **CUDA OOM at k=1 forced-answer** — killing the grid by PID left a vLLM child holding 11 GiB. Lesson: verify the GPU is clear before relaunching, not just that the parent died.
- **Pod CPU fitting ~55× slower than the laptop** (55 min vs ~1 min per probe). Switched the pod to GPU-only; saved ~4 h (~$3).
- **Tyler launched the k=50 analysis three times concurrently**, racing on the same files. No data lost. Then mis-diagnosed a JSON key error as file corruption and reported it as such — corrected. Third instance this session of acting before verifying state.
