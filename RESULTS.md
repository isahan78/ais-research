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

---

## Run 006 — LLM-judge baseline, and the confound it exposed
**2026-08-30 · Claude Opus 5 via stdlib HTTP · k=25 then k=1 control · ~$8 API · 102 rows each, disk-cached**

### Result

| Reader | k=1 *(question only)* | k=25 |
|---|---:|---:|
| **LLM judge (Opus 5)** | **0.876** | **0.959** |
| Tuned TF-IDF | 0.732 | 0.781 |
| Forced-answer | — | 0.745 |
| Probe | 0.621 *(below floor)* | 0.647 |

### The judge is a difficulty oracle, not a text reader — and must be reported separately

The k=25 figure of 0.959 looked like a decisive win for the text side. **The k=1 control refutes that reading.** With essentially only the question in front of it — no reasoning to read — Opus 5 still reaches **0.876**, i.e. **91% of its k=25 performance**. The trace contributes ~0.08.

MMLU-Pro is a public benchmark; Opus 5 has very likely seen these items. What the judge is mostly doing is **solving the question itself** and inferring that a weaker model will fail a hard one. That is a different quantity from *"what does this trace tell you"*, which is what Δ is meant to measure.

**Consequence for the analysis:** admitting the judge into `S_text` yields Δ = **−0.312** at k=25 and **−0.255 at k=1** — but a Δ at k=1, where no reasoning exists, is not a statement about reasoning at all. It measures Opus 5's competence against a Qwen3-8B probe. **The judge is therefore reported separately as a difficulty oracle, not pooled into S_text.** Pooling it would overstate the headline by attributing Opus 5's own knowledge to information in the trace.

### Selective dropout — checked, not material
7 of 102 rows returned unparseable probabilities at k=25, and the drop was skewed (5 negatives, 2 positives — a 20% loss of the minority class vs 2.6% of the majority). Rescoring with dropped rows imputed to chance: **0.9587 → 0.9548**. Immaterial. Reported for completeness.

### What the k=1 control has now caught, across three readers
- **Probe: 0.621, below the noise floor** ⇒ probe signal at k≥10 is genuinely about the reasoning, not question difficulty.
- **TF-IDF: 0.732, and ~0.78 flat thereafter** ⇒ substantially a difficulty detector; it barely improves as the trace grows.
- **Judge: 0.876** ⇒ overwhelmingly a difficulty oracle with outside knowledge.
- **Forced-answer: rises 0.706 → 0.961 across k** ⇒ **the only reader that genuinely tracks the reasoning**, and the one that beats the probe.

That last line is the finding. It survives all three controls.

### Engineering note
`anthropic` SDK 1.2.0 fails here with an internal `TypeError: process() takes no keyword arguments`, re-raised as a misleading `APIConnectionError`; newer versions were unresolvable. The Anthropic path was rewritten to use stdlib `urllib` (matching the existing OpenRouter path), verified with a live call. One fewer dependency.

---

## Run 007 — RED-TEAM: the headline was wrong. Corrected result.
**2026-08-30 · two adversarial agents + independent second-route recomputation · no new data collected**

### The fatal flaw: forced-answer was scored against the gold answer

`forced_answer.py:325` computes the score as `grade(forced_answer, GOLD)`, while the label is `grade(final_answer, GOLD)`. So score = 1[a_k = g] and label = 1[a_final = g]: **whenever the interrupted answer equals the final answer, the score IS the label, by construction.**

| k | rows where score ≡ label | AUC (all rows) | AUC where they differ |
|---:|---:|---:|---:|
| 10 | 65% | 0.734 | **0.000** |
| 25 | 72% | 0.769 | **0.000** |
| 50 | 88% | 0.889 | **0.000** |
| 75 | 94% | 0.925 | **0.000** |
| 90 | **97%** | 0.974 | **0.000** |

On rows where it is not simply copying the trace's own final answer, its AUC is **zero**. The reported 0.706 → 0.961 rise is the answer-copy rate approaching 1 — a construction artifact, not a finding. And no deployable monitor could compute it: it requires the gold answer, which is the thing being predicted.

**By this project's own rule this is disqualifying a fortiori.** The LLM judge was excluded from `S_text` for *probably having memorised* a public benchmark; forced-answer is handed the key outright.

**WITHDRAWN as a correctness predictor.** `analysis.py` now admits only readers whose score is a function of the prefix alone (`TEXT_BASELINE_NAMES`), with exclusions and reasons recorded in `EXCLUDED_FROM_S_TEXT` so the artifact is self-describing.

### Corrected Δ(k) — the "widening" was the artifact

| k | S_probe | S_text (tuned TF-IDF) | **Δ** | CI95 | excl. 0 |
|---:|---:|---:|---:|---|:--:|
| 1 | 0.621 | 0.732 | **−0.111** | [−0.235, +0.012] | no |
| 10 | 0.761 | 0.774 | **−0.013** | [−0.151, +0.117] | no |
| 25 | 0.647 | 0.781 | **−0.135** | [−0.244, −0.036] | **yes** |
| 50 | 0.681 | 0.777 | **−0.097** | [−0.197, +0.004] | no |
| 75 | 0.677 | 0.776 | **−0.099** | [−0.233, +0.046] | no |
| 90 | 0.701 | 0.776 | **−0.075** | [−0.176, +0.031] | no |

Δ is **flat at ≈ −0.10, not widening**. One of six cuts excludes zero. The honest claim is now *"suggestive evidence that a probe does not beat a tuned text classifier in this setting"* — materially weaker than what Run 005 reported.

### Second finding: the k%-truncation protocol leaks the trace's eventual length

`corr(prefix thinking tokens at k=50, FULL trace thinking tokens) = 0.99999999` — **by construction**. Cutting at a fixed fraction hands every reader the trace's eventual length, which a real-time monitor cannot know. Prefix length alone (single feature, log tokens, fitted on train only) scores **0.655 / 0.700 / 0.702 / 0.697 / 0.696 / 0.694** across k — i.e. most of the tuned TF-IDF's ~0.78, and above the probe at four of six cuts. Its AUC is essentially **flat in k**, the signature of a pure length oracle rather than a reader of reasoning.

This is a limitation of the *protocol*, not of our implementation — and the audited literature uses the same k%-of-trace construction. **It is now a headline finding rather than a footnote.**

### What survives

- **The commitment measurement, and it needs no gold.** TRUE answer identity between the interrupted answer and the trace's final answer (extracted from `generated_text` vs the post-`</think>` span — NOT `forced_correct ≡ label`, which also counts both-wrong-with-different-answers as agreement and inflated earlier figures to 72→97%): **59.5% (k=10) → 68.5% (k=25) → 86.5% (k=50) → 92.7% (k=75) → 96.5% (k=90)**, all 289 rows. A pure prefix-and-completion comparison with no answer key; the cleanest result in the project.
- **The k=1 control.** Probe = 0.621, below the shuffled floor's p95 (0.669) ⇒ probe signal at k≥10 is not question difficulty.
- **The judge as difficulty oracle.** 0.876 at k=1 vs 0.959 at k=25.
- **Pipeline correctness.** An independent audit verified against the data: truncation exact to the token at all six cuts (no prefix contains `</think>`, kept fraction never exceeds k%), activation/label alignment 1:1 and in order, all 300 traces re-graded with 0 disagreements, `hidden_states[L+1]` confirmed the raw residual, all four baselines correctly oriented, and the paired bootstrap genuinely paired. The code was computing the wrong quantity correctly.
- **Second-route replication** of k=50 from raw artifacts with independent code: probe 0.6945 (vs 0.6805), forced-answer 0.8421 (exact), split verified disjoint.

### Further corrections
- **Probe was under-tuned and test-selected.** `probe_C=1.0` was never swept (CV picks 1e-3/1e-4) and `best_layer` is chosen on the *test* set. Honest train-CV tuning gives ≈0.59/0.70/0.70/0.69/0.73/0.70 — **flat**, so the previously reported "non-monotonicity (0.761→0.647→0.681)" is an artifact of those two choices, not small-n noise as Run 005 claimed.
- **Judge Δ was subtracting AUCs over different row sets** (102 vs 95); properly paired, k=25 is −0.272 not −0.312.
- **Split seed 0 leaves only 27 training negatives**; over 10 alternate splits the k=10 probe averages 0.704, so the reported 0.761 is a favourable draw.
- **"probe on the prompt alone" was imprecise** — the k=1 prefix is prompt + `<think>` + ~1% of thinking (mean 0.97%, ~12 tokens).
- **"ungradeable 3.7%" was wrong** — ungradeable is 3/300 = 1.0%; 3.7% is total exclusions (8 truncated + 3 ungradeable).
- `k90/results.json` lost its `text_floor` block to a write-order race; `JUDGE_EFFORT` is recorded in notes but never actually sent.

### Assessment
The project's instrument caught its own headline baseline cheating, four days before submission, using a control built for a different purpose. The corrected result is weaker and more honest: **a probe on internals does not clearly beat a tuned text classifier, most of the text side is trace length, and the length signal is an artifact of the truncation protocol the literature uses.**

---

## Run 010 — Honest probe + length control (local, no GPU)
**2026-08-30 · analysis of existing Block-2 artifacts · scripts: `honest_probe.py`, `length_control.py`**

### Honest probe (layer AND C selected by CV inside the training split only)

| k | reported (test-selected, C=1.0) | **honest** |
|---:|---:|---:|
| 1 | 0.621 | 0.609 |
| 10 | 0.761 | **0.697** |
| 25 | 0.647 | 0.627 |
| 50 | 0.681 | 0.695 |
| 75 | 0.677 | 0.668 |
| 90 | 0.701 | 0.701 |

The honest curve sits in a flat 0.61–0.70 band. The k=10 outlier (0.761) was the test-selection artifact. Adjacent-cut differences: paired-bootstrap CIs all include zero ⇒ **the previously reported "non-monotonicity" does not survive honest selection; the curve is flat within noise.** These are the probe numbers the write-up quotes.

### Length control — and a result that CORRECTS Run 007's interpretation

Partial correlation of each reader's score with the label, controlling for log prefix length (test rows):

| k | probe (honest) | **tuned TF-IDF** | length-only |
|---:|---:|---:|---:|
| 10 | 0.33 | **0.39** | −0.06 |
| 25 | 0.15 | **0.42** | −0.06 |
| 50 | 0.34 | **0.44** | −0.06 |
| 75 | 0.33 | **0.44** | −0.06 |
| 90 | 0.08 | **0.44** | −0.06 |

And `corr(TF-IDF score, log length)` ≈ **−0.04 to +0.02** — essentially zero.

**Run 007 said the TF-IDF classifier was "substantially a question-difficulty/trace-length detector." That interpretation is WRONG and is hereby corrected.** TF-IDF's signal is nearly orthogonal to length and survives length control at partial r ≈ 0.42–0.44 — the strongest length-independent signal of any reader, above the honest probe everywhere. What Run 007 actually established is that *a length-only reader matches TF-IDF's AUC* — two readers reaching similar AUC by **different routes**, not one reader secretly being the other. The length leak is still real (corr(prefix len, full len) = 0.9999999, and length alone scores ~0.69), but it is a property of the PROTOCOL and of the length-only reader — not an explanation of TF-IDF.

**Consequence for sealed Pre-registration II, noted BEFORE Run 008 executes:** the sealed forecast predicts S_text (TF-IDF) "drops materially" under fixed-length cuts because it "loses the length crutch." This length-control evidence suggests TF-IDF has no length crutch to lose, i.e. **the sealed forecast is probably wrong a second time.** Recorded now so the pre-registration's test is clean: if TF-IDF holds ~0.78 under fixed-length cuts, the forecast fails and Δ stays negative.

### Bottom line after all corrections
A fairly-tuned linear probe on Qwen3-8B's residual stream (flat ~0.61–0.70) does not beat a tuned bag-of-words reader of the trace text (flat ~0.77–0.78, length-independent) at any cut, on n_test=102. The commitment curve (59→97%) and the k=1 controls stand. Run 008 (fixed-length cuts) now tests the length-leak hypothesis directly, with its prediction sealed and already under suspicion.

### Process note
Built during three laptop-sleep interruptions that killed the analysis agent mid-work; artifacts survived on disk each time and the final tests were completed by hand. Suite: 439 passing.

---

## Runs 008/009 — Fixed-length cuts + gold-free confidence: the final data
**2026-08-31 01:10–02:04 UTC · RTX 4090 (US-NC-1, CUDA 13.0) · commit `24dc0d4` · pod terminated & verified · session ≈ $1.05 (incl. a bad-host detour: first pod drew a CUDA 12.8 host that current torch refuses)**

Reused the 300 Block-2 traces (no generation cost). Population fixed once: traces with ≥1024 thinking tokens ⇒ **242 traces, 192/50**, identical at every cut — survivor bias structurally impossible. Cuts at exactly N ∈ {64,128,256,512,1024} thinking tokens ⇒ within a cut, prefix length carries zero information: **the Run 007 length leak cannot exist here by construction.**

### Honest results (probe: layer+C by train-CV, Run 010 protocol; n_test=85, only 16 negatives)

| N | probe (honest) | CI95 | tuned TF-IDF | forced-confidence (gold-free) | Δ (probe−TF-IDF) |
|---:|---:|---|---:|---:|---:|
| 64 | 0.685 | [0.52, 0.83] | 0.563 | 0.546 | **+0.12** |
| 128 | 0.587 | [0.44, 0.73] | 0.571 | 0.524 | +0.02 |
| 256 | 0.452 | [0.29, 0.61] | 0.612 | 0.542 | −0.16 |
| 512 | 0.605 | [0.46, 0.74] | 0.629 | 0.621 | −0.02 |
| 1024 | 0.618 | [0.47, 0.75] | 0.650 | 0.639 | −0.03 |

### Findings

**1. The TF-IDF collapse is the solid result.** From ~0.78 (k% protocol) to **0.56–0.65** at every fixed-length cut. The text reader's apparent dominance in Block 2 does not survive removing the protocol's length information. **Confound to state plainly:** two things changed at once — the cut geometry AND the population (all-long ⇒ all-hard traces). Run 010 showed TF-IDF's k%-signal was length-orthogonal (partial r ≈ 0.43), so the likeliest reconciliation is that its signal was *difficulty vocabulary*, which has little variance inside an all-hard population. Fixed-length cuts and population restriction each destroy part of its edge; this design cannot fully separate the two.

**2. Probe vs text under fixed cuts: UNRESOLVED.** Δ swings +0.12 → −0.16 with no stable sign; every CI is ~0.3 wide (16 negatives). The N=64 point (probe 0.685, CI barely excluding 0.5, Δ=+0.12) is suggestive — the direction Pre-registration II called — but one point at n=85 is not a claim.

**3. Pre-registration II: substantially vindicated on its load-bearing clause, after we publicly doubted it.** Sealed forecast: "S_text drops materially; probe roughly unchanged; Δ shrinks toward zero, possibly slightly positive at small budgets." The S_text drop happened (0.78→~0.6). The Δ sign at small N matches at face value but is unresolved statistically. Notably, Run 010's length-control evidence led us to record — before this run — that the forecast was "probably wrong." It wasn't. The lesson cuts the other way this time: a partial correlation on one population does not predict behaviour on another.

**4. Forced-confidence (the gold-free monitor) is honest but weak-to-moderate:** ~0.52–0.55 with little reasoning to read, rising to ~0.62–0.64 by N≥512 — always below TF-IDF. The deployable version of "just ask the model" carries real but modest signal, nothing like its invalid gold-reading predecessor's 0.96.

### Assessment
Data collection for this project is now closed. Final tally: the k%-truncation protocol used by the audited literature leaks eventual trace length; under a leak-free protocol the strongest text reader loses most of its advantage and the probe-vs-text question becomes genuinely open at this sample size — an honest "unresolved, here is exactly what it would take to resolve it" (≈4× the negatives).

---

## Run 011 — FINAL, at full power: 1,000 traces, all 35 layers
**2026-09-02 · overnight pod (RTX 4090, terminated & verified) + local honest refit · ~$5 session, ~$11 project · `compute_final_table.py`, `final_table.json`**

Motivated by an external review. Three upgrades over Runs 005–009: **1,000 traces (213 negatives, was 52)**, **all 35 layers harvested (was 3)**, and **layer+C chosen by CV inside the training split** (Run 010 protocol) rather than on test.

### k% grid (n_test=337, 77 negatives)

| k | probe (layer) | tuned TF-IDF | forced-conf | length-only | Δ = probe − best text | CI excl 0 |
|---:|---:|---:|---:|---:|---:|:--:|
| 10 | 0.768 (L34) | **0.817** | 0.582 | 0.652 | −0.049 [−0.097,−0.003] | **yes** |
| 25 | 0.758 (L33) | **0.828** | 0.623 | 0.645 | −0.068 [−0.120,−0.021] | **yes** |
| 50 | 0.762 (L34) | **0.841** | 0.754 | 0.638 | −0.078 [−0.120,−0.039] | **yes** |
| 75 | 0.764 (L26) | **0.852** | 0.846 | 0.633 | −0.088 [−0.138,−0.038] | **yes** |
| 90 | 0.781 (L22) | **0.855** | 0.847 | 0.632 | −0.074 [−0.118,−0.030] | **yes** |

### Fixed-length grid (n_test=274, 69 negatives)

| N | probe | TF-IDF | forced-conf | length-only | Δ | CI excl 0 |
|---:|---:|---:|---:|---:|---:|:--:|
| 64 | 0.671 | 0.743 | 0.487 | 0.476 | −0.072 [−0.141,−0.008] | **yes** |
| 128 | 0.733 | 0.745 | 0.477 | 0.476 | −0.012 [−0.060,+0.037] | no |
| 256 | 0.698 | 0.753 | 0.470 | 0.476 | −0.056 [−0.112,+0.002] | no |
| 512 | 0.740 | 0.764 | 0.529 | 0.476 | −0.024 [−0.061,+0.015] | no |
| 1024 | 0.730 | 0.771 | 0.598 | 0.476 | −0.041 [−0.086,+0.003] | no |

### The three answers

1. **Does the probe ever beat the best text reader?** **No.** Δ negative at all 11 cuts; on the k% grid all five CIs exclude zero. More data made the negative result *more* confident, not less.
2. **Does TF-IDF still collapse under fixed-length cuts?** **Partially, and less than at n=16.** It falls from ~0.85 (k%) to ~0.74–0.77 (fixed-length) — real, but the earlier 0.56–0.65 collapse was substantially small-n noise. Fixed-length Δ is mostly indistinguishable from zero: **probe and text are comparable once the length leak is removed, both ~0.7–0.77.**
3. **Pre-registration III: WRONG (third sealed miss).** Predicted probe ~unchanged (rose to ~0.77), fixed-length Δ at N=64 ≈ +0.04 (was −0.07), TF-IDF ≈ 0.60 (was 0.74). Directionally wrong that fixed-length cuts would rescue the probe. Track record on sealed forecasts: 0/3. That the predictions were sealed in git before each dataset existed is what makes the record trustworthy.

### What is now the honest headline
On the standard k% protocol at full power, a **tuned bag-of-words reader of the trace text significantly beats a linear probe on activations** at predicting final correctness, at every cut (Δ −0.05 to −0.09, CIs exclude zero). The probe was *understated* in earlier runs (test-side layer selection + only 3 layers); with honest CV over 35 layers it rises to ~0.76–0.78 but still loses to text. **The length leak is real but is not the whole story** — even under leak-free fixed-length cuts, text edges the probe or ties it; the probe never wins. Forced-confidence (gold-free) is weak early (~0.5) and only becomes competitive very late (~0.85 at k≥75), consistent with the commitment curve: it only knows once the model has effectively decided.

### Process (stated plainly)
The result files (traces, 35-layer activations, confidence generations) were computed on the pod and verified on-disk before teardown. The final table cost four buggy analysis passes to get right: a stalled sub-agent (waited on jobs it could not receive), a heredoc written to a stale directory, a wrong confidence field name, and `GroupShuffleSplit` returning index arrays so a printed `n_test` read 166,672 — each caught and fixed, none affecting the committed AUCs (verified by reproduction). No result was reported from code with a known error still in it.

---

## Run 012 — cross-fit, budget-matched, population-controlled (answers external review #2)
**2026-09-02 · local CPU, no new GPU · `cross_fit.py`, `cross_fit.json`**

Three upgrades over Run 011, each targeting a specific reviewer objection:

1. **Budget-matched probe.** The text baseline is ONE fixed TF-IDF config, so
   the fair probe is also one config: a single layer fixed a priori (layer 27,
   the deepest of the pre-registered 9/18/27 set — never tuned per cut), with
   only C chosen by inner CV. This neutralises the "probe advantage is search
   budget" critique — the +0.09 the probe gained in Run 011 came from a 35-layer
   search the text side never got. Δ is now probe − TF-IDF, no per-cut max on
   the text side (removes the uncorrected selection that biased toward our
   conclusion). The generous best-of-35 probe (Run 011) is kept as a labelled
   upper bound.
2. **Cross-fit.** Out-of-fold predictions over EVERY trace (StratifiedGroupKFold,
   5 folds), so all 213 negatives (k%) / 196 (fixed-length) are test data once,
   instead of 77 / 69 in a single split. Pooled AUC and mean-per-fold AUC agree
   throughout. Δ CI = cluster bootstrap over problems on pooled OOF preds.
3. **Population control.** The k% grid re-run on the SAME 781 long-trace
   population used for fixed-length, so k%-vs-fixed differs only in cut geometry.

### Headline (cross-fit, budget-matched)

| cut | probe (L27) | TF-IDF | length-only | Δ = probe−TFIDF | CI |
|---|---|---|---|---|---|
| k1  | 0.688 | 0.744 | 0.556 | −0.057 | [−0.098, −0.018] |
| k10 | 0.711 | 0.788 | 0.607 | −0.077 | [−0.107, −0.047] |
| k25 | 0.703 | 0.793 | 0.610 | −0.089 | [−0.124, −0.057] |
| k50 | 0.755 | 0.805 | 0.610 | −0.050 | [−0.078, −0.022] |
| k75 | 0.739 | 0.814 | 0.609 | −0.076 | [−0.107, −0.047] |
| k90 | 0.757 | 0.819 | 0.609 | −0.062 | [−0.090, −0.034] |
| abs64 | 0.669 | 0.748 | 0.518 | −0.079 | [−0.121, −0.037] |
| abs128 | 0.705 | 0.758 | 0.518 | −0.052 | [−0.087, −0.017] |
| abs256 | 0.724 | 0.762 | 0.520 | −0.038 | [−0.071, −0.006] |
| abs512 | 0.711 | 0.770 | 0.521 | −0.059 | [−0.087, −0.030] |
| abs1024 | 0.701 | 0.781 | 0.521 | −0.081 | [−0.111, −0.053] |

**All 11 CIs exclude zero.** More power + a fair budget did not rescue the probe
— it made the loss universal and significant, including the fixed-length cuts
that were unresolved in Run 011.

### Population control (k% AND fixed-length on the same 781 traces)

- **length-only:** k% 0.542–0.566 vs fixed-length 0.518–0.521. The leak is worth
  **+0.038 AUC** to a pure length reader, and vanishes to chance (~0.52) once
  removed. Real, small, and — critically — unusable by any online monitor
  (it needs the trace's eventual length).
- **TF-IDF:** k% 0.784 vs fixed-length 0.764 on the SAME population — a −0.021
  change. So removing the leak costs the text reader almost nothing.
- **The Run 011 "text collapse under fixed-length" (0.85→0.75) was a POPULATION
  artifact, not the leak.** Full-961 k% TF-IDF 0.794 vs long-781 k% 0.784; the
  drop to ~0.76 came from restricting to long (harder) traces, not from the cut
  geometry. Corrected.
- All 11 population-control Δ also exclude zero.

### The three answers, now settled
1. **Does the probe ever beat the best text reader?** No — 22/22 comparisons,
   every CI excludes zero, under a budget matched to the text side.
2. **Does removing the length leak change the verdict?** No. The leak is real
   but small (+0.038 to a length reader) and does not preferentially flatter
   text; the case for fixed-length cuts is REALIZABILITY (a monitor cannot
   compute fixed-k% length — the same defect that killed our forced-answer
   baseline), not effect size.
3. **Was the earlier fixed-length "collapse" real?** No — population + small-n
   artifact. Owned in the record.

### Framing added
Activations at the cut are a deterministic function of the prefix tokens, so the
probe cannot hold more information about the label than an ideal reader of the
text; Δ ≤ 0 is the information-theoretic default, and the live question is
whether internals make the label more *linearly accessible*. Here they do not.
The probe reads the LAST prefix token (not a mean-pool), so it carries no length
channel of its own. Generation was sampled (temp 0.6), so each trace is one
stochastic draw.
