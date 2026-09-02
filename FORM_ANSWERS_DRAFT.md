# Form answers — DRAFT v2 (voice pass: clear, direct, humble — 2026-09-01)

> Status: draft for the owner to rewrite in their own voice. Every number
> traces to RESULTS.md. Qs 9–12 are OWNER-ONLY and contain structure, not
> content. Neel explicitly rejects raw LLM output — these drafts are the
> skeleton and the facts; the voice must be yours.

---

## Q2. What question did you try to answer?

Recent work reports that a linear probe on a reasoning model's activations
can predict, early in its chain of thought, whether its final answer will be
correct. I asked: does that advantage survive comparison against the
strongest *cheap* predictors that read only the trace text — the same
information horizon, no internals? Concretely: Δ(k) = AUC(probe) −
AUC(best text-only reader), measured at six cut points of Qwen3-8B thinking
traces on MMLU-Pro, with predictions pre-registered in git before any data
existed.

## Q3. Why is this question interesting / why did you choose it?

Two published positions disagree. One line of work reports activation probes
beating capable text-based classifiers at predicting the outcome of a
reasoning trace; another finds ~95% of apparent internal signal disappears
once a reader sees the same prefix text. Both cannot be generally true, and
the answer matters for a live safety question: whether CoT monitoring needs
white-box access or whether reading the trace is enough. It also fits a
16-hour budget honestly: one model, one dataset, a pre-registerable headline
number, and every part of it could be tested cheaply and could fail.

## Q4. What conclusions have you reached?

Calibrated to what n=289 (52 negatives) supports:

1. **The standard k%-of-trace truncation protocol is confounded.** Cutting
   at a fixed fraction makes prefix length a near-perfect proxy for the
   trace's *eventual* length (corr = 0.9999999 by construction) — information
   no real-time monitor has. Prefix length alone predicts correctness at
   ~0.70 AUC. Early-prediction results built on this protocol inherit the
   leak.
2. **Under the leaky protocol, a tuned bag-of-words reader (~0.78) beats an
   honestly-tuned linear probe (0.61–0.70) at every cut** (Δ ≈ −0.10, flat;
   only one of six CIs excludes zero — suggestive, not decisive).
3. **Under fixed-length cuts, where the leak is impossible, the text
   reader's advantage largely evaporates** (0.78 → 0.56–0.65) and probe-vs-
   text is genuinely unresolved at my sample size. (Population caveat in Q7.)
4. **The model commits to its answer long before it stops reasoning**:
   forcing an answer mid-trace matches the eventual answer 59% at 10% of
   thinking, 97% at 90% — measured with no gold labels.
5. My most exciting result was wrong. A "forced-answer" baseline reached
   0.96 AUC, but its score was computed against the answer key (see Q6).
   The numbers above are the ones that survived the review that caught it.

## Q5. Technical setup

- **Quantities:** (a) early predictability of final-answer correctness from
  internals vs from trace text — held-out ROC-AUC of each reader at each
  cut, never accuracy (base rate 82%); (b) answer commitment — whether the
  forced mid-trace answer equals the eventual answer (gold-free).
- **Model:** Qwen/Qwen3-8B, thinking mode (16,384-token thinking budget —
  chosen after measuring that an 8k cap truncated 30% of traces, deleting
  disproportionately *incorrect* ones: survivor bias).
- **Data:** 300 MMLU-Pro questions (chosen after *measuring* base rates:
  Qwen3-8B is 89% accurate on MATH-500 level 5, leaving too few negatives;
  MMLU-Pro gave 18% error = 52 negatives). 289 usable traces; splits by
  problem-id, 187/102, identical across all readers and cuts.
- **Cuts:** k ∈ {1, 10, 25, 50, 75, 90}% of thinking tokens, plus a second
  grid at fixed N ∈ {64…1024} tokens on a fixed 242-trace population.
- **Readers:** logistic probe on residual stream (layers 9/18/27; layer and
  C selected by CV *inside the training split*), tuned TF-IDF classifier
  (48-config grid, grouped CV in-train), prefix-length-only logistic,
  frontier-LLM judge (reported separately — see Q6), and a gold-free
  forced-answer-confidence monitor.
- **Controls:** k=1 near-zero cut; shuffled-label floor (500 seeds,
  max-across-layers statistic); paired bootstrap CIs; pre-registered
  predictions (two, sealed in git with timestamps).

## Q6. Strongest evidence AGAINST these hypotheses

The strongest evidence against my hypotheses came from my own results
failing review. Three cases:

1. My headline result was false. The forced-answer baseline (interrupt the
   model, make it commit) reached 0.96 AUC and beat everything. An
   adversarial review of my own pipeline showed its score was computed
   against the gold answer: whenever the interrupted answer equalled the
   final answer, the score equalled the label by construction (97% of rows
   at k=90). On the remaining rows its AUC was 0.000. The apparent
   "gap widens with k" finding was just the copy rate approaching 1. I
   withdrew it. No deployed monitor could compute this score.
2. **Against the leak explanation:** the tuned TF-IDF's signal is nearly
   orthogonal to length (partial r ≈ 0.43 controlling for log length), so
   "the text reader is just a length detector" — my own first interpretation
   — was wrong. I corrected it in the record.
3. **Against the probe:** at the k=1 control the probe scores 0.62 and does
   not clear the noise floor — but a probe on the *question alone* in prior
   work does predict correctness, so I checked; here the confound was absent.
   The frontier judge scored 0.959 at k=25 but 0.876 at k=1, where there is
   almost no reasoning to read. It mostly knows this public benchmark, so I
   reported it separately rather than pool the most flattering number.

Both of my pre-registered predictions were also partly wrong. I predicted
Δ ≈ +0.10 at k=50; it came in around −0.10. I then doubted my second
prediction based on evidence that turned out not to transfer across
populations, and that prediction held up. All of this is timestamped in
the repo.

## Q7. Biggest limitations — and could I have addressed them?

1. **Statistical power.** n_test = 102 with 25 negatives (16 in the fixed-
   length runs). Most CIs are ±0.10–0.15; five of six k% cuts are
   individually unresolved. Could I have addressed it? Yes. About 4x the negatives would have
   cost roughly $10 more of GPU. I chose breadth (a second protocol and
   more controls) over more data, and in hindsight more data was the
   better choice.
2. **The fixed-length comparison changes two things at once** — cut geometry
   AND population (only traces ≥1024 thinking tokens survive, i.e. harder
   items). The TF-IDF collapse (0.78→0.6) cannot be fully attributed to
   removing the leak. Addressable with a length-matched resample; I ran out
   of budget.
3. **One model, one dataset; correctness, not alignment.** The flagship
   paper I'm auditing predicts *misalignment* of the response; I tested the
   correctness variant. My results speak to the protocol and the genre, not
   to that paper's specific claim.
4. **The judge comparison is bounded by memorization** — on a public
   benchmark, a frontier model's "prediction" is partly recall. Controlled
   for (k=1) but not removable without private items.

## Q8. How I used LLMs — and how I made sure it wasn't slop

[OWNER: this must be in your voice and match what you actually did. The
facts, from the project record:]

- **Tools:** Claude Code (Opus 5 / Fable 5) as the primary research
  assistant: pipeline code, experiment orchestration on rented GPUs,
  literature sweeps; Claude Opus 5 via API as the judge baseline (an
  experimental subject, not an assistant).
- **Verification, in order of rigor:**
  (a) 439 CPU-only tests, runnable without GPU/network, enforcing the
  invariants whose silent failure would fabricate results (split-by-problem,
  truncation boundaries, AUC orientation);
  (b) mutation testing: I deliberately broke the floor computation, the
  layer indexing, and the verdict logic, and checked that the test suite
  fails. An earlier version of the suite passed all three mutations, which
  meant it was not testing the decision logic at all, so it was rebuilt.
  (c) an adversarial review by separate agent instances instructed to break
  the result. They did (Q6.1). This was the most valuable step in the
  project.
  (d) hand-verification: randomly sampled traces, prefixes and forced
  answers read by eye at every stage; grader spot-checked against 40 traces.
- **What I checked vs didn't:** every reported number traces to a committed
  artifact and was recomputed at least twice by independent code paths
  (headline: independently re-derived from raw activations). I did NOT
  hand-verify all 300 traces (10-sample spot checks per stage), and single-
  generation labels mean a trace's correctness is one sample of a stochastic
  model. **Where a major error would least surprise me:** residual subtleties
  of the fixed-length population shift (Q7.2), and any place where n=16
  negatives makes a CI unstable.

## Q9. Prior experience with mech interp — OWNER ONLY
[Structure: pre-project experience (steering-vector experiment you ran);
what you learned via this project (activation harvesting, probing, the
literature). Do not inflate; he explicitly welcomes low experience.]

## Q10. 1–3 pieces of evidence you'd do good research (~100 words) — OWNER ONLY
[Must NOT be this project. Non-standard welcome. Think: things you've built,
debugged, shipped; evidence of persistence, skepticism, or fast learning.]

## Q11. Why Neel's stream — OWNER ONLY
[Honest version only. Useful raw material: his stated selection for
skepticism/truth-seeking matches how you want to work; pragmatic-interp
focus; his write-ups taught you X. Avoid flattery; be specific.]

## Q12. Likelihood you'd join Sept 28–Oct 30 — OWNER ONLY
[Check your actual calendar first. A number or "very likely/certain" +
one sentence. Do not overpromise.]

## Q13. (Optional) Anything else

Two things a reader can verify independently: (1) both sets of predictions
were committed to git *before* the corresponding data existed (commits
`2446d69`, `82c4f99`) — the timestamps, the refutations, and my incorrect
doubt of the second forecast are all in history; (2) the adversarial review
scripts that falsified my own headline are preserved verbatim in the repo
(`redteam/`), alongside every run's raw artifacts. The repo is public as of
submission.
