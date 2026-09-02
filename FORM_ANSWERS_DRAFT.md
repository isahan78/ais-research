# Form answers — DRAFT v3 (Run 011 final numbers; voice: clear, direct, humble)

> Status: draft for the owner to rewrite in their own voice. Every number
> traces to RESULTS.md Run 011 / final_table.json. Qs 9–12 are OWNER-ONLY and
> contain structure, not content. Neel explicitly rejects raw LLM output —
> these drafts are the skeleton and the facts; the voice must be yours.

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

From 1,000 traces (961 usable, 213 negatives; k% test set n=337, 77
negatives):

1. **A properly tuned linear probe on activations does not beat a tuned
   bag-of-words reader of the same trace text at any cut.** On the k% grid the
   probe runs 0.70–0.78 and the text reader 0.77–0.86; Δ = AUC(probe) −
   AUC(best text) is −0.05 to −0.09, flat in k, and five of six cuts
   individually exclude zero. Quadrupling the pilot made this negative result
   *more* confident, not less. Fixing my probe protocol and harvesting all 35
   layers *raised* the probe from ~0.68 to ~0.77 — it was understated before,
   and it still loses.
2. **The standard k%-of-trace truncation protocol is confounded.** Cutting at
   a fixed fraction makes prefix length a near-perfect proxy for the trace's
   *eventual* length (corr = 0.9999999 by construction) — information no
   real-time monitor has. Prefix length alone predicts correctness at ~0.63
   AUC. Early-prediction results built on this protocol inherit the leak.
3. **Removing the leak (fixed-length cuts) makes it a near-tie, but the probe
   still never wins.** The text reader falls to 0.74–0.77 and the probe to
   0.67–0.74; Δ stays negative at all five cuts but is resolved only at N=64.
   (A smaller pilot had suggested a larger text collapse to 0.56–0.65; that
   was mostly small-sample noise. Population caveat in Q7.)
4. **The model commits to its answer long before it stops reasoning**: forcing
   an answer mid-trace matches the eventual answer 59% at 10% of thinking, 97%
   at 90% — measured with no gold labels.
5. My most exciting result was wrong. A "forced-answer" baseline reached 0.96
   AUC, but its score was computed against the answer key (see Q6). The
   numbers above are the ones that survived the review that caught it.

## Q5. Technical setup

- **Quantities:** (a) early predictability of final-answer correctness from
  internals vs from trace text — held-out ROC-AUC of each reader at each
  cut, never accuracy (base rate 78%); (b) answer commitment — whether the
  forced mid-trace answer equals the eventual answer (gold-free).
- **Model:** Qwen/Qwen3-8B, thinking mode (16,384-token thinking budget —
  chosen after measuring that an 8k cap truncated 30% of traces, deleting
  disproportionately *incorrect* ones: survivor bias).
- **Data:** 1,000 MMLU-Pro questions (chosen after *measuring* base rates:
  Qwen3-8B is 89% accurate on MATH-500 level 5, leaving too few negatives;
  MMLU-Pro gave 22% error = 213 negatives). 961 usable traces; splits by
  problem-id, identical across all readers and cuts (k% test set n=337, 77
  negatives).
- **Cuts:** k ∈ {1, 10, 25, 50, 75, 90}% of thinking tokens, plus a second
  grid at fixed N ∈ {64…1024} tokens on a fixed 781-trace population (traces
  ≥1,024 thinking tokens; n_test=274, 69 negatives).
- **Readers:** logistic probe on residual stream (all 35 layers; layer and C
  selected by CV *inside the training split*), tuned TF-IDF classifier
  (grid, grouped CV in-train), prefix-length-only logistic, frontier-LLM judge
  (reported separately — see Q6), and a gold-free forced-answer-confidence
  monitor.
- **Controls:** k=1 near-zero cut; shuffled-label floor (500 seeds,
  max-across-layers statistic); paired bootstrap CIs; pre-registered
  predictions (three, sealed in git with timestamps).

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
3. **Against the probe:** it never leads. At the k=1 control it scores 0.70,
   close to the text reader there, so most of what it "reads" this early is
   question difficulty, not reasoning. The frontier judge scored 0.959 at k=25
   but 0.876 at k=1, where there is almost no reasoning to read — it mostly
   knows this public benchmark, so I reported it separately rather than pool
   the most flattering number.

All three of my pre-registered predictions were also wrong. I predicted
Δ ≈ +0.10 at k=50; it came in around −0.08. My third pre-registration (before
the full-power run) predicted the probe would stay near 0.68 and that
fixed-length cuts would put it ahead; instead it rose to ~0.77 and still lost.
My track record on sealed forecasts is 0 for 3 — and that each was committed
before its dataset existed is what makes the record worth anything. All of it
is timestamped in the repo.

## Q7. Biggest limitations — and could I have addressed them?

1. **Statistical power.** n_test = 337 with 77 negatives (274 / 69 in the
   fixed-length runs) — after I quadrupled the pilot in response to a review
   that flagged it as underpowered. The k% Δ is now resolved at five of six
   cuts; the fixed-length Δ is resolved only at N=64. Another doubling would
   settle the rest for roughly $10 more of GPU.
2. **The fixed-length comparison changes two things at once** — cut geometry
   AND population (only traces ≥1024 thinking tokens survive, i.e. harder
   items). The near-tie there cannot be attributed to removing the leak alone.
   Addressable with a length-matched resample; I ran out of budget.
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
  artifact and was recomputed at least twice by independent code paths (the
  final probe table cross-checks final_table.json against a separate
  honest-refit run). I did NOT hand-verify all 1,000 traces (10-sample spot
  checks per stage), and single-generation labels mean a trace's correctness
  is one sample of a stochastic model. **Where a major error would least
  surprise me:** residual subtleties of the fixed-length population shift
  (Q7.2), and the fixed-length Δ, which is resolved at only one of five cuts.

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

Two things a reader can verify independently: (1) all three sets of
predictions were committed to git *before* the corresponding data existed
(commits `2446d69`, `82c4f99`, and pre-registration III in EXPERIMENT.md) —
the timestamps and all three refutations are in history; my forecasts went
0 for 3; (2) the adversarial review scripts that falsified my own headline are
preserved verbatim in the repo (`redteam/`), alongside every run's raw
artifacts. The repo is public as of submission.
