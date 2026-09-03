# Form answers — DRAFT v4 (Run 012 cross-fit; voice: clear, direct, humble)

> Status: draft for the owner to rewrite in their own voice. Every number
> traces to RESULTS.md Run 012 / cross_fit.json. Qs 9–12 are OWNER-ONLY and
> contain structure, not content. Neel explicitly rejects raw LLM output —
> these drafts are the skeleton and the facts; the voice must be yours.

---

## Q2. What question did you try to answer?

Recent work reports that a linear probe on a reasoning model's activations
predicts, early in its chain of thought, whether its final answer will be
correct — and beats a reader of the trace text while doing it. I asked how
much of that *advantage over reading the text* survives a fair, realizable
test: a text baseline tuned as hard as the probe, a truncation protocol a real
monitor could actually run, and enough data to resolve the difference.
Concretely: Δ(k) = AUC(probe) − AUC(text reader) at cut points of Qwen3-8B
thinking traces on MMLU-Pro, with predictions pre-registered in git before any
data existed.

## Q3. Why is this question interesting / why did you choose it?

Published measurements point opposite ways: one line reports activation probes
beating text-based classifiers at predicting a trace's outcome; another finds
most apparent internal signal disappears once a reader sees the same prefix
text. Which you get depends on how hard the text side is tuned and what the
protocol leaks — so the useful contribution is a careful measurement, not a
new method. It matters for a live safety question: whether CoT monitoring needs
white-box access or whether reading the trace is enough. And it fits a 16-hour
budget honestly: one model, one dataset, a pre-registerable headline number,
every part cheap to test and able to fail.

## Q4. What conclusions have you reached?

From 1,000 traces (961 usable, 213 negatives), cross-fitted over all of them:

1. **A budget-matched linear probe never beats a single bag-of-words reader of
   the trace text — 22 of 22 comparisons, every 95% interval excluding zero.**
   Across 6 k% cuts and 5 fixed-length cuts, Δ = AUC(probe) − AUC(TF-IDF) is
   −0.04 to −0.09; the probe scores 0.67–0.76, the text reader 0.74–0.82. To
   keep it fair, the probe gets one config (a single a-priori layer) — the same
   budget as the one fixed TF-IDF. Given a generous 280-config search the probe
   rises to ~0.77 but still loses everywhere. Worth noting *why* a null is the
   default: activations at the cut are a deterministic function of the prefix
   tokens, so the probe cannot carry more label-information than an ideal text
   reader; the live question is only whether internals are more *linearly
   accessible*, and here they are not.
2. **The standard k%-of-trace protocol leaks trace length.** Cutting at a fixed
   fraction makes prefix length very nearly a deterministic function of the
   trace's *eventual* length (measured correlation 0.99999) — information no
   real-time monitor has. Length alone then predicts correctness at ~0.61. The
   fix is fixed-token cuts; the argument for it is realizability, not effect
   size (see Q6).
3. **But the leak is small and does not explain text beating the probe.** On
   one fixed population, removing the leak drops a length-only reader from 0.56
   to 0.52 (chance) but moves TF-IDF by only −0.02. An earlier underpowered
   pass of mine reported a large "text collapse" under fixed-length cuts; that
   was a population + small-n artifact, which I corrected (Q7).
4. **The model commits to its answer long before its correctness is legible.**
   A forced mid-trace answer matches the eventual answer 57% → 86% → 97% across
   cuts (gold-free, full set), yet no reader predicts *correctness* above ~0.8
   at those cuts. You can tell what the model will say well before you can tell
   whether it is right. This is the most robust result — no probe, no tuning.
5. My most exciting result was wrong. A "forced-answer" baseline reached 0.96
   AUC, but its score was computed against the answer key (see Q6). The numbers
   above are the ones that survived the review that caught it.

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
  MMLU-Pro gave 22% error = 213 negatives). 961 usable traces.
- **Cuts:** k ∈ {1, 10, 25, 50, 75, 90}% of thinking tokens, plus a second
  grid at fixed N ∈ {64…1024} tokens on a fixed 781-trace population (traces
  ≥1,024 thinking tokens, 196 negatives).
- **Fair budget:** the text reader is one fixed TF-IDF config; the probe is
  matched to one config (a single a-priori layer, residual stream at the last
  prefix token, C by inner CV). The generous 35-layer search is reported as a
  labelled upper bound. Δ = probe − TF-IDF, no per-cut max on the text side.
- **Cross-fit:** out-of-fold predictions over all 961/781 traces (5-fold
  StratifiedGroupKFold, split by problem id), so every negative is test data
  once; pooled and per-fold AUC agree. Δ CI = cluster bootstrap over problems.
- **Population control:** the k% grid re-run on the same 781 traces used for
  fixed-length, isolating cut geometry from population.
- **Other readers:** prefix-length-only (prices the leak); frontier-LLM judge
  (reported separately — Q6); gold-free forced-answer-confidence monitor.
- **Controls:** k=1 near-zero cut; shuffled-label floor (500 seeds,
  max-across-layers); three pre-registered predictions sealed in git.

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
2. **Against my own leak story:** I first reported that removing the leak made
   the text reader "collapse" — evidence the leak was doing text's work. A
   cross-fit on a fixed population showed I was wrong: the leak is small (+0.04
   AUC to a length-only reader, and does not move TF-IDF), and the apparent
   collapse was a population artifact. Corrected in the record.
3. **Against the probe:** it never leads, even given a 280-config search it
   never got matched on the text side. At the k=1 control it scores 0.69, so
   little of what it reads this early is reasoning. The frontier judge scored
   0.959 at k=25 but 0.876 at k=1, where there is almost no reasoning to read —
   it mostly knows this public benchmark, so I reported it separately rather
   than pool the most flattering number.

All three of my pre-registered predictions were also wrong. I predicted
Δ ≈ +0.10 at k=50; it came in negative. My third pre-registration (before the
full-power run) predicted the probe would stay near 0.68 and that fixed-length
cuts would put it ahead; instead it rose (to ~0.77 with a full search) and
still lost at every cut. My record on sealed forecasts is 0 for 3 — and that
each was committed before its dataset existed is what makes the record worth
anything. All of it is timestamped in the repo.

## Q7. Biggest limitations — and could I have addressed them?

1. **The probe's degrees of freedom.** The fair probe is one a-priori layer;
   the generous one searches 35. Both lose, but a richer probe (multiple layers
   concatenated, attention over the prefix) might close some of the gap — though
   by the accessibility argument it can only ever match, not exceed, what the
   text supports. I have not tested a richer probe.
2. **Power, now largely handled.** After a first reviewer flagged the pilot as
   underpowered I quadrupled the data (52 → 213 negatives) and cross-fit over
   all of it, which resolves every one of the 22 Δ comparisons. That is still a
   few hundred failures on one benchmark; the intervals are tight, not
   zero-width.
3. **One model, one dataset; correctness, not alignment.** The flagship paper
   predicts *misalignment* of the response, on a different model; I tested the
   correctness variant. My results speak to the protocol and the genre, not to
   that paper's specific claim.
4. **The judge comparison is bounded by memorization** — on a public benchmark
   a frontier model's "prediction" is partly recall. Controlled for (k=1) but
   not removable without private items.

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
  artifact; the final table cross-checks two independent code paths (pooled vs
  per-fold AUC agree, and the cross-fit reproduces the single-split direction).
  I did NOT hand-verify all 1,000 traces (10-sample spot checks per stage), and
  single-generation labels mean a trace's correctness is one sample of a
  stochastic model. **Where a major error would least surprise me:** the choice
  of a single a-priori probe layer (I mitigate by also reporting the 35-layer
  search), and any systematic bias in TF-IDF features I did not think to probe.

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
