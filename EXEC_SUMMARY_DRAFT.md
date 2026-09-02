# Executive summary — DRAFT v2 (voice: clear, direct, humble)

> First 1–3 pages of the Google Doc. Numbers from RESULTS.md. Figures from
> outputs/block2/figures/.

---

# Do probes on reasoning-model activations predict outcomes better than reading the trace?

**Summary.** Several recent papers report that a linear probe on a reasoning
model's activations can predict its final outcome early in the chain of
thought: [Can We Predict Alignment Before Models Finish Thinking?](https://arxiv.org/abs/2507.12428)
(probes predict response alignment from early CoT activations, reportedly
beating text classifiers by ~13 F1), [Temporal Predictors of Outcome in
Reasoning Language Models](https://arxiv.org/abs/2511.14773) ("eventual
correctness is highly predictable after only a few tokens"), and
[No Answer Needed](https://arxiv.org/abs/2509.10625) (question-only probes
predict coming accuracy). Pointing the other way,
[Current activation oracles are hard to use](https://www.lesswrong.com/posts/LXQBcztrWKhtcgQfJ/current-activation-oracles-are-hard-to-use)
found that ~95% of apparent internal signal disappeared once a reader saw
the same prefix text. Both positions cannot be generally true. I tested
whether the probe's advantage holds up against strong text-only baselines
that read the same truncated trace. I found three
things. First, the standard evaluation protocol has a confound: cutting
traces at a fixed percentage makes prefix length a near-perfect proxy for
the trace's eventual length (r = 0.9999999 by construction), and length
alone predicts correctness at about 0.70 AUC. Second, under that protocol a
tuned bag-of-words classifier (about 0.78) beats a properly tuned probe
(0.61–0.70) at every cut; when I re-ran the experiment with fixed-length
cuts, where the confound cannot exist, most of the text advantage went away
(0.78 → 0.56–0.65) and the probe-vs-text question became too close to call
at my sample size. Third, separate from any predictor: the model commits to
its answer well before it stops reasoning. An answer forced mid-trace
matches the eventual answer 59% of the time at 10% of thinking and 97% at
90%, measured without gold labels.

I should also say up front that my strongest initial result was wrong. A
"forced-answer" baseline reached 0.96 AUC and appeared to beat everything.
An adversarial review of my own pipeline showed its score was computed
against the answer key, which no deployed monitor could see. I withdrew it.
The results above are the ones that survived that review. My two
pre-registered predictions (committed to git before the data existed) were
also partly wrong, and the repo records both.

## What I did

I ran Qwen3-8B (thinking mode, 16k thinking budget) on 300 MMLU-Pro
questions, giving 289 usable traces at 82% accuracy. I chose MMLU-Pro after
measuring base rates on two other datasets and finding the model too
accurate for a correctness target (89% on MATH-500 level 5 leaves too few
failures to study). Each trace is cut at k ∈ {1, 10, 25, 50, 75, 90}% of
thinking tokens. At each cut, predictors that see exactly the same prefix
predict final correctness. I report held-out ROC-AUC with splits by problem
id, identical across all predictors:

- a **linear probe** on residual-stream activations (layers 9/18/27). Layer
  and regularization are chosen by cross-validation inside the training
  split. My first version selected the best layer on the test set; fixing
  this removed an apparent non-monotonicity and lowered the probe's peak
  from 0.76 to 0.70.
- a **tuned TF-IDF classifier** (48-config grid, grouped CV, train only).
- a **prefix-length-only** logistic model (one feature).
- a **frontier-LLM judge** (Claude Opus 5), reported separately: it scores
  0.959 at k=25 but 0.876 at k=1, where there is almost no reasoning to
  read. On a public benchmark it mostly knows the answers, so I did not pool
  it with readers of the trace.
- **forced-answer confidence** (gold-free): close the thinking block, force
  an answer, use the model's own token probability as the score.

Controls: a k=1 near-zero cut, a shuffled-label floor (500 seeds, corrected
for best-of-three layer selection), paired bootstrap CIs, and a second grid
at fixed token counts (N ∈ {64…1024}) on a fixed population of 242 traces,
which removes the length confound by construction.

## Findings

1. **The k% protocol leaks trace length.** Length alone scores about 0.70,
   flat in k. Early-prediction results that use this protocol inherit the
   leak. This is my most confident claim.
2. **Under the k% protocol, text beats the probe by a modest margin.**
   Δ(probe − best text) is about −0.10 and flat; only one of six cuts
   individually excludes zero. I would call this suggestive, not decisive.
3. **Under fixed-length cuts, the question is open.** TF-IDF falls to
   0.56–0.65. The probe ranges 0.45–0.69 with wide intervals (16 test
   negatives). At N=64 the probe leads by 0.12, which is the direction my
   second pre-registered prediction expected, but one point at this power is
   not a finding. About 4× more negatives would settle it. One caveat: the
   fixed-length population is long traces only, so the cut geometry and the
   population change together.
4. **Commitment happens early.** The forced answer equals the final answer
   59% → 68% → 86% → 93% → 97% across cuts. Much of what an early predictor
   "predicts" already exists in the model's plan. This is a behavioral,
   gold-free measurement of the commitment point that
   [Beyond the Commitment Boundary](https://arxiv.org/abs/2606.13603)
   studies with black-box methods.
5. **The withdrawn baseline, for the record.** Scoring the forced answer
   against gold gave 0.96. Scoring the same idea without gold (confidence)
   gives 0.52–0.64. The difference between those numbers is the effect of
   reading the answer key, and nothing in the standard evaluation loop
   flagged it. I found it only because I ran an adversarial review against
   my own results.

## What this means

For chain-of-thought monitoring: cheap text baselines are stronger than the
early-prediction literature's comparisons suggest, but part of that strength
came from the evaluation protocol rather than the text. If you benchmark
monitors on truncated traces, cut at fixed token counts, not fixed
fractions, and ask of every baseline whether a deployed monitor could
actually compute it. Mine could not, and it looked like the best result I
had.

## Limitations

n_test = 102, with 25 negatives (16 in the fixed-length runs), so most
individual intervals are wide and I have worded claims accordingly. One
model and one dataset; I test the correctness version of the claim, not the
alignment version in the paper that motivated this. The judge comparison is
limited by benchmark memorization. Labels come from single generations. The
fixed-length grid confounds population with protocol. Most of these were
addressable for another $10–30 of compute; I spent my budget on a second
protocol and controls instead of more data, and in hindsight more data was
the better choice.

**Repo:** [public GitHub link] — raw traces, all ten runs with provenance,
the sealed pre-registrations, and the review scripts that found the error in
my own headline result.
