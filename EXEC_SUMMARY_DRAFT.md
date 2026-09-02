# Executive summary — DRAFT v3 (voice: clear, direct, humble)

> First 1–3 pages of the Google Doc. Numbers from RESULTS.md Run 011
> (final, 1,000 traces). Figures from outputs/expansion/figures/.

---

# Do probes on reasoning-model activations predict outcomes better than reading the trace?

**Summary.** Several recent papers report that a linear probe on a reasoning
model's activations can predict its final outcome early in the chain of
thought: [Can We Predict Alignment Before Models Finish Thinking?](https://arxiv.org/abs/2507.12428)
(probes on early-CoT activations predict whether the final response will be
safe or unsafe, beating the best text-based baselines — including capable
LLMs and fine-tuned classifiers — by an average of 13 F1),
[Temporal Predictors of Outcome in Reasoning Language Models](https://arxiv.org/abs/2511.14773)
("eventual correctness is highly predictable after only a few tokens"), and
[No Answer Needed](https://arxiv.org/abs/2509.10625) (question-only probes
predict coming accuracy). Pointing the other way,
[Current activation oracles are hard to use](https://www.lesswrong.com/posts/LXQBcztrWKhtcgQfJ/current-activation-oracles-are-hard-to-use)
found that ~95% of apparent internal signal disappeared once a reader saw
the same prefix text. Both positions cannot be generally true. I tested
whether the probe's advantage holds up against strong text-only baselines
that read the same truncated trace, on 1,000 traces with all predictions
sealed in git before the data existed.

I found three things. First, a properly tuned linear probe on activations
does not beat a tuned bag-of-words reader of the trace text at any cut. On
the standard protocol the text reader wins at all six cuts, and the gap is
statistically resolved at five of them. Second, the standard protocol has a
confound that flatters any reader: cutting traces at a fixed percentage
makes prefix length a near-perfect proxy for the trace's eventual length
(r = 0.9999999 by construction), and length alone predicts correctness at
about 0.63 AUC. When I remove the confound by cutting at fixed token counts,
the text reader's lead shrinks to a near-tie — but the probe still never
wins. Third, separate from any predictor: the model commits to its answer
well before it stops reasoning.

I should also say up front that my strongest early result was wrong. A
"forced-answer" baseline reached 0.96 AUC and appeared to beat everything.
An adversarial review of my own pipeline showed its score was computed
against the answer key, which no deployed monitor could see. I withdrew it.
The results above are the ones that survived that review. All three of my
pre-registered predictions (committed to git before each dataset existed)
were also wrong, and the repo records them.

## What I did

I ran Qwen3-8B (thinking mode, 16k thinking budget) on 1,000 MMLU-Pro
questions, giving 961 usable traces at 78% accuracy (213 incorrect). I chose
MMLU-Pro after measuring base rates on two other datasets and finding the
model too accurate for a correctness target (89% on MATH-500 level 5 leaves
too few failures to study). Each trace is cut at k ∈ {1, 10, 25, 50, 75, 90}%
of thinking tokens. At each cut, predictors that see exactly the same prefix
predict final correctness. I report held-out ROC-AUC with splits by problem
id, identical across all predictors (n_test = 337, 77 negatives):

- a **linear probe** on residual-stream activations. Both the layer (all 35)
  and the regularization are chosen by cross-validation inside the training
  split only. My first version selected the layer on the test set and swept
  nothing; fixing that, and harvesting all layers instead of three, *raised*
  the probe from about 0.68 to about 0.77 — and it still loses to text.
- a **tuned TF-IDF classifier** (grid over vectorizer and classifier,
  grouped CV, train only).
- a **prefix-length-only** logistic model (one feature), to price the leak.
- a **frontier-LLM judge** (Claude Opus 5), reported separately: on the
  original dataset it scored 0.959 at k=25 but 0.876 at k=1, where there is
  almost no reasoning to read. On a public benchmark it mostly knows the
  answers, so I did not pool it with readers of the trace.
- **forced-answer confidence** (gold-free): close the thinking block, force
  an answer, use the model's own answer-token probability as the score.

Controls: a k=1 near-zero cut, a shuffled-label floor (500 seeds, corrected
for best-of-many-layers selection), paired bootstrap CIs, and a second grid
at fixed token counts (N ∈ {64…1024}) on a fixed population of 781 long
traces, which removes the length confound by construction (length-only there
scores 0.48, at chance).

## Findings

1. **The probe never beats the best text reader.** On the k% grid, Δ(probe −
   best text) runs −0.05 to −0.09 and five of six cuts individually exclude
   zero. More data made this negative result more confident, not less. The
   probe is a real predictor (0.70–0.78); a tuned word-counter on the same
   text is simply better (0.77–0.86).
2. **The k% protocol leaks trace length.** Length alone scores about 0.63,
   flat in k. Early-prediction results built on this protocol inherit the
   leak, and a real-time monitor cannot compute it — it needs the trace's
   eventual length. This is the cleanest structural finding.
3. **Removing the leak makes it a near-tie, but the probe still does not
   win.** Under fixed-length cuts the text reader falls to 0.74–0.77 (an
   earlier, smaller run had suggested a larger collapse to 0.56–0.65; that
   was mostly noise). The probe sits at 0.67–0.74. Δ stays negative at all
   five cuts but is only resolved at N=64. So once the leak is gone, probe
   and text are comparable — with text still nominally ahead.
4. **Commitment happens early.** On the original dataset, an answer forced
   mid-trace equals the eventual answer 59% → 68% → 86% → 93% → 97% across
   cuts, measured without gold labels. Much of what an early predictor
   "predicts" already exists in the model's plan. This is a behavioral,
   gold-free measurement of the point that
   [Beyond the Commitment Boundary](https://arxiv.org/abs/2606.13603)
   studies with other methods.
5. **The withdrawn baseline, for the record.** Scoring the forced answer
   against gold gave 0.96. Scoring the same idea without gold (confidence)
   gives 0.48–0.85, rising only as the model commits. The difference between
   those numbers is the effect of reading the answer key, and nothing in the
   standard evaluation loop flagged it. I found it only because I ran an
   adversarial review against my own results.

## What this means

For chain-of-thought monitoring: on this task, cheap text baselines match or
beat a linear probe on activations at predicting final correctness, and part
of the probe's reported edge in the literature can come from an evaluation
protocol that leaks length rather than from the internals. If you benchmark
monitors on truncated traces, cut at fixed token counts, not fixed fractions,
and ask of every baseline whether a deployed monitor could actually compute
it. Mine could not, and it looked like the best result I had. None of this
refutes the flagship paper's specific claim — it predicts misalignment, not
incorrectness — but it does say the comparison has to be run carefully, and
that reading the trace text is a stronger baseline than reported.

## Limitations

n_test = 337 with 77 negatives on the k% grid (274 / 69 on the fixed-length
grid), so several individual intervals are still wide. One model and one
dataset; I test the correctness version of the claim, not the alignment
version in the paper that motivated this. The judge comparison is limited by
benchmark memorization and was run only on the original dataset. Labels come
from single generations. The fixed-length grid changes the population (long
traces only) as well as the cut geometry, so its near-tie cannot be
attributed to leak removal alone. Whole-project compute was about $11.

**Repo:** [public GitHub link] — raw traces, every run with provenance, the
three sealed pre-registrations, and the review scripts that found the error
in my own headline result.
