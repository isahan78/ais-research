# Executive summary — DRAFT v5 (voice: clear, direct, humble)

> First 1–3 pages of the Google Doc. Numbers from RESULTS.md Run 012
> (cross-fit, budget-matched) / cross_fit.json.
>
> FIGURES: three are marked inline below — `fig1_delta.png` (probe never wins),
> `fig4_population_control.png` (leak is small), and `fig3_decoupling.png`
> (commitment precedes legibility — the headline figure). For a tight 1–3 pages,
> two is ideal: I'd keep the decoupling figure and fig1 and drop fig4 to the body
> if space is short. Figure NUMBERS follow reading order, so renumber after you
> reorder the findings (the decoupling figure should lead if commitment becomes
> Finding 1). In the Google Doc, Insert > Image at each marker, delete the marker
> line, keep images ≤ ~6.5" wide. The full four-figure set lives in the write-up
> body; all PNGs are in outputs/expansion/figures/ (regenerate via
> make_figures_v2.py).

---

# How much reported "early prediction from activations" survives a fair, realizable test?

**Summary.** Several recent papers report that a linear probe on a reasoning
model's activations predicts its final outcome early in the chain of thought,
and beats text-based baselines while doing it:
[Can We Predict Alignment Before Models Finish Thinking?](https://arxiv.org/abs/2507.12428)
(probes on early-CoT activations beat capable LLM and fine-tuned text
classifiers at predicting a safe/unsafe final response, by an average of 13
F1), [Temporal Predictors of Outcome in Reasoning Language Models](https://arxiv.org/abs/2511.14773)
("eventual correctness is highly predictable after only a few tokens"), and
[No Answer Needed](https://arxiv.org/abs/2509.10625). A LessWrong analysis,
[Current activation oracles are hard to use](https://www.lesswrong.com/posts/LXQBcztrWKhtcgQfJ/current-activation-oracles-are-hard-to-use),
reports the opposite: most apparent internal signal disappears once a reader
sees the same prefix text. These are measurements of the same quantity under
different conditions, so I measured it directly, and carefully: **how much of
the probe's reported advantage over reading the trace survives a text baseline
tuned as hard as the probe, a protocol a real monitor could actually run, and
enough data to resolve the difference?**

On this task (Qwen3-8B, MMLU-Pro, correctness), the answer is: none of the
*advantage*. A linear probe on activations is a real predictor of final
correctness, but a single bag-of-words reader of the same trace text beats it
at every cut I tested — all 11 cuts, every 95% interval excluding zero, and
again when the k% grid is re-run on the fixed-length population (17 distinct
comparisons in all) — under a search budget matched to the probe. The
contribution is the measurement and two structural lessons that come with it.

## What I did

Qwen3-8B in thinking mode on 1,000 MMLU-Pro questions (961 usable, 213
incorrect; I chose MMLU-Pro after measuring that the model was too accurate on
MATH-500 to leave enough failures to study). Each trace is cut at k ∈ {1, 10,
25, 50, 75, 90}% of its thinking tokens; at each cut, predictors that see the
*same* prefix predict final correctness. I report cross-fitted ROC-AUC (out-of-
fold over all 961 traces, so all 213 negatives are test data once), split by
problem id:

- a **linear probe** on the residual stream at the last prefix token. To make
  the comparison fair I fix it to one layer chosen a priori and let only its
  regularization be tuned — the same one-config budget the text reader gets.
  (Given the generous 280-config best-of-35-layers search, the probe rises but
  still loses; that number is reported too.)
- a **TF-IDF classifier**, one fixed configuration, named as *the* text
  baseline a priori — not a per-cut maximum over several readers.
- **prefix-length-only**, one feature, to price the length confound.
- a **frontier-LLM judge** (Claude Opus 5), reported separately and measured
  only on the original 289-trace subset (k1 and k25): on a public benchmark it
  largely knows the answers, so it is a difficulty oracle, not a trace reader,
  and is kept out of the head-to-head — which is why I did not spend to expand
  it to the full set.
- **forced-answer confidence** (gold-free): close the thinking block, force an
  answer, use the model's own answer-token probability.

I also re-ran the whole k% grid on the 781 long traces used for the fixed-
length grid, so the two protocols differ only in cut geometry, not population.

## Findings

1. **A budget-matched probe never beats the text reader.** Across 6 k% cuts
   and 5 fixed-length cuts, Δ(probe − TF-IDF) is −0.04 to −0.09 and every CI
   excludes zero. The probe is genuinely predictive (0.67–0.76); the text
   reader is simply better (0.74–0.82). This is not an artifact of tuning: the
   probe here gets one config, the text reader one config. Worth stating plainly
   — the activations at a cut are computed *from* the prefix tokens, so the
   probe cannot carry more information about the label than an ideal reader of
   the text; a non-positive Δ is the expected default, and the live question is
   only whether internals make the label more *linearly accessible*. Here they
   do not.

   > **[INSERT FIGURE 1 HERE — `outputs/expansion/figures/fig1_delta.png`]**
   > *Figure 1. Budget-matched probe minus TF-IDF at every cut. All 11 points
   > are below zero and every 95% CI excludes it (cross-fit, 213/196 negatives):
   > a linear probe on activations never beats a single bag-of-words reader of
   > the same text.*

2. **The standard k% protocol leaks trace length, and no monitor can use it.**
   Cutting at a fixed percentage makes prefix length very nearly a
   deterministic function of the trace's eventual length (measured correlation
   0.99999), which a real-time monitor cannot know. Length alone then predicts
   correctness at ~0.61. The fix is to cut at fixed token counts; the argument
   for it is realizability, not effect size — the same defect that made me
   withdraw my best early result (below).
3. **But the leak is small, and it does not explain text beating the probe.**
   On one fixed population, removing the leak drops a length-only reader from
   0.56 to 0.52 (chance) but moves TF-IDF by only −0.02 and the probe barely at
   all. The much larger "text collapse" I reported in an earlier, underpowered
   pass was a population artifact (long traces are harder), not the leak. I had
   that wrong and corrected it.

   > **[INSERT FIGURE 2 HERE — `outputs/expansion/figures/fig4_population_control.png`]**
   > *Figure 2. The same 781 traces under both protocols. Removing the length
   > leak sends a length-only reader to chance (grey) but barely moves the text
   > reader (red) or the probe (blue): the leak is real but small, and does not
   > explain text beating the probe.*

4. **The model commits to its answer long before its correctness is legible.**
   An answer forced mid-trace matches the eventual answer 57% → 68% → 86% → 95%
   → 97% across cuts (gold-free, full set). Yet at the same cuts no reader
   predicts *correctness* above ~0.8. You can tell *what* the model will answer
   well before you can tell *whether it is right*. This finding does not depend
   on any tuning choice.

   > **[INSERT FIGURE 3 HERE — `outputs/expansion/figures/fig3_decoupling.png`]**
   > *Figure 3. Commitment precedes legibility. The forced-answer agreement curve
   > (purple) climbs from chance to ~97% — the model settles what it will say —
   > while the best correctness reader (red, ROC-AUC) plateaus near 0.8. Two
   > different quantities, both on [0.5, 1]: by mid-trace the answer is largely
   > fixed, yet whether it is right stays only moderately readable.*

My strongest early result was wrong, and it belongs here in one line: a
"forced-answer" baseline scored 0.96 by comparing the forced answer to the
answer key — which no deployed monitor can see. I withdrew it. The realizability
rule that killed it is the same one that says fixed-percentage cuts are unusable
(finding 2). Full account in the write-up.

## What this means

For chain-of-thought monitoring: on this task, a cheap text reader is a
stronger and more honest baseline than the early-prediction literature's
comparisons suggest, and some of the reported edge for internals can come from
an evaluation protocol that leaks information a monitor could not use. If you
benchmark monitors on truncated traces, cut at fixed token counts, hold the
population fixed across protocols, tune both sides equally, and ask of every
baseline whether a deployed monitor could compute it. None of this refutes the
flagship paper's specific claim — it predicts misalignment, not incorrectness,
on a different model — but it says the comparison has to be run carefully, and
that the interesting safety question may be legibility of *correctness*, which
stays hard even after the model has decided.

## Limitations

One model, one dataset, correctness rather than alignment. Cross-fitting uses
all 213 negatives, but that is still a few hundred failures; the probe is fixed
to one a-priori layer (the generous multi-layer search is reported alongside).
The judge is limited by benchmark memorization and was run only on the original
289-trace subset (k1, k25); I kept it there deliberately, since it is a
quarantined difficulty oracle that never enters Δ. Labels are single generations. Whole-project compute was about $11, all
of it before this analysis, which added no GPU cost.

**Repo:** [public GitHub link] — raw traces, every run with provenance, the
three sealed pre-registrations (all three wrong, and that is the point), and the
review scripts that found the error in my own headline result.
