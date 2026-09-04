# The Probe–Text Gap: how much reported early-prediction-from-activations survives a fair, realizable test

_Isahan Khan · MATS 12.0 application (Neel Nanda stream) · Qwen3-8B, MMLU-Pro_
_Repo: [public GitHub link] · every number traces to a committed artifact_

---

# Part 1 — Executive summary


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


---

# Part 2 — Full write-up


## 1. Background and the question

Several recent papers report that a reasoning model's final outcome can be read
out of its activations early in the chain of thought, and that a probe on those
activations beats a reader of the trace text. *Can We Predict Alignment Before
Models Finish Thinking?* (arXiv:2507.12428) reports activation probes beating
capable LLM and fine-tuned text classifiers at predicting a safe/unsafe final
response, by an average of 13 F1, and attributes the gap to "performative" CoTs
whose text misleads a reader. *Temporal Predictors of Outcome* (arXiv:2511.14773)
reports that eventual correctness is highly predictable after a few tokens.
*No Answer Needed* (arXiv:2509.10625) finds question-only probes predict coming
correctness. A LessWrong analysis, *Current activation oracles are hard to use*,
reports the reverse: filter out cases a reader could get from the preceding text
and most of the apparent internal signal disappears.

These are not flatly contradictory; they are measurements of the same quantity
under different baselines and protocols. That quantity is Δ(k): the held-out AUC
of a probe on internals at cut k, minus the AUC of the best predictor that reads
only the identical text prefix. If Δ is positive, internals add usable signal
there. If Δ is at or below zero, reading the trace was enough. The value of the
measurement depends entirely on how hard the text side is tuned, whether the
protocol hands either side information a real monitor could not use, and whether
there are enough failures to resolve the difference. My aim was to measure Δ
with all three controlled, and with predictions sealed in git before the data
existed.

One framing worth stating at the outset. The activations at a cut are a
deterministic function of the prefix tokens. So a probe on those activations
cannot hold more information about the label than an ideal reader of the tokens;
Δ ≤ 0 is the information-theoretic default, and a probe can only win by making
the label more *linearly accessible* than any text featurization manages. The
question is therefore not "is the signal in there" — it is "does putting a
linear model on internals beat putting one on the text." That reframing is why
a careful null is informative rather than disappointing.

## 2. Method

**Model and data.** Qwen/Qwen3-8B in thinking mode, one generation per problem,
sampled (temperature 0.6). I set the thinking budget to 16,384 tokens after
measuring that an 8,192 cap truncated 30% of traces mid-thought, deleting
disproportionately incorrect ones — a survivor bias that would leak into the
labels. On the final dataset 3.1% truncate.

I did not pick the dataset by assumption. MATH-500 level 5 was rejected by
measurement (Qwen3-8B is 89% accurate there; ~15 negatives projected). MMLU-Pro
measured 23% error on a calibration run, zero ungradeable. The final run is
1,000 MMLU-Pro questions, 961 usable, 748 correct and 213 incorrect (22.2%
error). Grading is a single letter in `\boxed{}`; a red-team audit re-graded
traces with zero disagreements.

**Cuts.** Each trace is truncated at k ∈ {1, 10, 25, 50, 75, 90}% of its
thinking tokens, exact to the token: no prefix contains `</think>`. The k=1 cut
is prompt + `<think>` + about a dozen thinking tokens, not the prompt alone; I
say so because I earlier described it imprecisely.

**Activations.** Residual-stream vector at the last prefix token — not a
mean-pool, so the probe carries no length channel of its own. Layer L means
`hidden_states[L+1]`, never `hidden_states[-1]` (post-final-RMSNorm); the
indexing was verified against raw activations.

**A fair budget.** The core comparison must give both sides the same search
budget, or the winner is whoever the analyst tuned harder. The text baseline is
a single fixed TF-IDF configuration, named as *the* text reader a priori (not a
per-cut maximum over several readers, which would be an uncorrected selection
biased toward my conclusion). So the fair probe is also one configuration: I fix
it to layer 27 — the deepest of the three pre-registered probe layers
(9/18/27), chosen before this analysis and never tuned per cut — and let only
its regularization be selected, by cross-validation inside each training fold.
I *also* report the generous probe that searches all 35 layers and 8
regularization values (280 configs, from the single-split run); the point is
that even that loses. Reporting one number without the other would be the
asymmetry I am trying to avoid.

**Cross-fit.** Rather than one 65/35 split (77 test negatives), I take out-of-
fold predictions over every trace with stratified grouped 5-fold CV, so all 213
negatives (196 on the fixed-length grid) are test data exactly once. AUC is
reported pooled over the out-of-fold predictions and also as the mean of per-
fold AUCs; the two agree throughout. Δ intervals are a cluster bootstrap over
problems.

**Population control.** The fixed-length grid restricts to the 781 traces with
≥1,024 thinking tokens. To separate protocol from population, I re-ran the k%
grid on that same 781, so k%-vs-fixed differs only in cut geometry.

**Other readers and controls.** Prefix-length-only (one feature) prices the
leak; a frontier-LLM judge (Claude Opus 5) is reported separately; forced-answer
confidence is the gold-free monitor. Controls: the k=1 near-zero cut, a
shuffled-label floor (500 seeds, max-across-layers), and three sets of
predictions sealed in git before their data existed. Whole-project compute was
about $11; this analysis added no GPU cost.

## 3. Results

**The budget-matched comparison** (cross-fit ROC-AUC; k% n=961/213 neg,
fixed-length n=781/196 neg; Figures 1–2):

| cut | probe (L27) | TF-IDF | length-only | Δ = probe − TF-IDF | CI |
|---|---|---|---|---|---|
| k1  | 0.688 | 0.744 | 0.556 | −0.057 | [−0.098, −0.018] |
| k10 | 0.711 | 0.788 | 0.607 | −0.077 | [−0.107, −0.047] |
| k25 | 0.703 | 0.793 | 0.610 | −0.089 | [−0.124, −0.057] |
| k50 | 0.755 | 0.805 | 0.610 | −0.050 | [−0.078, −0.022] |
| k75 | 0.739 | 0.814 | 0.609 | −0.076 | [−0.107, −0.047] |
| k90 | 0.757 | 0.819 | 0.609 | −0.062 | [−0.090, −0.034] |
| N64 | 0.669 | 0.748 | 0.518 | −0.079 | [−0.121, −0.037] |
| N128 | 0.705 | 0.758 | 0.518 | −0.052 | [−0.087, −0.017] |
| N256 | 0.724 | 0.762 | 0.520 | −0.038 | [−0.071, −0.006] |
| N512 | 0.711 | 0.770 | 0.521 | −0.059 | [−0.087, −0.030] |
| N1024 | 0.701 | 0.781 | 0.521 | −0.081 | [−0.111, −0.053] |

Every Δ is negative and every interval excludes zero: **11 of 11 cuts, and
again when the k% grid is re-run on the fixed-length population — 17 distinct
comparisons in all** (the five fixed-length cuts are shared between the two
analyses, so I count them once). More data and a fair budget did not rescue the
probe;
they made the loss universal and resolved. The generous 280-config probe (Run
011) reaches ~0.76–0.78 on the k% grid — higher, because searching 35 layers
helps — but it still loses to TF-IDF at every cut. Whether the probe gets one
configuration or hundreds, it does not beat a single bag-of-words reader.

**The protocol leaks length.** Cutting at a fixed percentage makes prefix length
very nearly a deterministic function of the trace's eventual length (measured
correlation 0.99999), which no real-time monitor can know. Length alone then
scores ~0.61 on the k% grid. Any early-prediction result built on k%-truncation
inherits this, and it is a property of the protocol the audited literature uses,
not of my pipeline.

**But the leak is small, and does not explain text beating the probe** (Figure
4, the population control). On the same 781 traces, moving from k% to fixed-
length drops a length-only reader from 0.557 to 0.519 — to chance — but moves
TF-IDF by only −0.021 (0.784 → 0.764) and the probe barely at all. So the leak
is worth about +0.04 AUC to a length-based reader and essentially nothing to the
text reader. The case for fixed-length cuts is realizability: a monitor cannot
compute fixed-k% length. It is not that the leak was doing the text reader's
work.

**A correction I owe the record.** In an earlier, underpowered pass (16 test
negatives) I reported that TF-IDF "collapsed" from ~0.85 to 0.56–0.65 under
fixed-length cuts, and read that as the leak being unmasked. That was wrong on
two counts. The collapse was mostly small-sample noise, and what remained was a
population effect — the fixed-length grid keeps only long traces, which are
harder — not the cut geometry. Cross-fitting on a fixed population (above) shows
TF-IDF at ~0.76–0.80 under both protocols. I corrected this before writing the
final draft.

**Commitment happens early; correctness stays hard to read** (Figure 3). Force
the model to answer mid-trace and compare that answer to the one the trace
eventually gives — answer identity on both sides, no gold label. On the full
1,000-trace set, agreement is 57% at k=10, 68% at 25, 86% at 50, 95% at 75, and
97% at 90. So by halfway the answer is largely decided. Yet at those same cuts
no reader — probe or text — predicts *correctness* above ~0.8. The model knows
what it will say well before an observer can tell whether it is right. This is
the most robust result in the project: it uses no probe, no tuning, and only the
model's own outputs.

**The judge, reported separately** (original 289-trace subset, k1 and k25 only).
Claude Opus 5 scores 0.959 at k=25 and 0.876 at k=1, where there is almost no
reasoning to read — 91% of its performance from essentially the question.
MMLU-Pro is public; the judge is largely solving the item and inferring that a
weaker model will fail a hard one. That is a different quantity, so I report it
as a difficulty oracle and do not pool it into the text baseline. Pooling it
would have flattered the headline. I deliberately did not expand it to the full
1,000-trace set or the other cuts: it costs paid API calls, it is excluded from
Δ by design, and k1-vs-k25 already makes its one point — so the spend would buy
no evidence the argument needs.

**Forced-confidence**, the gold-free "just ask the model" monitor, scores
0.58–0.82 on the k% grid: near chance early, rising as the model commits, always
below TF-IDF. Honest but modest.

## 4. The result I withdrew, and why it belongs next to the leak

My strongest early result did not survive review, and it is the same failure
mode as the length leak, so it belongs here.

The forced-answer baseline interrupts the model, closes the thinking block, and
makes it commit. I scored that answer for correctness and used the score as a
predictor. It reached 0.96 AUC at k=90 and beat everything. It was the headline
for about a day. An adversarial review of my own pipeline — two agent instances
told to break the result, plus an independent recomputation from raw artifacts —
found the flaw. The score was grade(forced answer, gold) while the label is
grade(final answer, gold); whenever the interrupted answer equals the eventual
answer, the score *is* the label, by construction (97% of rows at k=90). On the
remaining rows its AUC is 0.000. The apparent "gap widens with k" was the copy
rate approaching 1.

The reason to withdraw it is not that it copied the label — it is that no
deployed monitor could compute the score, because it needs the answer key. That
is exactly why fixed-percentage cuts are unusable: they hand the reader the
trace's eventual length, which a monitor also cannot have. One realizability
rule covers both: a reader is admitted only if its score is a function of
information available at inference time. The analysis code now enforces that
mechanically and records exclusions. The valid, gold-free version of the same
idea — score by the model's own answer-token confidence — is the 0.58–0.82
monitor above. The distance from that to 0.96 is the value of reading the answer
key, and nothing in the standard evaluation loop flagged it.

## 5. What I got wrong along the way

**Three pre-registrations, three misses.** Before the main grid I predicted
Δ ≈ +0.10 at k=50; my research assistant (Claude, as an adversarial
collaborator) predicted +0.02; both expected the probe to win, and it came in
around −0.05 to −0.09. Before the fixed-length grid I doubted a sealed forecast
that turned out closer to right than my doubt. Before the full-power run I
predicted the probe would stay near 0.68 and that fixed-length cuts would put it
ahead; instead the probe rose (to ~0.77 with a full layer search) and still
lost. My record on sealed forecasts is 0 for 3. That each was committed before
its dataset existed is what makes the record worth anything.

**The fixed-length "collapse."** Covered in section 3: an underpowered,
population-confounded reading that I corrected with cross-fitting on a fixed
population.

**The probe protocol.** My first probe selected the best layer on the test set
from three layers with an unswept C. Fixing it both removed a manufactured
non-monotonicity and, once all layers were searched honestly, *raised* the
probe by ~0.09 — the error had been hiding real signal, not inventing it. That
same +0.09 is why I now report a budget-matched probe: a swing of that size from
search alone is on the order of the effect, so the comparison has to hold search
budget equal.

## 6. Limitations, and what I would do next

**Scope.** One model, one dataset, and the correctness version of the claim —
the flagship paper predicts misalignment, on a different model. My results speak
to the protocol and the genre, not to that paper's specific claim.

**The probe's degrees of freedom.** The fair probe is one a-priori layer; the
generous one searches 35. Both lose, but a probe with a richer feature map
(multiple layers concatenated, attention over the prefix) might close some gap —
though by the accessibility argument in section 1 it can only ever match, not
exceed, what the text supports. Worth testing.

**Power.** Cross-fitting uses all 213/196 negatives and resolves every Δ, but
that is still a few hundred failures on one benchmark; the CIs are tight, not
zero-width.

**The judge and labels.** The judge comparison is bounded by benchmark
memorization and would need private items. Labels are single generations, so a
trace's correctness is one sample of a stochastic model.

**Next, in order:** (1) an alignment-flavored target rather than correctness,
the claim that motivated this; (2) a richer probe (multi-layer, pooled) under
the same realizable protocol, to test the accessibility ceiling; (3) a second
model, to see whether the probe's steady loss to text generalizes.

## 7. Reproducibility

The repository contains every run that produced a number (RESULTS.md, runs
001–012, append-only, with commits and config hashes), the raw traces and
per-cut artifacts, the adversarial review scripts that falsified my own headline
(`redteam/`), the three sealed pre-registrations with git timestamps, and the
analysis code (`cross_fit.py`, `compute_final_table.py`) with its committed
outputs. All figures regenerate from `make_figures_v2.py` and the committed
artifacts. The pipeline's invariants — split-by-problem, truncation boundaries,
layer indexing, AUC orientation — are enforced by CPU-only tests that run
without a GPU or network.

---

*Draft word count (body, sections 1–7): ~2,700 words excluding tables.*
