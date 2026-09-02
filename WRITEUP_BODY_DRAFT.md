# Write-up body — DRAFT v2 (follows the executive summary in the same doc)

> Voice target: clear, direct, humble. Every number traces to RESULTS.md
> Run 011 or the final_table.json / honest_refit.json artifacts. Figures are
> in outputs/expansion/figures/ and regenerate from make_figures.py.

---

## 1. Background and the question

Several recent papers report that a reasoning model's final outcome can be
read out of its activations early in the chain of thought. *Can We Predict
Alignment Before Models Finish Thinking?* (arXiv:2507.12428) reports
activation probes beating capable LLM and fine-tuned text classifiers at
predicting whether a response will be misaligned, by an average of 13 F1, and
attributes the gap to "performative" CoTs whose text misleads a reader.
*Temporal Predictors of Outcome* (arXiv:2511.14773) reports that eventual
correctness is highly predictable after only a few tokens. *No Answer Needed*
(arXiv:2509.10625) finds that probes on the question alone predict coming
correctness, though it reports the effect falters on mathematical reasoning.
And work on the commitment boundary (arXiv:2606.13603) argues the model has
often settled on its answer well before it stops writing.

A second camp disagrees about what these probes are reading. The LessWrong
post *Current activation oracles are hard to use* reports that after filtering
out cases an LLM could answer from the preceding text alone, about 95% of the
apparent internal signal disappeared. On that view the probes are mostly
re-reading the page.

Both positions cannot be generally true. The quantity they disagree about is
Δ(k): the held-out AUC of a probe on internals at cut point k, minus the AUC
of the strongest predictor that reads only the identical text prefix. If Δ is
positive somewhere, internals add signal there and text-only CoT monitoring
needs backup in that regime. If Δ is near zero or negative everywhere, reading
the trace was enough. Nobody had measured this curve with text baselines tuned
as hard as the probe, so I did, with both camps' predictions sealed in git
before the data existed.

## 2. Method

**Model and data.** Qwen/Qwen3-8B in thinking mode, one generation per
problem. I set the thinking budget to 16,384 tokens after measuring that an
8,192 cap truncated 30% of traces mid-thought, and that the deleted traces
were disproportionately incorrect ones — a survivor bias that would have
leaked into the labels. Raising the budget cut truncation to 8% (Run 002) and
to 3.1% on the final dataset.

I did not pick the dataset by assumption. My first choice, MATH-500 level 5,
was rejected by measurement: Qwen3-8B is 89% accurate there (32/36 gradeable
in Run 002), and halving the model to Qwen3-4B only moved the error rate to
14% (Run 003) — the dataset, not the model, was the binding constraint, and
134 level-5 items project to roughly 15 negatives, too few to study. I then
measured MMLU-Pro before adopting it: 23% error on a 40-item calibration run,
zero ungradeable (Run 004). The final run is 1,000 MMLU-Pro questions, 961
usable traces, 748 correct and 213 incorrect (22.2% error). Grading is a
single letter in `\boxed{}`; an earlier red-team audit re-graded traces with
zero disagreements.

**Cuts.** Each trace is truncated at k ∈ {1, 10, 25, 50, 75, 90}% of its
thinking tokens. Truncation is exact to the token: no prefix contains
`</think>`, and the kept fraction never exceeds k. The k=1 cut is not quite
"the prompt alone" — it is prompt + `<think>` + a small fraction of thinking,
about a dozen tokens. I state it that way because I earlier described it
imprecisely.

**Activations.** Residual-stream vectors at the last prefix token, harvested
at all 35 usable layers. Layer L means `hidden_states[L+1]`, never
`hidden_states[-1]`, which is post-final-RMSNorm and would discard the raw
residual; the indexing was verified against raw activations in the audit.
Activations come from a prefill pass over the byte-identical prefix the trace
was generated from.

**The probe, and why my first protocol was wrong.** The probe is logistic
regression on the activations — deliberately simple, because the linear probe
is the claim under audit. My first protocol had two flaws: the regularization
constant was fixed at C=1.0 and never swept, and the "best layer" was chosen
by its AUC on the test set, from only three harvested layers. That is
selection on test data. The honest protocol selects both the layer (all 35)
and C (an 8-point grid) by stratified grouped cross-validation inside the
training split only, then evaluates once on test. Fixing the leak and
harvesting every layer *raised* the probe from about 0.68 to about 0.77: I had
been understating it, not flattering it. All probe numbers below use the
honest protocol.

**Text baselines, and how each was tuned.**

- *Tuned TF-IDF classifier:* a grid over vectorizer and classifier settings,
  selected by grouped CV on the training split only. It sees the same
  training examples as the probe.
- *Prefix-length-only:* logistic regression on a single feature, log prefix
  tokens, fitted on train. This exists to price the length confound.
- *Frontier LLM judge:* Claude Opus 5 reads the prefix and predicts the
  outcome. Reported separately, for a reason given in the results, and run
  only on the original dataset.
- *Forced-answer confidence (gold-free):* close the thinking block early,
  force the model to answer, and use its own answer-token probability as the
  score. This is the valid, deployable version of a baseline I had to
  withdraw (section 4).

**Splits and floors.** Train/test is split by problem id, identical across
every reader and every cut, so no reader can memorize problems. The k% test
set is 337 problems with 77 negatives. Controls: the k=1 near-zero cut (if a
reader scores well there, it is reading question difficulty, not reasoning);
a shuffled-label noise floor from 500 seeds, computed as a max-across-layers
statistic so the probe's best-layer selection is not compared against a floor
that never selected anything; and paired bootstrap CIs on every Δ.

**Pre-registration.** Three sets of predictions were committed to git before
the corresponding data existed: commit `2446d69` before the main grid,
`82c4f99` before the fixed-length grid, and pre-registration III in
EXPERIMENT.md before the expanded run. All three are scored openly below.

**The second grid.** After the length confound surfaced, I re-cut the traces
at fixed token counts, N ∈ {64, 128, 256, 512, 1024}, on a population fixed
once across all cuts: the 781 traces with at least 1,024 thinking tokens (585
correct / 196 incorrect; n_test = 274 with 69 negatives). Within a cut every
prefix is the same length, so length carries zero information by construction —
length-only scores 0.48 there, at chance. Whole-project compute was about
$11: roughly $3.30 for the main grid, $8 of judge API, and about $5 for the
expanded overnight run.

## 3. Results

**The main grid** (held-out ROC-AUC; n_test = 337, 77 negatives; Figure 2):

| k (%) | 1 | 10 | 25 | 50 | 75 | 90 |
|---|---|---|---|---|---|---|
| Probe (honest) | 0.700 | 0.768 | 0.758 | 0.762 | 0.764 | 0.781 |
| Tuned TF-IDF | 0.772 | 0.817 | 0.828 | 0.841 | 0.852 | 0.855 |
| Length-only | 0.613 | 0.652 | 0.645 | 0.638 | 0.633 | 0.632 |
| Δ (probe − best text) | −0.074 | −0.049 | −0.068 | −0.078 | −0.088 | −0.074 |
| CI excludes 0 | no | yes | yes | yes | yes | yes |

The probe is a real predictor, in a 0.70–0.78 band that rises slightly with k.
The tuned text reader is better everywhere. Δ runs −0.05 to −0.09 and is flat
in k, and five of six cuts individually exclude zero — the exception is k=1,
where there is almost nothing to read. This is the headline, and it is the
opposite of the exciting claim: with both sides tuned honestly, a linear probe
on activations does not beat a bag-of-words reader of the same text at any
cut. Quadrupling the data from the pilot made this negative result more
confident, not less.

**The protocol leaks length.** Cutting at a fixed percentage makes prefix
length a near-perfect proxy for the trace's eventual length:
corr(prefix tokens, full-trace tokens) = 0.9999999, by construction. A
real-time monitor cannot know how long the trace will eventually be. And
length alone is a real predictor here: the one-feature model scores about
0.63, flat in k. Any early-prediction result built on k%-truncation inherits
this leak. This is a property of the protocol the audited literature also
uses, not of my pipeline.

**A correction to an earlier claim.** In Run 007 I concluded the text
classifier was "substantially a trace-length/difficulty detector." A later
length control showed that was wrong: TF-IDF's score is essentially
uncorrelated with log length, and its partial correlation with the label given
length is about 0.43 — the strongest length-independent signal of any reader.
A length-only reader *reaches* a similar AUC by a different route; it is not
that TF-IDF secretly is the length reader. I corrected the record before
running the next experiment.

**The fixed-length grid** (n_test = 274, 69 negatives; Figure 4):

| N tokens | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|
| Probe (honest) | 0.671 | 0.733 | 0.698 | 0.740 | 0.730 |
| Tuned TF-IDF | 0.743 | 0.745 | 0.753 | 0.764 | 0.771 |
| Length-only | 0.476 | 0.476 | 0.476 | 0.476 | 0.476 |
| Δ (probe − best text) | −0.072 | −0.012 | −0.056 | −0.024 | −0.041 |
| CI excludes 0 | yes | no | no | no | no |

Two things happen. First, the text reader falls from about 0.85 under the k%
protocol to 0.74–0.77 here. Some of its apparent dominance was the protocol's
length information — though less than a smaller pilot had suggested. An earlier
n=16 run put this collapse at 0.56–0.65; at 196 negatives it is milder, and I
read the earlier number as largely small-sample noise. Second, probe-vs-text
becomes a near-tie: Δ stays negative at all five cuts but is resolved only at
N=64. So once the leak is removed, probe and text are comparable — with text
still nominally ahead, and the probe never in front. Length-only sits at 0.48,
confirming the leak is gone by construction.

**Commitment happens early** (Figure 3; measured on the original dataset).
Independent of any predictor: force the model to answer mid-trace and compare
that answer to the one the trace eventually gives. This is answer identity,
extracted from generated text on both sides — no gold label anywhere.
Agreement is 59.5% at k=10, 68.5% at 25, 86.5% at 50, 92.7% at 75, and 96.5%
at 90. By halfway, the answer is usually already decided. (In the expanded run
the forced-answer generator hit a lower parse rate, so I report the commitment
curve from the original dataset, where forced answers parsed reliably; the
confidence AUC below needs no parsing and uses the full population.) This says
much of what an early predictor "predicts" already exists in the model's plan.

**The judge, reported separately** (original dataset). Claude Opus 5 scores
0.959 at k=25 — and 0.876 at k=1, where there is almost no reasoning to read.
That is 91% of its performance from essentially the question alone. MMLU-Pro
is public; the judge is largely solving the item itself and inferring that a
weaker model will fail a hard one. That is a different quantity from "what does
this trace tell you," so I report it as a difficulty oracle and do not pool it
into the text baseline. Pooling it would have flattered my headline.

**Forced-confidence**, the gold-free "just ask the model" monitor, scores
0.58–0.85 on the k% grid: near chance early, rising to about 0.85 by k≥75, and
always below TF-IDF. On the fixed-length grid it is weaker (0.47–0.60),
consistent with the commitment curve — it only knows once the model has
effectively decided. Honest but modest.

## 4. The result I withdrew

My strongest early result did not survive review, and it should be on the
record in full.

The forced-answer baseline interrupts the model mid-trace, closes the thinking
block, and makes it commit to an answer. I scored that answer for correctness
and used the score as a predictor. It reached 0.96 AUC at k=90, rising
smoothly from 0.71 at k=10, and beat every other reader including the probe.
It was the headline for about a day.

An adversarial review of my own pipeline — two agent instances instructed to
break the result, plus an independent recomputation from raw artifacts — found
the flaw. The score was computed as grade(forced answer, gold) while the label
is grade(final answer, gold). Whenever the interrupted answer equals the
eventual answer, the score *is* the label, by construction. That happens on
65% of rows at k=10 and 97% at k=90. On the rows where the score is not simply
copying the trace's own final answer, its AUC is 0.000. The apparent "gap
widens with k" was the copy rate approaching 1. And no deployable monitor
could compute this score: it requires the answer key, which is the thing being
predicted.

I withdrew it under the rule already applied to the judge: a reader is
admitted to the text baseline only if its score is a function of the prefix
alone. The analysis code now enforces that rule mechanically and records
exclusions in the output artifact. The valid, gold-free version of the same
idea — score by the model's own answer-token confidence instead of by
correctness — is the forced-confidence monitor reported above. The distance
between that and 0.96 is the value of reading the answer key, and nothing in
the standard evaluation loop flagged it.

## 5. What I got wrong along the way

**Pre-registration I.** I predicted Δ ≈ +0.10 at k=50. My research assistant
(Claude, running as an adversarial collaborator — see the LLM-usage answer in
the form) made an independent forecast of +0.02. Both forecasts expected the
probe to win. It came in around −0.08. The shared blind spot was the text
side: we predicted the best text reader far below where the tuned classifier
actually landed.

**Pre-registration II.** Based on a length control, I recorded — before the
fixed-length grid ran — that the sealed forecast of a TF-IDF collapse under
fixed-length cuts was "probably wrong." It was not wrong in direction: TF-IDF
did fall. A partial correlation measured on one population did not cleanly
predict behavior on another.

**Pre-registration III.** Before the expanded run I predicted the probe would
stay near 0.68, that fixed-length cuts would rescue it (Δ ≈ +0.04 at N=64),
and that TF-IDF would sit near 0.60 under fixed-length. All three were wrong:
the probe rose to about 0.77, the N=64 Δ was −0.07, and fixed-length TF-IDF
was about 0.74. My track record on sealed forecasts is 0 for 3. That the
predictions were committed before each dataset existed is what makes the
record worth anything.

**Run 007's interpretation.** I wrote that TF-IDF was substantially a length
detector; the partial-correlation control showed its signal is nearly
orthogonal to length. Corrected in the run log.

**The probe protocol.** My first probe results selected the best layer on the
test set, from three layers, with an unswept C. Fixing that both removed a
manufactured non-monotonicity and, once all layers were harvested, raised the
probe by about 0.09. The error had been hiding real signal, not inventing it.

## 6. Limitations, and what I would do next

**Power.** n_test = 337 with 77 negatives on the main grid, and 274 with 69
negatives on the fixed-length grid. The k% Δ is now resolved at five of six
cuts; the fixed-length Δ is resolved only at N=64. Another doubling would
resolve the rest, at roughly $10 of GPU.

**Population confound.** The fixed-length grid changes two things at once: the
cut geometry and the population (only traces ≥1,024 thinking tokens survive,
which skews hard). The near-tie there cannot be attributed to leak removal
alone. A length-matched resample of the original population would separate
them; I ran out of budget.

**Scope.** One model, one dataset, and the correctness version of the claim —
the flagship paper I audited predicts misalignment, not incorrectness. My
results speak to the protocol and the genre, not to that paper's specific
claim. The judge comparison is bounded by benchmark memorization and would
need private items to fix. Labels come from single generations, so each
trace's correctness is one sample of a stochastic model.

**Next, in order:** (1) a length-matched population resample, to separate the
protocol from the population; (2) an alignment-flavored target rather than
correctness, which is the claim that motivated this; (3) a second model, to
see whether the probe's steady loss to text is a Qwen3-8B fact or a general
one.

## 7. Reproducibility

The repository contains every run that produced a number (RESULTS.md, runs
001–011, append-only, with commits and config hashes), the raw traces and
per-cut artifacts, the adversarial review scripts that falsified my own
headline (`redteam/`), the three sealed pre-registrations with git timestamps,
and the final-table computation (`compute_final_table.py`, `final_table.json`).
All figures regenerate from `make_figures.py` and the committed artifacts. The
pipeline's invariants — split-by-problem, truncation boundaries, layer
indexing, AUC orientation — are enforced by CPU-only tests that run without a
GPU or network.

---

*Draft word count (body, sections 1–7): ~2,600 words excluding tables.*
