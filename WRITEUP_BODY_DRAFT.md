# Write-up body — DRAFT v1 (follows the executive summary in the same doc)

> Voice target: clear, direct, humble. Every number traces to RESULTS.md or the
> Run 010 / Run 008 JSON artifacts. Figures referenced are in
> outputs/block2/figures/ and regenerate from make_figures.py.

---

## 1. Background and the question

Several recent papers report that a reasoning model's final outcome can be read
out of its activations early in the chain of thought. *Can We Predict Alignment
Before Models Finish Thinking?* (arXiv:2507.12428) reports activation probes
beating capable LLM and fine-tuned text classifiers at predicting whether a
response will be misaligned, and attributes the gap to "performative" CoTs
whose text misleads a reader. *Temporal Predictors of Outcome*
(arXiv:2511.14773) reports that eventual correctness is highly predictable
after only a few tokens. *No Answer Needed* (arXiv:2509.10625) finds that
probes on the question alone predict coming correctness, though it reports the
effect falters on mathematical reasoning. And work on the commitment boundary
(arXiv:2606.13603) argues the model has often settled on its answer well before
it stops writing.

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
to 2.7% on the final dataset.

I did not pick the dataset by assumption. My first choice, MATH-500 level 5,
was rejected by measurement: Qwen3-8B is 89% accurate there (32/36 gradeable
in Run 002), and halving the model to Qwen3-4B only moved the error rate to
14% (Run 003) — the dataset, not the model, was the binding constraint, and
134 level-5 items project to roughly 15 negatives, too few to study. I then
measured MMLU-Pro before adopting it: 23% error on a 40-item calibration run,
zero ungradeable (Run 004). The final run is 300 MMLU-Pro questions, 289
usable traces, 237 correct and 52 incorrect (18.0% error). Grading is a
single letter in `\boxed{}`; a red-team audit later re-graded all 300 traces
with zero disagreements.

**Cuts.** Each trace is truncated at k ∈ {1, 10, 25, 50, 75, 90}% of its
thinking tokens. Truncation is exact to the token: no prefix contains
`</think>`, and the kept fraction never exceeds k. The k=1 cut is not quite
"the prompt alone" — it is prompt + `<think>` + a mean of 0.97% of thinking,
about 12 tokens. I state it that way because I earlier described it
imprecisely.

**Activations.** Residual-stream vectors at layers 9, 18, and 27, taken at the
last prefix token. Layer L means `hidden_states[L+1]`, never
`hidden_states[-1]`; the indexing was verified against raw activations in the
audit. Activations come from a prefill pass over the byte-identical prefix the
trace was generated from.

**The probe, and why my first protocol was wrong.** The probe is logistic
regression on the activations — deliberately simple, because the linear probe
is the claim under audit. My first protocol had two flaws: the regularization
constant was fixed at C=1.0 and never swept, and the "best layer" was chosen
by its AUC on the test set. That is selection on test data. The honest
protocol (Run 010) selects both the feature set (each layer, plus their
concatenation) and C from a 15-point grid by stratified grouped
cross-validation inside the training split only, then evaluates once on test.
All probe numbers below use the honest protocol; where the old numbers differ
I say so.

**Text baselines, and how each was tuned.**

- *Tuned TF-IDF classifier:* a 48-configuration grid over vectorizer and
  classifier settings, selected by grouped CV on the training split only. It
  sees the same training examples as the probe.
- *Prefix-length-only:* logistic regression on a single feature, log prefix
  tokens, fitted on train. This exists to price the length confound.
- *Frontier LLM judge:* Claude Opus 5 reads the prefix and predicts the
  outcome. Reported separately, for a reason given in the results.
- *Forced-answer confidence (gold-free):* close the thinking block early,
  force the model to answer, and use its own answer-token probability as the
  score. This is the valid, deployable version of a baseline I had to
  withdraw (section 4).

**Splits and floors.** Train/test is split by problem id, 187/102, identical
across every reader and every cut, so no reader can memorize problems. The
test set has 25 negatives. Controls: the k=1 near-zero cut (if a reader
scores well there, it is reading question difficulty, not reasoning); a
shuffled-label noise floor from 500 seeds, computed as a max-across-layers
statistic so the probe's best-of-three layer selection is not compared
against a floor that never selected anything; and paired bootstrap CIs on
every Δ.

**Pre-registration.** Two sets of predictions were committed to git before
the corresponding data existed: commit `2446d69` before the main grid, and
`82c4f99` before the fixed-length grid. Both are scored openly below.

**The second grid.** After the length confound surfaced, I re-cut the traces
at fixed token counts, N ∈ {64, 128, 256, 512, 1024}, on a population fixed
once across all cuts: the 242 traces with at least 1,024 thinking tokens
(192 correct / 50 incorrect; n_test=85 with 16 negatives). Within a cut every
prefix is the same length, so length carries zero information by
construction. Total compute for the whole project was small: about $3.30 for
the main grid, about $8 of judge API, about $1.05 for the fixed-length runs.

## 3. Results

**The main grid** (held-out ROC-AUC; Figure 2, fig2_readers):

| k (%) | 1 | 10 | 25 | 50 | 75 | 90 |
|---|---|---|---|---|---|---|
| Probe (honest) | 0.609 | 0.697 | 0.627 | 0.695 | 0.668 | 0.701 |
| Tuned TF-IDF | 0.732 | 0.774 | 0.781 | 0.777 | 0.776 | 0.776 |
| Length-only | 0.655 | 0.700 | 0.702 | 0.697 | 0.696 | 0.694 |

The probe sits in a flat 0.61–0.70 band. Under my first protocol it appeared
non-monotonic (0.761 at k=10, then 0.647); honest selection removes the
outlier, and paired bootstraps on adjacent cuts all include zero. The curve
is flat within noise.

Δ(probe − best text) is about −0.10 and flat in k (Figure 1, fig1_delta).
Only the k=25 cut individually excludes zero ([−0.244, −0.036]). So the
honest claim is suggestive, not decisive: in this setting, a fairly tuned
linear probe does not beat a tuned bag-of-words reader anywhere, and at one
of six cuts the deficit is individually resolved.

**The protocol leaks length.** Cutting at a fixed percentage makes prefix
length a near-perfect proxy for the trace's eventual length:
corr(prefix tokens, full-trace tokens) = 0.9999999, by construction. A
real-time monitor cannot know how long the trace will eventually be. And
length alone is a real predictor here: the one-feature model scores about
0.70, essentially flat in k, above the honest probe at most cuts. Any
early-prediction result built on k%-truncation inherits this leak. This is
my most confident finding, and it is a property of the protocol the audited
literature also uses.

**But TF-IDF is not a length detector — a correction.** In Run 007 I
concluded the text classifier was "substantially a trace-length/difficulty
detector." Run 010's length control shows that interpretation was wrong.
TF-IDF's score is essentially uncorrelated with log length (−0.04 to +0.02),
and its partial correlation with the label given length is about 0.43 —
0.42–0.44 across cuts, the strongest length-independent signal of any
reader, above the honest probe everywhere. What Run 007 actually
established is that a length-only reader *matches* TF-IDF's AUC: two readers
reaching similar numbers by different routes, not one secretly being the
other. I corrected the record before running the next experiment.

**The fixed-length grid** (Figure 4, fig4_protocol_comparison):

| N tokens | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|
| Probe (honest) | 0.685 | 0.587 | 0.452 | 0.605 | 0.618 |
| Tuned TF-IDF | 0.563 | 0.571 | 0.612 | 0.629 | 0.650 |
| Forced-confidence | 0.546 | 0.524 | 0.542 | 0.621 | 0.639 |

Two things happen. First, the TF-IDF collapse is solid: from about 0.78
under the k% protocol to 0.56–0.65 at every fixed-length cut. The text
reader's apparent dominance does not survive removing the protocol's length
information — though see the population caveat in section 6. Second,
probe-vs-text becomes genuinely unresolved. Δ swings from +0.12 at N=64 to
−0.16 at N=256 with no stable sign, and every probe CI is roughly 0.3 wide
(16 test negatives). The N=64 point — probe 0.685, Δ = +0.12 — is in the
direction my second pre-registered forecast called, and I would call it
suggestive only. One point at this power is not a finding.

**Commitment happens early** (Figure 3, fig3_commitment). Independent of any
predictor: force the model to answer mid-trace and compare that answer to the
one the trace eventually gives. This is answer identity, extracted from the
generated text on both sides — no gold label anywhere. Agreement is 59.5% at
k=10, 68.5% at 25, 86.5% at 50, 92.7% at 75, 96.5% at 90, over all 289
traces. By halfway, the answer is usually already decided. This is the
cleanest result in the project, and it says much of what an early predictor
"predicts" already exists in the model's plan.

**The judge, reported separately.** Claude Opus 5 scores 0.959 at k=25 — and
0.876 at k=1, where there is almost no reasoning to read. That is 91% of its
performance from essentially the question alone. MMLU-Pro is public; the
judge is largely solving the item itself and inferring that a weaker model
will fail a hard one. That is a different quantity from "what does this
trace tell you," so I report it as a difficulty oracle and do not pool it
into the text baseline. Pooling it would have flattered my headline. (A
dropout check — 7 of 102 rows unparseable, skewed toward negatives —
moves 0.959 to 0.955 under chance imputation; immaterial.)

**Forced-confidence**, the gold-free "just ask the model" monitor, scores
0.52–0.64: near chance with little reasoning to read, rising to about
0.62–0.64 by N≥512, always below TF-IDF. Honest but modest.

## 4. The result I withdrew

My strongest result did not survive review, and it should be on the record in
full.

The forced-answer baseline interrupts the model mid-trace, closes the
thinking block, and makes it commit to an answer. I scored that answer for
correctness and used the score as a predictor. It reached 0.96 AUC at k=90,
rising smoothly from 0.706 at k=10, and beat every other reader including
the probe. It was the headline for about a day.

An adversarial review of my own pipeline — two agent instances instructed to
break the result, plus an independent recomputation from raw artifacts —
found the flaw. The score was computed as grade(forced answer, gold) while
the label is grade(final answer, gold). Whenever the interrupted answer
equals the eventual answer, the score *is* the label, by construction. That
happens on 65% of rows at k=10 and 97% at k=90. On the rows where the score
is not simply copying the trace's own final answer, its AUC is 0.000. The
apparent "gap widens with k" was the copy rate approaching 1. And no
deployable monitor could compute this score: it requires the answer key,
which is the thing being predicted.

I withdrew it under the rule already applied to the judge: a reader is
admitted to the text baseline only if its score is a function of the prefix
alone. The analysis code now enforces that rule mechanically and records
exclusions in the output artifact. The valid, gold-free version of the same
idea — score by the model's own answer-token confidence instead of by
correctness — is the 0.52–0.64 monitor reported above. The distance between
0.52–0.64 and 0.96 is the value of reading the answer key, and nothing in
the standard evaluation loop flagged it.

## 5. What I got wrong along the way

**Pre-registration I.** I predicted Δ ≈ +0.10 at k=50. My research assistant
(Claude, running as an adversarial collaborator — see the LLM-usage answer
in the form) made an independent forecast of +0.02. Both forecasts expected
the probe to win. It came in around −0.10. The shared blind spot was the
text side: we predicted the best text reader at 0.60 and 0.70, and the tuned
classifier measured about 0.78.

**Run 007's interpretation.** After finding the length leak I wrote that the
TF-IDF classifier was substantially a length detector. Run 010's partial
correlations showed its signal is nearly orthogonal to length (partial
r ≈ 0.43). I was wrong, and the correction is in the run log.

**Doubting pre-registration II.** Based on that same length-control evidence,
I recorded — before Run 008 executed — that the sealed forecast of a TF-IDF
collapse under fixed-length cuts was "probably wrong." It was not: TF-IDF
fell from 0.78 to 0.56–0.65. A partial correlation measured on one
population did not predict behavior on another.

**The probe protocol.** My first probe results selected the best layer on the
test set with an unswept C. That manufactured an apparent non-monotonicity
(0.761 at k=10) which I initially explained away as small-n noise. It was
selection leakage, and fixing it lowered the probe's peak from 0.76 to 0.70.

## 6. Limitations, and what I would do next

**Power.** n_test = 102 with 25 negatives on the main grid, and 85 with 16
negatives on the fixed-length grid. Most individual CIs are wide, five of
six k% cuts are individually unresolved, and the fixed-length Δ is
unresolved everywhere. About 4× more negatives would cost roughly $10 of
GPU. I chose a second protocol and more controls over more data; in
hindsight more data was the better choice.

**Population confound.** The fixed-length grid changes two things at once:
the cut geometry and the population (only traces ≥1,024 thinking tokens
survive, which skews hard). The TF-IDF collapse cannot be fully attributed
to removing the leak. A length-matched resample of the original population
would separate them; I ran out of budget.

**Scope.** One model, one dataset, and the correctness version of the claim
— the flagship paper I audited predicts misalignment, not incorrectness. My
results speak to the protocol and the genre, not to that paper's specific
claim. The judge comparison is bounded by benchmark memorization and would
need private items to fix. Labels come from single generations, so each
trace's correctness is one sample of a stochastic model — the right target
for "will this run end correctly," but not a statement about the problem.

**Next, in order:** (1) 4× negatives, about $10, which resolves the
fixed-length Δ either way; (2) the length-matched population resample; (3)
an alignment-flavored target rather than correctness, which is the claim
that motivated this; (4) a second model, to see whether the flat probe band
is a Qwen3-8B fact or a general one.

## 7. Reproducibility

The repository contains every run that produced a number (RESULTS.md, runs
001–010, append-only, with commits and config hashes), the raw traces and
per-cut artifacts, the adversarial review scripts that falsified my own
headline (`redteam/`), and both sealed pre-registrations with git timestamps
(`2446d69`, `82c4f99`). All four figures regenerate from `make_figures.py`
and the committed artifacts. The pipeline's invariants — split-by-problem,
truncation boundaries, layer indexing, AUC orientation — are enforced by 439
CPU-only tests that run without a GPU or network.

---

*Draft word count (body, sections 1–7): 2,538 words excluding tables, 2,658 including them.*
