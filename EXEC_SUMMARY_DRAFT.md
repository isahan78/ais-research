# Executive summary — DRAFT v1 (target: first 1–3 pages of the Google Doc)

> Owner: rewrite in your voice before it ships. Every number is from
> RESULTS.md. Figures referenced are in outputs/block2/figures/.

---

# Do probes on reasoning-model activations beat reading the trace? An audit that caught its own headline

**TL;DR.** Recent work reports that linear probes on a reasoning model's
activations predict, early in the chain of thought, whether its final answer
will be correct. I tested whether that advantage survives strong text-only
baselines that read the same truncated trace. Three results. (1) The standard
**k%-of-trace truncation protocol is confounded**: it makes prefix length a
near-perfect proxy for the trace's *eventual* length (r = 0.9999999 by
construction) — information no real-time monitor has — and length alone
predicts correctness at ~0.70 AUC. (2) Under that protocol a tuned
bag-of-words classifier (~0.78) beats an honestly-tuned probe (0.61–0.70) at
every cut; under **fixed-length cuts, where the leak is impossible, the text
advantage largely evaporates** (0.78 → 0.56–0.65) and probe-vs-text is
unresolved at my sample size. (3) Independent of any predictor: **the model
commits to its answer long before it stops reasoning** — an answer forced
mid-trace matches the eventual answer 59% at 10% of thinking and 97% at 90%,
measured without gold labels. A fourth result is about process: **my most
exciting finding — a "forced-answer" baseline at 0.96 AUC — was false**, and
the adversarial review I ran against my own pipeline is what caught it: its
score was computed against the answer key. Everything below survived that
review; the thing I'm most confident of is the audit trail (pre-registered
predictions sealed in git before data; both partly wrong; every number
traceable to a committed artifact in the public repo).

## What I did

Qwen3-8B (thinking mode, 16k thinking budget) on 300 MMLU-Pro questions →
289 usable traces, 82% correct — the negative class exists because I
*measured* base rates first and rejected two datasets where the model was
too strong (MATH-500 L5: 89% correct, only ~15 possible negatives). Each
trace is cut at k ∈ {1, 10, 25, 50, 75, 90}% of thinking tokens; at every
cut, predictors that see exactly the same prefix race to predict final
correctness (held-out ROC-AUC, split by problem, identical across readers):

- **linear probe** on residual-stream activations (layers 9/18/27; layer and
  regularization chosen by CV *inside the training split* — selecting them
  on test, as I first did, inflates the probe and manufactures a spurious
  non-monotonicity);
- **tuned TF-IDF classifier** (48-config grouped CV, in-train);
- **prefix-length-only** logistic (one feature);
- a **frontier-LLM judge** (Claude Opus 5) — reported separately: it scores
  0.959 at k=25 but 0.876 at k=1 *with no reasoning to read*; on a public
  benchmark it is a difficulty oracle, and pooling it would attribute its
  own knowledge to the trace;
- **forced-answer confidence** (gold-free): close the think block, force an
  answer, score the model's own token probability.

Controls: k=1 near-zero cut, shuffled-label floor (500 seeds, max-statistic),
paired-bootstrap CIs, and a second grid at **fixed token counts**
(N ∈ {64…1024}) on a fixed 242-trace population, which makes the length leak
structurally impossible. Two forecasts were pre-registered in git before the
corresponding data existed (commits `2446d69`, `82c4f99`).

## Findings (calibrated)

1. **Protocol confound (my strongest claim).** Cutting at a fixed *fraction*
   hands every reader the trace's eventual length. Length alone: ~0.70 AUC,
   flat in k — a pure length oracle. Early-prediction results using this
   protocol inherit the leak. *(Figure: protocol comparison.)*
2. **Under the leaky protocol, text wins — modestly.** Δ(probe − best text)
   ≈ −0.10, flat across cuts; only k=25 individually excludes zero.
   *Suggestive, not decisive.*
3. **Remove the leak and the race reopens.** Fixed-length cuts: TF-IDF falls
   to 0.56–0.65; probe 0.45–0.69 with wide CIs (16 test negatives). At N=64
   the probe leads (+0.12) — the direction my second pre-registered forecast
   predicted — but this is one point at low power. Honest verdict:
   **unresolved**; ~4× the negatives would resolve it.
   *Caveat: the fixed-length population is long-trace-only (harder items),
   so cut geometry and population change together.*
4. **Commitment is early and gold-free.** Forced answer = final answer:
   59% → 68% → 86% → 93% → 97% across cuts. Whatever probes or monitors
   measure mid-trace, the answer they are "predicting" mostly already
   exists.
5. **The invalidated result, kept on display.** The forced-answer *accuracy*
   baseline hit 0.96 and beat everything — because its score used the gold
   key: where it wasn't simply copying the trace's own final answer, its AUC
   was 0.000. Withdrawn. The gold-free version (confidence) scores an honest
   0.52–0.64. The 0.96 → 0.6 gap is what "reads the answer key" is worth.

## What this means

For CoT monitoring: on this task, cheap text baselines are stronger than the
early-prediction literature's comparisons suggest — but part of that
strength was an artifact of the evaluation protocol itself. Anyone
benchmarking internals-based monitors against truncated traces should cut at
fixed token counts, not fixed fractions, and should ask of every baseline:
*could a deployed monitor actually compute this?* Mine couldn't, it scored
0.96, and nothing in the standard evaluation loop would have flagged it.

## Limitations

n_test = 102 (25 negatives; 16 in fixed-length runs) — most individual CIs
are wide, and claims are worded accordingly. One model, one dataset;
correctness, not alignment (the flagship paper I audit predicts response
misalignment). The judge comparison is bounded by benchmark memorization.
Single-generation labels. Fixed-length grid confounds population with
protocol. Each of these was addressable for roughly $10–30 more compute; I
chose breadth (a second protocol + controls) over power, and in hindsight
power was the better buy.

**Repo:** [public GitHub link] — all raw traces, every run with provenance
(RESULTS.md, Runs 001–010), the sealed pre-registrations, and the
adversarial-review scripts that falsified my own headline (`redteam/`).
