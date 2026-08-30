# The Deliverable — plan of record

**What this document is:** how the three submitted artifacts get written, in what order, and what goes in each. The science lives in [EXPERIMENT.md](EXPERIMENT.md), the build in [ENGINEERING.md](ENGINEERING.md), the numbers in [RESULTS.md](RESULTS.md). This file is the only place the *communication* decisions live.
**Status:** written 2026-08-29, **before Block 2 data exists** — deliberately. The narrative is built with blanks so that finishing it tells us which remaining runs are load-bearing.
**Due:** Fri Sept 4, 11:59pm PT · extension to Sept 11 available (gate in §10).

---

## 1. There are three deliverables, and the priority is inverted

Neel, verbatim: *"The application form has a bunch of Qs about the project and I will read these for every single app and use it as a preliminary filter - communicating well here is important and should be prioritized!"* and *"I don't have time to read every write-up."*

| Artifact | When he reads it | Length | Share of the 2h+ writing effort |
|---|---|---|---|
| **Form questions** | Always, first, as a filter | Whatever the form allows | **35%** |
| **Executive summary** | If the form Qs earn it | ~1 page / ~500 words | **30%** |
| **Write-up** | Only if the first two earn it | ~2,000–3,000 words + 3 figures | **35%** |

The write-up is the *least*-read artifact and gets a third of the effort. This is not laziness — it is Neel's own paper-writing advice applied honestly: *"you should spend about the same amount of time on each of: the abstract, the intro, the figures, and everything else."* Here the form Qs are the abstract.

**Corollary:** the form questions get written **first**, from the compressed narrative in §2 — not last, from a finished write-up. Writing them first is also the cheapest test of whether the narrative actually holds.

---

## 2. The compressed narrative (Neel's step 1: compress, then iteratively expand)

### One line

> Several recent papers report that linear probes read a reasoning model's eventual success out of its mid-trace activations. We raced those probes against the strongest text-only readers we could build — including one nobody has reported: **interrupting the model mid-thought and forcing it to answer** — and measured where, along the trace, internals actually add anything.

### The three claims, with blanks

Blanks are filled from RESULTS.md. Every blank has a pre-registered forecast (EXPERIMENT.md §12), so filling them is also scoring the bet.

**Claim 1 — the free baseline.** *Forcing the model to answer from its own truncated trace is a strong outcome predictor: AUC `[__]` at k=50, versus `[__]` for the activation probe.*
- Why it leads: it is on-policy, needs no labels, no activations, no training, and appears in none of the three audited papers. If it matches the probe, the takeaway is transferable and blunt — **you did not need the probe, you could just ask.** If it does not match, that gap is a sharper model-biology claim than "text vs internals": the model knows something it cannot state even when made to commit.
- This is the claim most likely to *teach Neel something he'd use*, which is his stated bar.

**Claim 2 — the gap, measured.** *Δ(k) = S_probe(k) − max(S_text(k)) is `[__]` across k ∈ {10,25,50,75,90}%, with bootstrap CI `[__]`.*
- Read horizontally, this is the quotable version: *"the probe at 25% of the trace knows what the best text reader only knows at 75%"* — or its null, *"the two track each other within noise for the whole trace."*
- This is the number two published camps disagree about (EXPERIMENT.md §2), measured on a domain neither covered, with predictions sealed beforehand.

**Claim 3 — the adjudication.** *On the cases where the best text reader is confidently wrong, Δ is `[__]`.*
- Camp Internals predicts a gap concentrated here; Camp Text predicts none anywhere. Costs no new data. n will be small — say so and hedge to match.

### What survives whichever way the numbers fall

The contribution is **the race and where the line sits**, not the direction. Written this way the narrative does not depend on the result — which is the only honest way to have written it before the result existed.

| If | Headline becomes | Standard of evidence |
|---|---|---|
| Δ CI overlaps 0 everywhere | "The probe's early-prediction advantage does not survive strong text baselines in this setting" — plus the free-baseline result | *Practitioners should take care when…* — probe papers here should report a forced-answer baseline |
| Δ clearly > 0 at some k | "Internals lead the written trace by ~`[__]`% of the trace, concentrated in `[__]`" | *Provides compelling evidence that…* — and Tyler's forecast was wrong; the write-up says so |
| S_probe ≈ 0.5 | High-quality failed replication on a new domain | Nanda, verbatim: replications and negative results *"are all very valuable contributions"* |
| **Both ≈ ceiling already at k=10** | The predictable thing is the *question*, not the reasoning — see the k=0 control in §10 | Would demote Claims 1–2; hence the control |

---

## 3. Claims ledger

| # | Claim | Key evidence | What would break it | Status |
|---|---|---|---|---|
| 1 | Forced-answer is a strong / weak outcome predictor | AUC vs probe at each k, same split | Forced-answer contaminated by the answer already being written in the prefix — check prefixes for a stated answer | pending Block 2 |
| 2 | Δ(k) is ≈0 / >0 | Δ curve + bootstrap CI + shuffled-label floor | Probe reading trace *length* not content; length-matched subset control | pending Block 2 |
| 3 | Gap concentrates on performative traces | Split test set by text-reader confidence × correctness | n too small; report CI and say so | pending Block 2 |
| 0 | Everything is predictable from the question alone | k=0 control | If AUC(k=0) ≈ AUC(k=50), Claims 1–2 are about difficulty, not reasoning | **not yet run — §10** |

---

## 4. Figures — three, planned now, built once data lands

Figures get the same effort as the abstract. Plan them before the data so the runs produce what the figures need.

1. **The race.** x = % of thinking trace, y = ROC-AUC. Two bold lines (probe, best text) + forced-answer + the shuffled-label floor as a grey band. Anchored at k=0. **This figure is the whole paper.** If a reader sees only one thing, this is it.
2. **The gap.** Δ(k) with bootstrap CI, zero line drawn. Redundant with fig 1 by construction, and worth it: it puts the CI-vs-zero question in front of the eye.
3. **The adjudication.** Δ split by whether the text reader was right / confidently wrong. Bar chart with CIs, n printed on each bar.

Rules: no figure without axis labels, n, and what the error bars are. No figure that needs the caption to be understood.

---

## 5. Executive summary — sentence-by-sentence skeleton

Built on Neel's own abstract recipe (corpus, *Highly Opinionated Advice on How to Write ML Papers*). ~500 words. Each bullet is one sentence.

1. **Situate (uncontroversial):** reasoning models are state of the art, and recent work reports that linear probes predict a model's eventual correctness from mid-trace activations.
2. **The need:** two published positions now disagree about whether those probes read internal state or re-read the text — and nobody has raced them against the strongest available text reader.
3. **Contribution:** we ran that race on `[n]` MMLU-Pro problems with Qwen3-8B, at five truncation points, with predictions sealed in advance.
4. **Clarify:** both racers see byte-identical prefixes; the text side is scored as the *max* over four readers, including a frontier LLM judge, a tuned classifier, and the model itself interrupted and forced to answer.
5. **Headline result, with a number:** `[__]`.
6. **Second result:** `[__]` (the forced-answer comparison, if not already the headline).
7. **The skeptical control that most changes the reading:** `[__]` (floor / length control / k=0).
8. **Limitations in one sentence:** one model, one domain, n=`[__]`, linear probes only.
9. **Standard of evidence + why it matters:** CoT monitoring is a live safety proposal; this says `[__]` about when reading the text is enough.

**Test before shipping:** hand this to someone with no context. If they cannot say back what was measured and what came out, it fails and gets rewritten. Not "does it sound impressive" — *does the point survive transmission.*

---

## 6. Write-up skeleton

~2,000–3,000 words. Section budget in words, so nothing sprawls.

| § | Content | Words |
|---|---|---|
| Title + one-paragraph summary | The narrative, once, plainly | 150 |
| 1. The question and why it's live | Two camps, both cited; why math/knowledge correctness is the untested domain; why this bears on CoT monitoring | 350 |
| 2. Setup | Model, dataset, cuts, the two views, the split-by-problem rule and why it matters | 400 |
| 3. The racers | Four text readers, why each is tuned hard rather than strawmanned, forced-answer explained properly | 350 |
| 4. Results | Figures 1–3 with the numbers in prose beside them | 600 |
| 5. **Did I believe it? — controls and red-teaming** | Floor, length control, k=0, second-route re-derivation, what I checked by hand (§8) | 500 |
| 6. What I got wrong | Pre-registration scored openly; MATH-500 killed by measurement (Runs 001–004); the base-rate lesson | 300 |
| 7. Limitations and what I'd do next | Honest, specific, short | 250 |
| Appendix | Config hash, repo link, run log pointer | — |

**§5 and §6 are the differentiators.** Most applicants submit a clean linear story with a results section. Neel's stated top disqualifiers are unverified agent output and insufficient skepticism; §5 and §6 answer both directly, and they are already paid for — the controls are built and the pivot is documented.

**No related-work section.** Citations go inline in §1. A literature review would spend words the reader will not reach.

---

## 7. Form questions — BLOCKED

Our copy of the application doc (`neel-nanda-mats-application.md`) lists the *Advice on producing a good application*, *What does a good application look like*, and *Recommended research problems* tabs in its table of contents but does not contain their bodies, and we do not have the actual form question text.

**Action:** pull the live form and paste the questions in here. These are the highest-leverage words in the entire application and they cannot be drafted blind. Also worth pulling: the linked *examples of successful past write-ups*, which calibrate length and tone better than any advice can.

Draft answers here once the questions are known. Default shape for each: **claim → the single strongest piece of evidence → the limitation**, in that order, in that number of sentences.

---

## 8. The sanity-check section (spec)

Neel's #1 listed disqualifier: *"if your write-up contains key results you clearly never verified, or don't understand, that's disqualifying. I want scholars with value add over prompting Claude myself."*

So the write-up states, concretely and briefly:

- **Every number traces to code.** Config hash + commit in the appendix; RESULTS.md is append-only and public with the repo.
- **What was hand-checked, with a real example.** Not "we verified outputs" — one named example: *e.g.* the grader was hand-checked on 40 traces and found 1 genuinely ungradeable (Run 002); truncation prefixes were read to confirm the answer is not already stated. Pick the checks that actually happened and name their n.
- **What the tests catch.** 264 CPU tests, three of which were written by mutating the code until the suite failed — the un-shuffled floor, the off-by-one layer index, and an inverted GO/NO-GO all passed the *first* suite. Say this. It is the most credible sentence available about agent oversight, and it is true.
- **Where the agent was wrong, and how it was caught.** Runs 002–004: a plausible fix was applied without measuring, and measurement refuted it. One sentence, no self-flagellation.

**Judgment call, flagged for the owner:** the pre-registration (EXPERIMENT.md §12) records two independent forecasts — the owner's and the coding agent's — sealed before Block 2, and they disagree on the crux. Reporting that disagreement is unusually strong evidence of real skepticism, and Neel explicitly wants applicants to use LLMs and say how. Risk: it can read as leaning on the agent for research judgment. **Recommendation: include it, in one sentence, framed as the owner's prediction versus a deliberately-adversarial second forecast, with the research decisions plainly the owner's.** Owner's call.

---

## 9. What we deliberately do not write

- No related-work section (§6).
- No appendix beyond provenance — appendices are read by nobody on a 6-day clock.
- No second domain, no second model, no nonlinear probes (scope refusals already logged 2026-08-25); the write-up names them as future work in one line each, and does not apologise for them.
- No verb stronger than the CI supports. EXPERIMENT.md §9 is binding on the prose: *"suggestive evidence that…"*, *"in our setting…"*, *"we found at least one regime where…"*.
- No raw LLM prose. Draft with an agent if useful; every sentence gets rewritten by hand. Neel: *"it's obvious, unpleasant to read, tends to be vague, and harms your application."*

---

## 10. Schedule, and the one experiment to add

**Six days. Block 2 has not run. The text-baseline code (`llm_judge.py`, `forced_answer.py`, `text_classifier.py`, `analysis.py`) is written but uncommitted and has never executed on hardware — and Run 001 established that the GPU box finds bugs macOS does not.**

| Date | Work | Gate |
|---|---|---|
| **Aug 30** | Commit the baseline code. Full-pipeline smoke on ~20 problems **including all four text readers** end to end. Fix what hardware breaks. | Do all four readers produce a number on real data? |
| **Aug 31** | Block 2: n=300, five cuts (+ k=0), all readers. Copy `outputs/` off the box. **Terminate the pod.** | Data in hand |
| **Sept 1** | Analysis. Δ curve, CIs, floor, length control. Re-derive the headline by a second route. Draft figures. | Gate 3: what can actually be claimed? |
| **Sept 2** | Fill §2 blanks. Draft form answers and exec summary from them. | **Extension gate — see below** |
| **Sept 3** | Write-up, first full draft. | |
| **Sept 4** | Edit, figures to final, cold read, flip repo public, submit. | |

**Extension gate, decided Sept 2 (not later):** if Block 2 data is not analysed by end of Sept 2, take the Sept 11 extension. The extension is stated as freely available and carries no penalty in the doc; a rushed exec summary and rushed form answers do carry one, because they are the artifacts Neel actually reads. Compressing the writing to protect a date is the single worst trade available here.

### The one experiment to add: k = 0

Add a **k=0 truncation point — the prompt alone, before any thinking token is generated.**

Why it is not optional: if the probe scores 0.72 at k=10 but *also* scores 0.70 at k=0, then the curve is measuring how hard the question is, not what the reasoning did — and Claims 1 and 2 are about problem difficulty. That confound is currently untested and it is the first thing a skeptical reader will ask. It is also precisely what *No Answer Needed* (2509.10625) reports, so a real value there is expected, not hypothetical.

Cost: one more k in the existing grid. Traces already exist; activations come from the same prefill pass; the LLM judge reads the question alone. Effectively free. **Run it.**

### Also fix before Block 2

- **EXPERIMENT.md §4 row 5 says the split is "~40/20"** — stale from the n=60 design. Current config is `n_problems=300`, `test_fraction=0.35` ⇒ ~195/105 by problem. Correct it now, before it becomes a wrong number in a submitted document.

---

## 11. Decision Log (append-only)

- **2026-08-29 — Narrative order set: forced-answer promoted to Claim 1.** Recorded *before* Block 2 data exists. This is a reporting decision, not a scoring change: Δ remains the max over all text readers exactly as sealed in EXPERIMENT.md §12, and Δ_forced = S_probe − S_forced is reported alongside it as a named secondary quantity. Rationale: the forced-answer race is the transferable, on-policy result that is absent from all three audited papers, and it is informative in both directions, whereas Δ's interest depends on its sign.
- **2026-08-29 — k=0 control added** to the truncation grid as a required baseline (§10). Without it the Δ curve cannot be distinguished from a difficulty curve.
- **2026-08-29 — Write-up effort allocated 35/30/35** across form questions / exec summary / write-up, against read-order rather than length.
