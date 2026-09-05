# Form answers, DRAFT v5 (Run 012 cross fit; voice: clear, direct, humble)

> Status: draft for the owner to rewrite in their own voice. Every number
> traces to RESULTS.md Run 012 / cross_fit.json. Qs 9 to 12 are OWNER ONLY and
> contain structure, not content. Neel explicitly rejects raw LLM output, so
> these drafts are the skeleton and the facts; the voice must be yours.
>
> Style note: no em dashes, en dashes, or compound hyphens in the answers. The
> only hyphens left are unavoidable proper names (TF-IDF, MMLU-Pro, MATH-500,
> Qwen3-8B) and the repo URL, plus the minus signs in negative numbers.

---

## Q2. What question did you try to answer?

Recent work reports that a linear probe on a reasoning model's activations
predicts, early in its chain of thought, whether its final answer will be
correct, and beats a reader of the trace text while doing it. I asked how much
of that *advantage over reading the text* survives a fair, realizable test: a
text baseline tuned as hard as the probe, a truncation protocol a real monitor
could actually run, and enough data to resolve the difference. Concretely:
Δ(k) = AUC(probe) − AUC(text reader) at cut points of Qwen3-8B thinking traces
on MMLU-Pro, with predictions registered in git before any data existed.

## Q3. Why is this question interesting / why did you choose it?

Published measurements point opposite ways: one line reports activation probes
beating text based classifiers at predicting a trace's outcome; another finds
most apparent internal signal disappears once a reader sees the same prefix
text. Which you get depends on how hard the text side is tuned and what the
protocol leaks, so the useful contribution is a careful measurement, not a new
method. It matters for a live safety question: whether CoT monitoring needs
white box access or whether reading the trace is enough. And it fits a 16 hour
budget honestly: one model, one dataset, a registerable headline number, every
part cheap to test and able to fail.

## Q4. What conclusions have you reached?

From 1,000 traces (961 usable, 213 negatives), cross fitted over all of them:

1. **A budget matched linear probe never beats a single bag of words reader of
   the trace text. All 11 cuts, every 95% interval excludes zero (17 distinct
   comparisons counting the population control rerun).**
   Across 6 k% cuts and 5 fixed length cuts, Δ = AUC(probe) − AUC(TF-IDF) is
   −0.04 to −0.09. The probe scores 0.67 to 0.76, the text reader 0.74 to 0.82.
   To keep it fair, the probe gets one config (a single fixed layer chosen in
   advance), the same budget as the one fixed TF-IDF. Given a generous 280
   config search the probe rises to ~0.77 but still loses everywhere. A null is
   the default for a reason. Activations at the cut are a deterministic function
   of the prefix tokens, so the probe cannot carry more label information than an
   ideal text reader. The live question is only whether internals are more
   *linearly accessible*, and here they are not.
2. **The standard k% of trace protocol leaks trace length.** Cutting at a fixed
   fraction makes prefix length very nearly a deterministic function of the
   trace's *eventual* length (measured correlation 0.99999), information no real
   time monitor has. Length alone then predicts correctness at ~0.61. The fix is
   fixed token cuts; the argument for it is realizability, not effect size
   (see Q6).
3. **But the leak is small and does not explain text beating the probe.** On one
   fixed population, removing the leak drops a length only reader from 0.56 to
   0.52 (chance) but moves TF-IDF by only −0.02. An earlier underpowered pass of
   mine reported a large "text collapse" under fixed length cuts; that was an
   artifact of the smaller sample and the harder population, which I corrected
   (Q7).
4. **The model commits to its answer long before its correctness is legible.**
   A forced mid trace answer matches the eventual answer 57% → 86% → 97% across
   cuts (gold free, full set), yet no reader predicts *correctness* above ~0.8
   at those cuts. You can tell what the model will say well before you can tell
   whether it is right. This is the most robust result: no probe, no tuning.
5. My most exciting result was wrong. A "forced answer" baseline reached 0.96
   AUC, but its score was computed against the answer key (see Q6). The numbers
   above are the ones that survived the review that caught it.

## Q5. Technical setup

- **Quantities:** (a) early predictability of final answer correctness from
  internals versus from trace text, held out AUC of each reader at each cut,
  never accuracy (base rate 78%); (b) answer commitment, whether the forced mid
  trace answer equals the eventual answer (gold free).
- **Model:** Qwen/Qwen3-8B, thinking mode (16,384 token thinking budget, chosen
  after measuring that an 8k cap truncated 30% of traces, deleting
  disproportionately *incorrect* ones: survivor bias).
- **Data:** 1,000 MMLU-Pro questions (chosen after *measuring* base rates:
  Qwen3-8B is 89% accurate on MATH-500 level 5, leaving too few negatives;
  MMLU-Pro gave 22% error = 213 negatives). 961 usable traces.
- **Prompts:** each MMLU-Pro item is rendered as a 10 option MCQ (labels A to J)
  ending in a fixed instruction, *"Reason step by step, then give your final
  answer as a single letter in \boxed{}"*, run through Qwen3's chat template with
  thinking mode ON. The gold free confidence monitor uses a forced answer
  prompt: `prefix + </think> + "Based only on the reasoning above, my single best
  answer is \boxed{"`, and scores the model's probability on the next (answer)
  token, with no gold ever read.
- **Cuts:** k ∈ {1, 10, 25, 50, 75, 90}% of thinking tokens, plus a second grid
  at fixed N ∈ {64…1024} tokens on a fixed 781 trace population (traces ≥1,024
  thinking tokens, 196 negatives).
- **Fair budget:** the text reader is one fixed TF-IDF config; the probe is
  matched to one config (a single layer chosen in advance, residual stream at the
  last prefix token, C by inner CV). The generous 35 layer search is reported as
  a labelled upper bound. Δ = probe − TF-IDF, no per cut max on the text side.
- **Cross fit:** out of fold predictions over all 961/781 traces (fivefold
  StratifiedGroupKFold, split by problem id), so every negative is test data
  once; pooled and per fold AUC agree. Δ CI = cluster bootstrap over problems.
- **Population control:** the k% grid rerun on the same 781 traces used for fixed
  length, isolating cut geometry from population.
- **Other readers:** prefix length only (prices the leak); frontier LLM judge
  (Claude Opus 5, original 289 trace subset only; reported separately, Q6); gold
  free forced answer confidence monitor.
- **Controls:** k=1 near zero cut; shuffled label floor (500 seeds, max across
  layers); three registered predictions sealed in git.

## Q6. Strongest evidence AGAINST these hypotheses

The strongest evidence against my hypotheses came from my own results failing
review. Three cases:

1. My headline result was false. The forced answer baseline (interrupt the
   model, make it commit) reached 0.96 AUC and beat everything. An adversarial
   review of my own pipeline showed its score was computed against the gold
   answer: whenever the interrupted answer equalled the final answer, the score
   equalled the label by construction (97% of rows at k=90). On the remaining
   rows its AUC was 0.000. The apparent "gap widens with k" finding was just the
   copy rate approaching 1. I withdrew it. No deployed monitor could compute this
   score.
2. **Against my own leak story:** I first reported that removing the leak made
   the text reader "collapse", evidence the leak was doing text's work. A cross
   fit on a fixed population showed I was wrong: the leak is small (+0.04 AUC to
   a length only reader, and does not move TF-IDF), and the apparent collapse was
   a population artifact. Corrected in the record.
3. **Against the probe:** it never leads, even given a 280 config search it never
   got matched on the text side. At the k=1 control it scores 0.69, so little of
   what it reads this early is reasoning. The frontier judge (Claude Opus 5,
   original 289 trace subset, k1/k25) scored 0.959 at k=25 but 0.876 at k=1,
   where there is almost no reasoning to read. It mostly knows this public
   benchmark, so I reported it separately as a difficulty oracle rather than pool
   the most flattering number. I did not spend to expand it, since it never
   enters Δ.

All three of my registered predictions were also wrong. I predicted Δ ≈ +0.10 at
k=50; it came in negative. My third registration (before the full power run)
predicted the probe would stay near 0.68 and that fixed length cuts would put it
ahead. Instead it rose (to ~0.77 with a full search) and still lost at every cut.
My record on sealed forecasts is 0 for 3, and that each was committed before its
dataset existed is what makes the record worth anything. All of it is timestamped
in the repo.

## Q7. Biggest limitations, and could I have addressed them?

1. **The probe's degrees of freedom.** The fair probe is one layer chosen in
   advance; the generous one searches 35. Both lose, but a richer probe (multiple
   layers concatenated, attention over the prefix) might close some of the gap,
   though by the accessibility argument it can only ever match, not exceed, what
   the text supports. I have not tested a richer probe.
2. **Power, now largely handled.** After a first reviewer flagged the pilot as
   underpowered I quadrupled the data (52 → 213 negatives) and cross fit over all
   of it. That resolves every one of the 17 distinct Δ comparisons, and all CIs
   exclude zero. It is still a few hundred failures on one benchmark, so the
   intervals are tight, not zero width.
3. **One model, one dataset; correctness, not alignment.** The flagship paper
   predicts *misalignment* of the response, on a different model; I tested the
   correctness variant. My results speak to the protocol and the genre, not to
   that paper's specific claim. A second model was a deliberate scope cut to fit
   the time budget, addressable with more compute rather than more cleverness.
4. **The judge comparison is bounded by memorization:** on a public benchmark a
   frontier model's "prediction" is partly recall. Controlled for (k=1) but not
   removable without private items.

## Q8. How I used LLMs, and how I made sure it wasn't slop

[OWNER: this must be in your voice and match what you actually did. The facts,
from the project record:]

- **Tools:** Claude Code (Opus 5 / Fable 5) as the primary research assistant:
  pipeline code, experiment orchestration on rented GPUs, literature sweeps;
  Claude Opus 5 via API as the judge baseline (an experimental subject, not an
  assistant).
- **Verification, in order of rigor:**
  (a) 441 passing CPU only tests (461 collected, 20 skipped; no GPU/network),
  enforcing the invariants whose silent failure would fabricate results (split by
  problem, truncation boundaries, AUC orientation);
  (b) mutation testing: I deliberately broke the floor computation, the layer
  indexing, and the verdict logic, and checked that the test suite fails. An
  earlier version of the suite passed all three mutations, which meant it was not
  testing the decision logic at all, so it was rebuilt.
  (c) an adversarial review by separate agent instances instructed to break the
  result. They did (Q6.1). This was the most valuable step in the project.
  (d) hand verification: randomly sampled traces, prefixes and forced answers
  read by eye at every stage; grader spot checked against 40 traces.
- **What I checked versus didn't:** every reported number traces to a committed
  artifact. The final table cross checks two independent code paths: pooled and
  per fold AUC agree, and the cross fit reproduces the single split direction. I
  did NOT hand verify all 1,000 traces (10 sample spot checks per stage), and
  single generation labels mean a trace's correctness is one sample of a
  stochastic model. **Where a major error would least surprise me:** the choice
  of a single fixed probe layer (I mitigate by also reporting the 35 layer
  search), and any systematic bias in TF-IDF features I did not think to probe.

## Q9. Prior experience with mech interp

More than none, though mostly on the engineering side, and I would rather be
exact about that than round it up. Before this project I built GlassBox, an
interpretability platform on top of Neel's TransformerLens. It implements the
standard toolkit end to end: activation tracing, activation patching, causal
tracing, circuit discovery, and sparse autoencoder feature extraction, wrapped in
a tested library with a REST API and a dashboard. It reimplements published
methods (patching from the ROME work, Anthropic's circuits framework, the
monosemanticity SAEs) rather than producing a new finding, and it runs at GPT-2
and Llama 2 scale. So I arrived fluent with hooks, activations, patching, and
SAEs as tools.

What this MATS project added was the science rather than the plumbing: picking a
real question, registering predictions before the data existed, racing a probe
against honestly tuned baselines, and withdrawing my own headline when it turned
out to read the answer key. GlassBox taught me how to instrument a model. This
taught me to be skeptical of what the instruments say.

## Q10. 1 to 3 pieces of evidence you'd do good research (~100 words)

1. I taught myself mechanistic interpretability by building in it, on my own.
   GlassBox reimplements the standard toolkit (patching, causal tracing, circuit
   discovery, SAE features) on TransformerLens as a tested library with a REST
   API and a dashboard. It is evidence I ramp into a field fast and build the
   tooling to work in it.
2. I also learned to distrust my own claims. I built AgentSentinel to trace risky
   outputs to "dangerous circuits", then had to admit the circuits were a
   hardcoded list of attention heads and the trigger was keyword matching. It was
   attribution against an assumption, not the mechanistic detection I had
   labelled it. I would rather name that than ship the bigger claim, and it is
   why this project was built around baselines a monitor could actually run.

## Q11. Why Neel's stream

Because the way he says research should be done is the way this project actually
went, and because the direction he is pushing is the one I want to work in. Four
specifics.

First, the move toward practical interpretability. I am a builder at heart. I
would rather build and experiment my way to a finding that is actually useful
than polish theory, and that is the exact turn his stream is taking. GlassBox and
this project are both that instinct: instrument a model, run the experiment, keep
what survives.

Second, his advice to start with the obvious thing, a prompt or a linear probe
before an SAE. That is exactly what made this project tractable in the time I
had; the linear probe was the right first tool and it was enough to reach a real
answer.

Third, his line that most results are false, especially the exciting ones. My
most exciting result was false, and I only caught it by attacking my own
pipeline. Getting better at that, in a stream that selects for it, is what I
want.

Fourth, I have already spent real time inside his TransformerLens, building
GlassBox on top of it, so his tooling and framing are literally how I learned to
instrument a model. I am not interested in interpretability as decoration; I want
to do useful things with it and be told plainly when they are not useful.

## Q12. Likelihood you'd join Sept 28 to Oct 30

Very likely. I have no conflicts across Sept 28 to Oct 30 and would treat the
program as my primary commitment for that window.

## Q13. (Optional) Anything else

Two things a reader can verify independently: (1) all three sets of predictions
were committed to git *before* the corresponding data existed (commits
`2446d69`, `82c4f99`, and registration III in EXPERIMENT.md). The timestamps and
all three refutations are in history; my forecasts went 0 for 3; (2) the
adversarial review scripts that falsified my own headline are preserved verbatim
in the repo (`redteam/`), alongside every run's raw artifacts. Repo:
https://github.com/isahan78/ais-research (public).
