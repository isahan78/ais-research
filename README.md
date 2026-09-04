# The Probe–Text Gap

Do linear probes on a reasoning model's activations beat cheap text-only readers
at predicting final-answer correctness from a truncated thinking trace? This is
a MATS 12.0 application project (Neel Nanda stream): Qwen3-8B on MMLU-Pro, every
number traced to a committed artifact.

## Final result (Run 012 — 1,000 traces, cross-fit, budget-matched)

**A linear probe on activations never beats a single bag-of-words reader of the
same trace text — at any cut, under a fair budget, with the length confound
removed.** Details in [RESULTS.md](RESULTS.md) Run 012; the analysis is
`cross_fit.py` / `cross_fit.json`.

- **The probe never wins.** With the probe given the same one-config budget as
  the text reader (a single a-priori layer, C tuned by CV) and predictions
  cross-fit over all 961 traces (213 negatives), Δ = AUC(probe) − AUC(TF-IDF) is
  −0.04 to −0.09 and **every one of the 11 cuts has a 95% CI excluding zero**
  (17 distinct comparisons counting a re-run on the fixed-length population).
  Probe 0.67–0.76; TF-IDF 0.74–0.82. Given a generous 280-config, 35-layer
  search the probe rises to ~0.77 and still loses.
  *Why a null is the default:* activations at a cut are a deterministic function
  of the prefix tokens, so the probe cannot hold more label-information than an
  ideal text reader; the live question is only whether internals are more
  *linearly accessible*, and here they are not.
- **The standard k% protocol leaks trace length.** Cutting at a fixed percentage
  makes prefix length very nearly a deterministic function of the trace's
  *eventual* length (measured correlation 0.99999) — a quantity no real-time
  monitor has. Length alone then scores ~0.61. The fix is fixed-token cuts; the
  argument is **realizability**, the same defect that made us withdraw our best
  early result.
- **…but the leak is small and does not explain text beating the probe.** On one
  fixed population, removing the leak drops a length-only reader from 0.56 to
  0.52 (chance) but moves TF-IDF only −0.02. An earlier underpowered pass
  reported a large "text collapse" under fixed-length cuts; that was a population
  + small-n artifact, corrected in the record.
- **Commitment precedes legibility.** The model's interrupted answer matches its
  final answer 57% → 97% across the trace (gold-free), yet no reader predicts
  *correctness* above ~0.8 at those cuts. You can tell **what** it will answer
  well before you can tell **whether it is right**.
- **A forced-answer baseline that reached 0.96 was withdrawn by our own
  red-team**: its score read the gold answer key — unusable by any deployed
  monitor. Same realizability rule that rejects fixed-percentage cuts.
- **All three sealed pre-registrations were wrong (0/3).** Each was committed to
  git before its dataset existed; that is what makes the record trustworthy.

## Read the write-up

- **[The_Probe_Text_Gap.docx](The_Probe_Text_Gap.docx)** — executive summary +
  full write-up, figures embedded (the submission document).
- [EXEC_SUMMARY_DRAFT.md](EXEC_SUMMARY_DRAFT.md) · [WRITEUP_BODY_DRAFT.md](WRITEUP_BODY_DRAFT.md)
  · [FORM_ANSWERS_DRAFT.md](FORM_ANSWERS_DRAFT.md) — the source drafts.
- [EXPERIMENT.md](EXPERIMENT.md) — design, decision log, the three sealed
  pre-registrations with outcomes.
- [RESULTS.md](RESULTS.md) — every run (001–012) with provenance.
- `outputs/expansion/figures/` — the four figures (regenerate with
  `make_figures_v2.py`).
- `redteam/` — the adversarial scripts that broke our own headline.

## Reproduce the analysis (CPU only, no GPU/network)

All final numbers recompute from committed artifacts under `outputs/expansion/`:

```bash
pip install -r requirements.txt
python cross_fit.py            # -> cross_fit.json (headline, budget-matched, cross-fit)
python compute_final_table.py  # -> final_table.json (single-split, 35-layer probe)
python make_figures_v2.py      # -> outputs/expansion/figures/*.png
pytest tests/ -q               # pipeline invariants (split-by-problem, truncation, AUC orientation)
```

## Reproduce the data pipeline (needs a GPU)

The traces and activations were generated on a rented 24 GB GPU (RTX 4090 class).

1. Rent a 24 GB box, CUDA 13 host + PyTorch, ≥60 GB disk (weights ~16 GB + HF
   cache). Note: torch built for cu13 fails on 12.4/12.8 driver hosts — check
   the host's CUDA version before deploying.
2. Clone this repo to the box, then:

   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   huggingface-cli download Qwen/Qwen3-8B     # optional pre-download
   ```

   vLLM pins its own torch; if pip reports a conflict, delete the `torch==` line
   from requirements.txt and let vLLM's pin win.
3. Sanity-check, then run the stages as **separate processes** (vLLM and HF each
   need the bf16 weights and cannot be co-resident on 24 GB):

   ```bash
   python -c "import torch; print(torch.cuda.get_device_name(0))"
   pytest tests/ -q                 # must pass BEFORE burning GPU time
   python -m experiment.generate_traces
   python -m experiment.truncate            # or truncate_abs.py for fixed-length cuts
   python -m experiment.harvest_activations
   python -m experiment.train_probe
   ```

   The grid runs via `run_grid.sh` / `run_abs_grid.sh`, which loop the pipeline
   once per cut into its own output dir using the `EXPERIMENT_*` env overrides in
   `config.py` (k%, fixed-N, layers, population, n_problems — all in the config
   hash). The expanded run used `pod_overnight.sh`.

## Notes on rigor (kept deliberately)

- **Never `hidden_states[-1]`** — it is post-final-RMSNorm; the raw residual is
  unrecoverable. Layer L means `hidden_states[L+1]`.
- **No quantization** (perturbs the residual stream being measured) and **no
  generation under forward hooks**.
- **Split by problem id**, identical across every reader and cut; ROC-AUC only,
  never accuracy; shuffled-label floor computed max-across-layers so the probe's
  layer selection is not compared against a floor that never selected anything.
- Every reported number carries the `config_hash` of the settings that produced
  it. The `tests/` suite enforces the invariants whose silent failure would
  fabricate a result.

## What is deliberately NOT here (future work)

A second model (to see whether the probe's steady loss to text generalizes), an
alignment-flavoured target rather than correctness (the claim that motivated the
flagship paper), and a richer probe (multi-layer / pooled) under the same
realizable protocol. By the accessibility argument above, a richer probe can at
best match, not exceed, what the text supports — worth testing.
