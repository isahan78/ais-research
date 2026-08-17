# Experimental Plan — MATS 12.0 Application

**Deadline:** Fri Sept 4 2026, 11:59pm PT (extension available to Sept 11 — insurance, not the plan).
**Budget:** 16 hours of research (hard cap 20) + 2 hours for the executive summary and application form.
**Written:** 2026-08-16. 19 days out.

---

## 1. The question

> For published claims of the form *"reasoning models internally know their outcome early in the trace,"* what fraction of the predictive signal is available from the trace **text alone** — and is there any regime where activations genuinely beat the strongest text-only predictor?

**Why this question.** A cluster of 2025–26 papers reports that linear probes on mid-trace activations predict a reasoning model's final answer or correctness well before it is stated. Separately, [an activation-oracle study](https://www.lesswrong.com/posts/LXQBcztrWKhtcgQfJ/current-activation-oracles-are-hard-to-use) found that after filtering cases an LLM could answer **from the preceding text alone**, roughly 95% of the apparent signal evaporated (~100 of 2,300 cases survived). If that confound generalises, a chunk of the early-prediction literature is measuring text, not cognition.

**Why it's a good application.** Neel's own writing pre-endorses the genre — *"rigorous, at-scale replications of shaky results, negative results of seemingly promising hypotheses, and high-quality failed replications of popular papers are all very valuable contributions. I would personally consider these novel."* And his most-repeated complaint about the field is *"I see a simple and boring explanation for the author's observations, and they didn't test for it."* This project is that complaint, operationalised.

**All three outcomes are reportable** — the project cannot leave you empty-handed:
1. Effect doesn't reproduce → high-quality failed replication.
2. Reproduces, text baselines match the probe → a leakage calibration the field needs.
3. Probe beats text in some regime → positive result: precisely what internals add, and where.

---

## 2. Calendar

19 days, 16 hours. **Do not** slice it into daily 45-minute fragments — that destroys feedback loops, which Neel names as the single biggest lever on research velocity.

| Block | When | Hours | Content |
|---|---|---|---|
| Block 1 | this week | ~4h | Hour 0 lit re-check + **Gate 1** + Explore |
| Block 2 | this weekend | ~4h | Gate 2 hypothesis lock → Understand |
| Block 3 | next week | ~4h | Understand: baselines + controls |
| Block 4 | by **Aug 31** | ~4h | Gate 3 → Distill, figures, red-team |
| Write-up | Sept 1–4 | +2h | Exec summary, form questions, one cooling-off read |

**Research finishes Aug 31.** Sept 1–4 is writing and a cold re-read, nothing else.

---

## 3. Getting a GPU (Runpod)

You need **one 24 GB card for roughly 90 minutes** to clear Gate 1. Total cost is about a dollar. Do not overthink this step — it is the cheapest part of the project.

### Procurement, step by step

1. **Account + credits.** Sign up at runpod.io, add **$25**. That covers Gate 1 many times over and every later block; you will not spend it all.
2. **Deploy → Pods → GPU Cloud.** Filter to **RTX 4090 (24 GB)**. Choose **Community Cloud** — materially cheaper than Secure Cloud, and this workload has no security or uptime requirement. Expect roughly **$0.35–0.60/hr**; verify live, prices drift.
3. **Template:** any **RunPod PyTorch (CUDA 12.x)** image. Do not hunt for a vLLM-specific template — `requirements.txt` installs it.
4. **Disk:** container disk 20 GB, **volume disk 60 GB**. Weights are ~16 GB plus HF cache; the default 10–20 GB will fail mid-download, which is the single most common way this step wastes an hour.
5. **Connect:** SSH from your terminal (Runpod gives you the command), or the web terminal for a first look. SSH is better — you want to `scp` results back.
6. **Optional but smart if you'll run more than one block:** attach a **Network Volume** and put the HF cache on it. Weights survive pod teardown, so later blocks skip a 15-minute download.

### The money trap

**"Stop" is not "Terminate."** A stopped pod still bills for storage. When a block is finished and you have copied results off the box: **Terminate**. Attach a network volume if you need persistence — that is the cheap way to keep state, not a parked pod.

### No auth needed for Gate 1

`Qwen/Qwen3-8B` and `HuggingFaceH4/MATH-500` are both ungated — no HuggingFace token required. (GPQA-Diamond *is* gated, but it belongs to deferred work, not this gate.)

### On the box

```bash
git clone https://github.com/isahan78/ais-research.git && cd ais-research
python -m venv .venv && source .venv/bin/activate
pip install -r experiment/requirements.txt
huggingface-cli download Qwen/Qwen3-8B      # start now, read the next section while it runs

pytest experiment/tests/ -v                  # MUST pass before you burn GPU time
python -c "import torch; print(torch.cuda.get_device_name(0))"

bash experiment/smoke_test.sh                # Gate 1
```

Full setup detail and the per-stage hand-verification checklist live in [README.md](README.md).

---

## 4. The gates

Each is a **stop-and-decide**, not a milestone to coast past.

### Gate 1 — de-risk (hours 0–2) ← you are here

- **0:00–0:30, no GPU:** arXiv/Google for anything from the last 8 weeks that already audits early-prediction claims against text baselines. Then read the methods sections of the two target papers: **do they run a text-only baseline, and which?** If a thorough audit already exists, stop and re-plan — that is a cheap kill, not a failure.
- **0:30–2:00, on the box:** `bash experiment/smoke_test.sh`.
- **GO:** pipeline completes and best-layer AUC clears the shuffled-label floor. Commit the remaining 14 hours.
- **NO-GO paths:** too slow → prototype on R1-Distill-1.5B, scale up only for finals. No signal at all → the project becomes outcome 1 (failed replication), which is still the project.

### Gate 2 — hypothesis lock (hour 6)

Write `predictions.md` **before** the main runs: your predicted `S_probe` and `S_text` at each k, per claim. A confirmed prediction is worth strictly more than post-hoc analysis, and you can only prove which you did if you wrote it down first.

### Gate 3 — evidence audit (hour 12)

Re-derive the headline number by a second route. Draft the exec summary *now*, while four hours remain — it will be wrong, and the way it's wrong tells you which experiment is missing while you can still run it.

---

## 5. What gets built after Gate 1

Deferred by design (see `_bmad-output/implementation-artifacts/deferred-work.md`) — none of it is worth building until Gate 1 says go:

- Full truncation grid k ∈ {0, 10, 25, 50, 75, 90}%
- Second dataset (MCQ: GPQA-Diamond or MMLU-Pro)
- **The three text-only baselines** — the actual scientific content:
  1. **Frontier LLM judge** — predict the outcome from the identical prefix (API, ~$20)
  2. **Trained text classifier** — same train split as the probe, tuned hard
  3. **Forced-answer** — the subject model itself, truncated at k, forced to answer immediately. On-policy, nearly free, and the baseline most papers skip.
- **Δ(k) = S_probe(k) − max(S_text(k))** with bootstrap CIs. The Δ(k) curve is Figure 1.

**Make the baselines strong.** Neel: *"there's a natural bias to invest more effort in making one's cool, shiny new technique look good than in optimizing boring baselines. Resist this."* Here the baseline **is** the result — a weak one invalidates the whole project.

---

## 6. Standing rules

**The agent-sanity rule.** For every number that reaches the write-up, you can point at the line of code that produced it, and you have hand-checked 10 randomly drawn examples behind it. Neel's #1 listed disqualifier is unverified agent output. This is the cheapest place in the entire application to beat most applicants.

**Six-gate skepticism**, applied to every headline result: random floor · strong competitor baseline · the boring explanation named in advance and tested · second route · randomly-sampled qualitative examples · noise floor with CIs.

**Ask at every gate:** *am I getting enough information per unit time?* Two hours with nothing learned is a signal to change what you're doing, not to push harder.

---

## 7. Budget

| Item | Cost |
|---|---|
| Gate 1 (~1.5 GPU-hr) | ~$1 |
| Blocks 2–4 GPU | ~$25 |
| API credits (LLM-judge baseline) | ~$20 |
| Buffer (the pod you forget to terminate) | ~$25 |
| **Total** | **~$70 of the $100 authorized** |
