# Engineering & Operations — the build PRD

**What this document is:** how the experiment gets built, run, and paid for. The science lives in [EXPERIMENT.md](EXPERIMENT.md); on-box commands and hand-verification checklists live in [README.md](README.md). Each fact lives in exactly one place.
**Status:** Gate-1 pipeline built and adversarially reviewed 2026-08-23; **rebuild queue below is open — do not run the pipeline until it clears.**

---

## 1. Pipeline architecture

Four stages, plain scripts, one config (`config.py`), JSONL/npz artifacts between stages. Every artifact carries the config hash and the input file that produced it (lineage).

```
Stage 1  generate_traces.py   vLLM, Qwen3-8B thinking  -> outputs/traces.jsonl
Stage 2  truncate.py          cut at k% of thinking    -> outputs/prefixes.jsonl
Stage 3  harvest_activations. HF prefill-only, bf16    -> outputs/acts.npz
Stage 4  train_probe.py       probe + floors + verdict -> outputs/results.json
```

Text-baseline stages (LLM judge, text classifier, forced-answer) and Δ(k) analysis are the post-Gate-1 build (tracked in `_bmad-output/implementation-artifacts/deferred-work.md`).

## 2. Correctness invariants (MUST hold; violating any silently fabricates results)

1. Train/test split by problem id — asserted in code, not just tested.
2. ROC-AUC only.
3. One rendered prompt string shared byte-identically by vLLM and HF; stage 3 asserts prompt-token prefix match.
4. vLLM and HF run as sequential processes with engine teardown (2 × 15.26 GiB > 24 GiB).
5. Never `hidden_states[-1]` (post-final-RMSNorm; raw layer-35 residual unrecoverable). `[L+1]` = after layer L.
6. `AutoModel` (Qwen3Model), never `*ForCausalLM` (~2.3 GiB of pointless logits).
7. No quantization (perturbs the residual stream being measured). No generation under hooks.
8. Correctness guards are `raise`, not bare `assert` (survive `python -O`).

## 3. Build status & rebuild queue (from the 2026-08-23 adversarial review)

**Exists:** 4 stages + grading + config + smoke_test.sh + 142 CPU-only tests + README runbook. Repo: https://github.com/isahan78/ais-research (private).

**Rebuild queue — blocking, in order:**
1. **Tests that execute the decision logic** (mutation review proved: un-shuffled floor, off-by-one layer index, inverted GO/NO-GO all pass the current suite): known-answer floor test (noise→~0.5, separable→high), `residual_index()`/verdict extracted to pure functions and table-tested, stage round-trip tests over synthetic JSONL (schema contract).
2. **Config changes per decisions A/B/C:** levels 4–5 filter, n_problems≈60, max_new_tokens≥8192 + max_model_len 12288, crude text baseline (prefix length + level) in Gate 1 output.
3. **Robustness:** write traces.jsonl before the ungradeable HALT; incomplete-fraction gate; min-included-rows gate; `accelerate` in requirements; OOM handler wraps `from_pretrained`; finiteness check on activations; floor seeds ≥500; probe C swept or in config hash; pytest as stage 0 of smoke_test.sh + tee to log.

## 4. Gates & hour budget (16h research + 2h write-up)

| Gate | When | Decision |
|---|---|---|
| **Gate 1** | h0–2 | 30-min arXiv re-check (last 8 weeks) + smoke test on ~20 level-4/5 problems. GO = pipeline sound, model fails at a usable rate, probe clears floor. NO-GO paths: too slow → prototype on smaller model; no failures → harder problems; no signal → project becomes failed replication (still the project) |
| **Gate 2** | h6 | Pre-registration filled (EXPERIMENT.md §12) BEFORE full runs |
| **Gate 3** | h12 | Re-derive headline by second route; draft exec summary; run whichever experiment its holes expose |

Calendar: research done **Aug 31**; Sept 1–4 write-up only.

## 5. Runpod runbook

**Provision (~5 min):**
1. runpod.io → add **$25** credits (total project GPU spend ≈ $5–15; the rest is buffer).
2. Deploy → Pods → **RTX 4090 (24 GB)** → **Community Cloud** (~$0.35–0.60/hr; verify live).
3. Template: any **RunPod PyTorch, CUDA 12.x**. No vLLM template needed — requirements.txt installs it.
4. **Volume disk 60 GB** (weights ~16 GB + HF cache; the 10–20 GB default fails mid-download — the #1 wasted hour).
5. Optional, recommended after Gate 1: a **Network Volume** holding the HF cache, so weights survive pod teardown.

**Connect:** use the SSH command Runpod displays (web terminal works for a first look; SSH lets you `scp` results back).

**Run (on the box):**
```
git clone https://github.com/isahan78/ais-research.git && cd ais-research
python -m venv .venv && source .venv/bin/activate
pip install -r experiment/requirements.txt
huggingface-cli download Qwen/Qwen3-8B        # start first; ~15 min
pytest experiment/tests/ -v                    # MUST pass before any GPU spend
bash experiment/smoke_test.sh
```
No HuggingFace token needed (Qwen3-8B and MATH-500 are ungated).

**Collect:** `scp` the `outputs/` directory AND the smoke-test log to your Mac before teardown. Console scrollback dies with the pod.

**Teardown — the money trap:** **Stop ≠ Terminate.** A stopped pod bills for storage indefinitely. When the block is done and results are copied off: **Terminate**. Persistence belongs on a network volume, never a parked pod.

## 6. Verification

- CPU anywhere: `pytest experiment/tests/ -v` (no GPU, no weights).
- Per-stage 10-random-examples hand-check: README.md checklist. Every number in the write-up must be traceable to the code that produced it and spot-checked by a human — Nanda's #1 stated disqualifier is unverified agent output.

## 7. Budget

| Item | Est. |
|---|---|
| Gate 1 | ~$1 |
| Full runs (blocks 2–4) | ~$5–15 |
| API credits (LLM-judge baseline) | ~$20 |
| Buffer | ~$25 |
| **Total** | **~$50–60 of $100 authorized** |

## 8. Git workflow

Commit after every block: code, configs, `results.json`, logs, and the filled pre-registration — **never** `*.npz`/weights (gitignored). The repo is private until submission day; flipping it public on Sept 4 is the plan (Nanda values shared code).
