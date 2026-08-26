"""Stage 3: prefill-only activation harvest with HuggingFace.

Reads prefixes.jsonl, writes acts.npz (last-prefix-token residual stream at
layers 9/18/27, aligned arrays keyed by the problem_ids array).

Hard rules encoded here (see spec Code Map — agent-verified, do not re-derive):
  * AutoModel -> Qwen3Model, NOT Qwen3ForCausalLM (the LM head would compute
    logits at every position, ~2.32 GiB wasted at 4k context).
  * outputs.hidden_states has num_layers+1 = 37 entries: [0] is embeddings,
    [L+1] is the raw residual after decoder layer L. [-1] is post-final-RMSNorm
    and must NEVER be used — the raw layer-35 residual is unrecoverable.
  * No generation, ever. Single forward pass under torch.inference_mode().
  * Prompt alignment: assert stage-1 prompt_token_ids is a prefix of each
    prefix_token_ids — if this fails, activations would describe a different
    context than the trace and every downstream number is quietly wrong.

Runs only after generate_traces.py has fully exited (engine teardown between
phases; co-residency OOMs on 24 GiB).
"""

from __future__ import annotations

import contextlib
import json
import time

import numpy as np

try:
    from experiment.config import CONFIG, THINK_END_ID, lineage
except ImportError:
    from config import CONFIG, THINK_END_ID, lineage


def oom_halt_message(err: object) -> str:
    """Actionable remedy text for a CUDA OOM during harvest (I/O matrix row 5).

    Module-level and torch-free so the OOM contract is testable on machines
    without CUDA (tests/test_invariants.py::TestOomHandling).
    """
    return (
        "HALT: CUDA OOM during harvest. Remedies: (1) reduce harvest batch size / "
        "sequence length, (2) confirm vLLM has fully exited (no co-residency), "
        "(3) if you switched to a *ForCausalLM class by mistake, switch back to "
        "AutoModel or pass logits_to_keep=1. Original error: " + str(err)
    )


def is_cuda_oom(err: BaseException) -> bool:
    """Recognize a CUDA OOM without importing torch (testable on CPU boxes).

    torch.cuda.OutOfMemoryError is class-named `OutOfMemoryError`; older paths
    surface a RuntimeError whose message contains 'CUDA out of memory'.
    """
    return type(err).__name__ == "OutOfMemoryError" or "CUDA out of memory" in str(err)


@contextlib.contextmanager
def oom_guard():
    """Convert any CUDA OOM into a SystemExit carrying the remedy text.

    Wraps BOTH model loading (from_pretrained can OOM if vLLM is still
    resident) and the forward loop. Non-OOM errors propagate unchanged.
    """
    try:
        yield
    except BaseException as e:  # noqa: BLE001 — must catch torch's OutOfMemoryError subclass
        if is_cuda_oom(e):
            raise SystemExit(oom_halt_message(e)) from e
        raise


def residual_index(layer: int, n_hidden_states: int) -> int:
    """hidden_states index of the RAW residual after decoder layer `layer`.

    hidden_states[0] is embeddings, so the residual after layer L is at L+1.
    hidden_states[-1] is post-final-RMSNorm — the raw last-layer residual is
    unrecoverable there — so mapping onto the final entry is refused.
    Raises (never asserts) so the guard survives `python -O`.
    """
    if layer < 0:
        raise ValueError(f"layer must be >= 0, got {layer}")
    idx = layer + 1
    if idx >= n_hidden_states - 1:
        raise ValueError(
            f"layer {layer} maps to hidden_states[{idx}] of {n_hidden_states} — the final "
            f"entry is post-final-RMSNorm (the [-1] trap); the raw residual is unrecoverable."
        )
    return idx


def check_activation_vector(vec: np.ndarray, problem_id: str, layer: int,
                            hidden_size: int = CONFIG.hidden_size) -> None:
    """Raise (not assert — must survive python -O) on a degenerate activation."""
    if vec.shape != (hidden_size,):
        raise RuntimeError(
            f"{problem_id} layer {layer}: activation shape {vec.shape}, expected ({hidden_size},)"
        )
    if not np.isfinite(vec).all():
        raise RuntimeError(
            f"{problem_id} layer {layer}: non-finite values in activation (NaN/Inf) — "
            f"harvest is corrupt; do not train a probe on this."
        )
    if not np.any(vec):
        raise RuntimeError(
            f"{problem_id} layer {layer}: activation is all-zero — the forward pass did not "
            f"produce a real residual; do not train a probe on this."
        )


def load_included_prefixes(path: str):
    meta, rows = None, []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("record_type") == "meta":
                meta = rec
                continue
            if rec["included"]:
                rows.append(rec)
    return meta, rows


def main() -> None:
    import torch
    from transformers import AutoModel

    t0 = time.time()
    meta_in, rows = load_included_prefixes(CONFIG.prefixes_path)
    if not rows:
        raise SystemExit("HALT: prefixes.jsonl contains no included rows — nothing to harvest.")

    # --- invariant: prompt/prefix alignment (stage 1 vs stage 2). ------------
    # raise, not assert: these guards must survive `python -O`.
    for r in rows:
        pids = r["prompt_token_ids"]
        if r["prefix_token_ids"][: len(pids)] != pids:
            raise RuntimeError(
                f"{r['problem_id']}: prefix does not start with the stage-1 prompt token ids — "
                "prompt alignment broken; refusing to harvest misaligned activations."
            )
        if THINK_END_ID in r["prefix_token_ids"][len(pids):]:
            raise RuntimeError(
                f"{r['problem_id']}: </think> present in prefix — truncation invariant violated."
            )

    if not torch.cuda.is_available():
        raise SystemExit("HALT: CUDA GPU required for harvest. This stage does not run on Mac.")

    # from_pretrained is inside the OOM guard on purpose: model load is the
    # first place a lingering vLLM co-residency OOMs.
    with oom_guard():
        model = AutoModel.from_pretrained(
            CONFIG.model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="cuda",
        )
    model.eval()

    # --- invariant: no generation possible/attempted --------------------------
    if model.__class__.__name__ != "Qwen3Model":
        raise RuntimeError(
            f"Expected bare Qwen3Model (no LM head), got {model.__class__.__name__}"
        )
    if model.can_generate():
        raise RuntimeError("Model must not be generation-capable in the harvest phase.")

    layer_acts = {L: [] for L in CONFIG.layers}
    problem_ids, labels = [], []

    with oom_guard(), torch.inference_mode():
        for r in rows:
            ids = torch.tensor([r["prefix_token_ids"]], dtype=torch.long, device="cuda")
            out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
            hs = out.hidden_states
            if len(hs) != CONFIG.num_decoder_layers + 1:
                raise RuntimeError(
                    f"Expected {CONFIG.num_decoder_layers + 1} hidden_states entries, got {len(hs)}"
                )
            for L in CONFIG.layers:
                # residual_index enforces [L+1], never [-1] (post-norm trap).
                vec = hs[residual_index(L, len(hs))][0, -1, :].float().cpu().numpy()
                check_activation_vector(vec, r["problem_id"], L)
                layer_acts[L].append(vec)
            problem_ids.append(r["problem_id"])
            labels.append(bool(r["label"]))
            del out, hs

    save_kwargs = {
        "problem_ids": np.array(problem_ids),
        "labels": np.array(labels, dtype=bool),
        "config_hash": np.array(CONFIG.config_hash()),
        "input_file": np.array(CONFIG.prefixes_path),
    }
    for L in CONFIG.layers:
        save_kwargs[f"acts_layer{L}"] = np.stack(layer_acts[L]).astype(np.float32)

    np.savez_compressed(CONFIG.acts_path, **save_kwargs)

    # JSONL lineage sidecar (every stage writes JSONL lineage; npz is binary).
    with open(CONFIG.acts_path + ".lineage.jsonl", "w") as f:
        f.write(json.dumps({
            "record_type": "meta",
            "stage": "harvest_activations",
            "n_rows": len(problem_ids),
            "layers": list(CONFIG.layers),
            "elapsed_s": round(time.time() - t0, 1),
            "upstream_config_hash": (meta_in or {}).get("config_hash"),
            **lineage(CONFIG.prefixes_path),
        }) + "\n")

    print(f"harvest_activations: {len(problem_ids)} prefixes x layers {list(CONFIG.layers)} "
          f"-> {CONFIG.acts_path} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
