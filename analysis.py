"""Stage 6: Δ(k) = S_probe(k) − S_text(k) — the experiment, as one curve.

`S_text(k)` is the MAX over every text-only reader available at that cut, not
an average and not a favourite. EXPERIMENT.md §5: "take the max; a weak
baseline inflates Δ for free". Taking the max is the whole integrity
mechanism — this file therefore also records, per k, exactly which baselines
were present and which one won, so a reader can see whether the reported Δ
rests on three readers or on one.

This module owns the on-disk CONTRACT the three baselines write to, so the
readers and the writers cannot drift apart:

    outputs/baseline_<name>.json
    {
      "schema_version": 1,
      "baseline": "text_classifier",
      "metric": "roc_auc",
      "per_k": {
        "50": {
          "k_percent": 50,
          "auc": 0.71, "auc_ci95": [0.55, 0.85],
          "n_test": 20, "n_pos": 15, "n_neg": 5,
          "test_row_keys": ["p1@k50", ...],
          "test_labels":   [true, false, ...],
          "test_scores":   [0.83, 0.12, ...],
          "notes": {...}
        }
      },
      "notes": {...}
    }

Per-row `test_scores` are part of the contract because they buy the correct
Δ interval: a PAIRED bootstrap that resamples test rows once and recomputes
both AUCs on the same resample. Δ's two terms are measured on the same rows,
so their sampling errors are correlated; treating them as independent inflates
the interval and would understate a real gap (or manufacture a fake null).

Outputs: analysis.json, fig1_delta_curve.png, fig2_baseline_comparison.png.
matplotlib is imported INSIDE the plotting functions so the analysis maths
stays importable (and testable) on a box with no plotting stack.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from experiment.config import CONFIG, lineage
except ImportError:  # run as a plain script from inside experiment/
    from config import CONFIG, lineage


BASELINE_SCHEMA_VERSION = 1

# The text-side racers, in the order they are reported. A baseline that has not
# been run is simply absent — `analyze` records it as missing rather than
# failing, so a no-API-key laptop run still produces a (weaker, clearly
# labelled) Δ curve.
# Readers admitted to S_text = max over readers. A reader belongs here only if
# its test-time score is a function of the PREFIX ALONE — the same information
# the probe sees. Two readers were withdrawn on 2026-08-30 after the red-team;
# see EXCLUDED_FROM_S_TEXT below and RESULTS.md Run 007.
TEXT_BASELINE_NAMES: Tuple[str, ...] = ("text_classifier",)

# Reported separately, never pooled into S_text. Each entry: why it is excluded.
EXCLUDED_FROM_S_TEXT: dict = {
    "forced_answer": (
        "Its score is grade(forced_answer, GOLD), so it reads the answer key at "
        "test time — an affordance the probe does not have and a monitor could "
        "never have. Whenever the interrupted answer equals the final answer the "
        "score is identical to the label by construction (65% of rows at k=10, "
        "97% at k=90); on the remaining rows its AUC is 0.000. Its apparent "
        "0.71->0.96 rise is the answer-copy rate approaching 1. Retained ONLY as "
        "the gold-free commitment measurement (agreement with the final answer)."
    ),
    "llm_judge": (
        "A difficulty oracle, not a trace reader: Opus 5 scores 0.876 at k=1 with "
        "essentially no reasoning to read — 91% of its k=25 score of 0.959. "
        "MMLU-Pro is public; it is largely solving the item itself."
    ),
}

# The crude Gate-1 floor (prefix token count + difficulty scalar) already lives
# inside results.json rather than its own file. It is a legitimate text-only
# reader, so it competes for the max like the others.
CRUDE_FLOOR_NAME = "crude_floor"

ANALYSIS_PATH = os.path.join(CONFIG.output_dir, "analysis.json")
FIG1_PATH = os.path.join(CONFIG.output_dir, "fig1_delta_curve.png")
FIG2_PATH = os.path.join(CONFIG.output_dir, "fig2_baseline_comparison.png")

# --- figure styling (module-level tunables) ---------------------------------
# Categorical slots from the project's validated palette, assigned in fixed
# order and never cycled. Contrast on a light surface is below 3:1 for slots
# 4-5, so every series is ALSO direct-labelled at its right-hand end — identity
# is never carried by colour alone.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID_COLOR = "#e3e2de"
SURFACE = "#ffffff"
FIG_DPI = 200


# ---------------------------------------------------------------------------
# Baseline contract: writers (used by the three baseline modules) and readers
# ---------------------------------------------------------------------------

def baseline_path(name: str) -> str:
    return os.path.join(CONFIG.output_dir, f"baseline_{name}.json")


def roc_auc(y_true: Sequence[bool], scores: Sequence[float]) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(np.asarray(y_true, dtype=bool), np.asarray(scores, dtype=float)))


def score_at_k(
    y_true: Sequence[bool],
    scores: Sequence[float],
    row_keys: Optional[Sequence[str]] = None,
    n_bootstrap: int = CONFIG.n_bootstrap,
    seed: int = CONFIG.seed,
) -> dict:
    """One baseline's held-out point at one k: AUC + bootstrap CI + raw scores.

    ROC-AUC, never accuracy (EXPERIMENT.md §7): the model's base success rate
    is ~77%, so accuracy is free for a constant predictor.
    """
    try:
        from experiment.train_probe import bootstrap_ci
    except ImportError:
        from train_probe import bootstrap_ci  # type: ignore

    y = np.asarray(y_true, dtype=bool)
    s = np.asarray(scores, dtype=float)
    if len(y) != len(s):
        raise ValueError(f"labels/scores length mismatch: {len(y)} vs {len(s)}")
    if len(set(y.tolist())) < 2:
        raise SystemExit(
            f"HALT: the test split has a single class ({int(y.sum())} True of {len(y)}) — "
            f"AUC is undefined. Regenerate with more problems or a different seed."
        )
    auc = roc_auc(y, s)
    lo, hi = bootstrap_ci(y, s, n_bootstrap, seed)
    return {
        "auc": round(auc, 4),
        "auc_ci95": [round(lo, 4), round(hi, 4)],
        "n_test": int(len(y)),
        "n_pos": int(y.sum()),
        "n_neg": int((~y).sum()),
        "test_row_keys": list(row_keys) if row_keys is not None else None,
        "test_labels": [bool(v) for v in y.tolist()],
        "test_scores": [float(v) for v in s.tolist()],
    }


def write_baseline_json(name: str, per_k: Dict[str, dict], notes: Optional[dict] = None) -> str:
    payload = {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "baseline": name,
        "metric": "roc_auc",
        "per_k": per_k,
        "notes": notes or {},
        "lineage": lineage(CONFIG.prefixes_path),
    }
    os.makedirs(CONFIG.output_dir, exist_ok=True)
    path = baseline_path(name)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def read_baseline_json(name: str) -> Optional[dict]:
    path = baseline_path(name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        payload = json.load(f)
    if payload.get("schema_version") != BASELINE_SCHEMA_VERSION:
        raise RuntimeError(
            f"{path}: schema_version {payload.get('schema_version')} != "
            f"{BASELINE_SCHEMA_VERSION} — regenerate this baseline."
        )
    return payload


# ---------------------------------------------------------------------------
# Pulling the probe and the crude floor out of results.json
# ---------------------------------------------------------------------------

def probe_points(results: dict) -> Dict[int, dict]:
    """{k: {"auc", "auc_ci95", "n_test", "layer"}} for the BEST layer at each k.

    Handles both the current single-k results.json and a future
    `results["per_k"] = {"10": {...}, ...}` layout, so the Δ curve needs no
    change when the k grid lands.
    """
    def one(block: dict, k: int) -> dict:
        best = block["best_layer"]
        p = block["per_layer"][best]
        return {
            "auc": float(p["auc"]),
            "auc_ci95": [float(v) for v in p["auc_ci95"]],
            "n_test": int(block.get("n_test", 0)),
            "layer": best,
        }

    if isinstance(results.get("per_k"), dict):
        return {int(k): one(b, int(k)) for k, b in results["per_k"].items()}
    k = int(results.get("truncation_k_percent", CONFIG.truncation_k_percent))
    return {k: one(results, k)}


def crude_floor_points(results: dict) -> Dict[int, dict]:
    """The Gate-1 crude floor as a competing text baseline (no per-row scores)."""
    def one(block: dict) -> Optional[dict]:
        tf = block.get("text_floor")
        if not tf:
            return None
        return {
            "auc": float(tf["auc"]),
            "auc_ci95": [float(v) for v in tf["auc_ci95"]],
            "n_test": int(tf.get("n_test", block.get("n_test", 0))),
            "test_row_keys": None,
            "test_labels": None,
            "test_scores": None,
            "notes": {"features": tf.get("features")},
        }

    out: Dict[int, dict] = {}
    if isinstance(results.get("per_k"), dict):
        for k, b in results["per_k"].items():
            p = one(b)
            if p:
                out[int(k)] = p
        return out
    p = one(results)
    if p:
        out[int(results.get("truncation_k_percent", CONFIG.truncation_k_percent))] = p
    return out


def probe_test_scores(results: dict, acts_path: str = CONFIG.acts_path) -> Dict[int, dict]:
    """Refit the best-layer probe to recover its per-row test scores.

    train_probe.py records the AUC but not the scores, and it is owned by
    another stage. Refitting here with the SAME seed, the SAME split (read back
    from results.json) and the SAME estimator reproduces its numbers exactly
    and unlocks the paired Δ bootstrap. Purely optional: if acts.npz is absent
    (e.g. analysing on a laptop from copied JSON) we fall back to the
    marginal-CI interval and say so in analysis.json.
    """
    if not os.path.exists(acts_path) or isinstance(results.get("per_k"), dict):
        # per_k acts would need one activation file per cut; not supported yet.
        return {}
    try:
        from experiment.text_floor import split_indices_from_results
        from experiment.train_probe import fit_and_auc
    except ImportError:  # pragma: no cover - script-mode fallback
        from text_floor import split_indices_from_results  # type: ignore
        from train_probe import fit_and_auc  # type: ignore

    try:
        data = np.load(acts_path, allow_pickle=False)
        pids = [str(p) for p in data["problem_ids"]]
        y = data["labels"].astype(bool)
        train_idx, test_idx = split_indices_from_results(pids, results)
        # results.json names the layer "layer_18"; harvest_activations.py stores
        # it as "acts_layer18" (no underscore before the index).
        best_layer = results["best_layer"]
        X = data[f"acts_layer{best_layer.rsplit('_', 1)[-1]}"]
        auc, scores = fit_and_auc(
            X[train_idx], y[train_idx], X[test_idx], y[test_idx], CONFIG.seed
        )
    except Exception as exc:  # never let an optional refit kill the analysis
        print(f"WARNING: could not refit the probe for paired Δ CIs ({exc}); "
              f"falling back to marginal-CI intervals.")
        return {}

    k = int(results.get("truncation_k_percent", CONFIG.truncation_k_percent))
    return {
        k: {
            "auc": float(auc),
            "test_row_keys": [f"{pids[i]}@k{k}" for i in test_idx],
            "test_labels": [bool(v) for v in y[test_idx].tolist()],
            "test_scores": [float(v) for v in scores.tolist()],
        }
    }


# ---------------------------------------------------------------------------
# S_text = max over readers, and Δ with an interval
# ---------------------------------------------------------------------------

def s_text_at_k(points_by_name: Dict[str, dict]) -> Tuple[Optional[float], Optional[str], List[str]]:
    """(S_text, winning baseline, all contributors) at one cut.

    The max — never the mean, never a preferred reader. A missing baseline
    lowers the max and therefore INFLATES Δ, which is exactly why the
    contributor list is reported alongside every number.
    """
    contributors = sorted(n for n, p in points_by_name.items() if p and p.get("auc") is not None)
    if not contributors:
        return None, None, []
    winner = max(contributors, key=lambda n: points_by_name[n]["auc"])
    return float(points_by_name[winner]["auc"]), winner, contributors


def align_scores(
    probe: dict, text: dict
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """(y, probe_scores, text_scores) over the rows both readers scored.

    Returns None when either side lacks per-row scores, or when their row keys
    disagree — better to fall back to a wider interval than to pair up rows
    that are not the same rows.
    """
    for side in (probe, text):
        if not side or side.get("test_scores") is None or side.get("test_row_keys") is None:
            return None
    p_keys = list(probe["test_row_keys"])
    t_keys = list(text["test_row_keys"])
    if set(p_keys) != set(t_keys):
        return None
    p_by = dict(zip(p_keys, zip(probe["test_labels"], probe["test_scores"])))
    t_by = dict(zip(t_keys, text["test_scores"]))
    t_labels = dict(zip(t_keys, text["test_labels"]))
    order = sorted(p_keys)
    if any(bool(p_by[k][0]) != bool(t_labels[k]) for k in order):
        raise RuntimeError(
            "probe and text baseline disagree on the label of a shared test row — "
            "the two readers are not looking at the same data."
        )
    y = np.array([bool(p_by[k][0]) for k in order])
    ps = np.array([float(p_by[k][1]) for k in order])
    ts = np.array([float(t_by[k]) for k in order])
    return y, ps, ts


def paired_delta_bootstrap(
    y: np.ndarray,
    probe_scores: np.ndarray,
    text_scores: np.ndarray,
    n_bootstrap: int = CONFIG.n_bootstrap,
    seed: int = CONFIG.seed,
) -> Tuple[float, float, float, float]:
    """(delta, ci_low, ci_high, frac_resamples_delta_gt_0), paired over rows.

    One resample of the test rows, both AUCs recomputed on it, difference
    taken — so the correlation between the two readers is carried into the
    interval instead of being thrown away.
    """
    y = np.asarray(y, dtype=bool)
    ps = np.asarray(probe_scores, dtype=float)
    ts = np.asarray(text_scores, dtype=float)
    delta = roc_auc(y, ps) - roc_auc(y, ts)

    rng = np.random.default_rng(seed)
    n = len(y)
    deltas: List[float] = []
    attempts = 0
    max_attempts = 50 * n_bootstrap
    while len(deltas) < n_bootstrap:
        attempts += 1
        if attempts > max_attempts:
            raise SystemExit(
                f"HALT: paired bootstrap drew {attempts - 1} resamples but only "
                f"{len(deltas)} had both classes (need {n_bootstrap}). Test set "
                f"n={n} is too small/skewed for a Δ interval."
            )
        idx = rng.integers(0, n, size=n)
        if len(set(y[idx].tolist())) < 2:
            continue
        deltas.append(roc_auc(y[idx], ps[idx]) - roc_auc(y[idx], ts[idx]))
    arr = np.asarray(deltas)
    return (
        float(delta),
        float(np.percentile(arr, 2.5)),
        float(np.percentile(arr, 97.5)),
        float(np.mean(arr > 0)),
    )


def delta_ci_from_marginals(
    auc_probe: float,
    ci_probe: Sequence[float],
    auc_text: float,
    ci_text: Sequence[float],
) -> Tuple[float, float, float]:
    """Fallback Δ interval when per-row scores are unavailable on one side.

    Treats the two AUCs as independent and normal, with each SE recovered from
    its own 95% interval width. Independence is FALSE here (both readers are
    scored on the same rows and their errors are positively correlated), which
    makes this interval CONSERVATIVE — wider than the truth. That direction is
    the safe one for a claim of the form "Δ is greater than zero", and
    analysis.json labels every point that used it.
    """
    se_p = (float(ci_probe[1]) - float(ci_probe[0])) / (2 * 1.959963985)
    se_t = (float(ci_text[1]) - float(ci_text[0])) / (2 * 1.959963985)
    se = float(np.sqrt(se_p**2 + se_t**2))
    delta = float(auc_probe) - float(auc_text)
    return delta, delta - 1.959963985 * se, delta + 1.959963985 * se


# ---------------------------------------------------------------------------
# The analysis
# ---------------------------------------------------------------------------

def analyze(
    results: dict,
    baselines: Dict[str, dict],
    probe_scores_by_k: Optional[Dict[int, dict]] = None,
    n_bootstrap: int = CONFIG.n_bootstrap,
    seed: int = CONFIG.seed,
) -> dict:
    """Δ(k) for every cut the probe was fit at.

    `baselines` maps baseline name -> its per-k point dict (already keyed by
    int k). Missing baselines are recorded, not fatal.
    """
    probe_scores_by_k = probe_scores_by_k or {}
    probes = probe_points(results)
    per_k: Dict[str, dict] = {}

    for k in sorted(probes):
        probe = probes[k]
        points = {name: pts.get(k) for name, pts in baselines.items()}
        present = {n: p for n, p in points.items() if p}
        s_text, winner, contributors = s_text_at_k(present)

        entry: dict = {
            "k_percent": k,
            "s_probe": round(probe["auc"], 4),
            "s_probe_ci95": [round(v, 4) for v in probe["auc_ci95"]],
            "probe_layer": probe["layer"],
            "n_test": probe["n_test"],
            "text_baselines": {
                n: {"auc": round(p["auc"], 4), "auc_ci95": [round(v, 4) for v in p["auc_ci95"]]}
                for n, p in sorted(present.items())
            },
            "baselines_present": contributors,
            "baselines_missing": sorted(set(baselines) - set(contributors)),
        }

        if s_text is None:
            entry.update(
                s_text=None, s_text_source=None, delta=None, delta_ci95=None,
                delta_ci_method="none",
                warning="no text baseline available at this k — Δ is not defined; "
                        "any probe number here is unaudited.",
            )
            per_k[str(k)] = entry
            continue

        entry["s_text"] = round(s_text, 4)
        entry["s_text_source"] = winner

        probe_point = probe_scores_by_k.get(k)
        aligned = align_scores(probe_point, present[winner]) if probe_point else None
        if aligned is not None:
            y, ps, ts = aligned
            delta, lo, hi, frac_pos = paired_delta_bootstrap(y, ps, ts, n_bootstrap, seed)
            entry.update(
                delta=round(delta, 4),
                delta_ci95=[round(lo, 4), round(hi, 4)],
                delta_ci_method="paired_bootstrap",
                frac_bootstrap_delta_gt_0=round(frac_pos, 4),
            )
        else:
            delta, lo, hi = delta_ci_from_marginals(
                probe["auc"], probe["auc_ci95"], s_text, present[winner]["auc_ci95"]
            )
            entry.update(
                delta=round(delta, 4),
                delta_ci95=[round(lo, 4), round(hi, 4)],
                delta_ci_method="independent_normal_approx_conservative",
            )
        per_k[str(k)] = entry

    all_names = sorted(baselines)
    ever_present = sorted({n for e in per_k.values() for n in e["baselines_present"]})
    return {
        "metric": "roc_auc",
        "definition": "delta(k) = S_probe(k) - S_text(k); S_text = max over text-only readers",
        "per_k": per_k,
        "baselines_configured": all_names,
        "baselines_contributing": ever_present,
        "baselines_never_available": sorted(set(all_names) - set(ever_present)),
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "lineage": lineage(CONFIG.results_path),
    }


def load_all_baselines(results: dict) -> Dict[str, Dict[int, dict]]:
    """Every text baseline currently on disk, keyed name -> k -> point."""
    out: Dict[str, Dict[int, dict]] = {}
    for name in TEXT_BASELINE_NAMES:
        payload = read_baseline_json(name)
        if payload is None:
            out[name] = {}
            continue
        out[name] = {int(k): p for k, p in payload["per_k"].items()}
    out[CRUDE_FLOOR_NAME] = crude_floor_points(results)
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_COLOR)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=0)


def _caption(fig, lines: Sequence[str]) -> None:
    """Footnote under the plot. Placed in FIGURE coordinates with reserved
    space, not as an axes annotation — an axes annotation wider than the axes
    stretches the saved canvas."""
    fig.text(0.012, 0.015, "\n".join(lines), fontsize=7.2, color=INK_SECONDARY,
             va="bottom", ha="left")


def _decollide(values: Sequence[float], min_gap: float) -> List[float]:
    """Nudge label positions apart, top-down, so end-of-line labels stay legible
    when two series finish within a hair of each other."""
    order = sorted(range(len(values)), key=lambda i: -values[i])
    out = list(values)
    prev = None
    for i in order:
        v = values[i]
        if prev is not None and prev - v < min_gap:
            v = prev - min_gap
        out[i] = v
        prev = v
    return out


def figure_delta_curve(analysis: dict, path: str = FIG1_PATH) -> Optional[str]:
    """Figure 1 — Δ(k) with its bootstrap CI band.

    One series, so no legend box: the title names it, and zero is drawn as a
    labelled reference line — the only comparison a reader needs.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = [e for e in analysis["per_k"].values() if e.get("delta") is not None]
    if not pts:
        print("WARNING: no k has both a probe and a text baseline — skipping Figure 1.")
        return None
    pts.sort(key=lambda e: e["k_percent"])
    ks = [e["k_percent"] for e in pts]
    d = np.array([e["delta"] for e in pts], dtype=float)
    lo = np.array([e["delta_ci95"][0] for e in pts], dtype=float)
    hi = np.array([e["delta_ci95"][1] for e in pts], dtype=float)

    fig, ax = plt.subplots(figsize=(6.8, 4.4), dpi=FIG_DPI, facecolor=SURFACE)
    _style_axes(ax)

    if len(ks) > 1:
        ax.fill_between(ks, lo, hi, color=SERIES_COLORS[0], alpha=0.18, linewidth=0, zorder=2)
        ax.plot(ks, d, color=SERIES_COLORS[0], linewidth=2, zorder=3)
    else:
        ax.errorbar(ks, d, yerr=[d - lo, hi - d], color=SERIES_COLORS[0],
                    elinewidth=2, capsize=6, capthick=2, linestyle="none", zorder=3)
    ax.plot(ks, d, "o", color=SERIES_COLORS[0], markersize=7,
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)

    # Zero is the reference the whole figure is about, so it must always be on
    # screen — otherwise an all-negative (or all-positive) Δ is drawn without
    # the line it is being compared to.
    y_min = min(float(lo.min()), 0.0)
    y_max = max(float(hi.max()), 0.0)
    span = max(y_max - y_min, 1e-6)
    ax.set_ylim(y_min - 0.16 * span, y_max + 0.16 * span)
    ax.axhline(0.0, color=INK_SECONDARY, linewidth=1.2, linestyle="--", zorder=1)
    ax.annotate("Δ = 0 — text monitoring loses nothing",
                xy=(0.0, 0.0), xycoords=("axes fraction", "data"),
                xytext=(4, -4), textcoords="offset points",
                ha="left", va="top", fontsize=8, color=INK_SECONDARY, zorder=5)

    # Selective direct labels — never a number on every point once the curve is
    # dense enough for them to collide.
    label_at = range(len(ks)) if len(ks) <= 3 else (0, len(ks) - 1)
    for i in label_at:
        ax.annotate(f"{d[i]:+.3f}", xy=(ks[i], d[i]), xytext=(0, 11),
                    textcoords="offset points", ha="center", fontsize=8.5,
                    color=INK_PRIMARY, zorder=5)

    ax.set_xlabel("truncation point k (% of thinking tokens)", fontsize=9.5, color=INK_SECONDARY)
    ax.set_ylabel("Δ  =  AUC(probe) − AUC(best text reader)", fontsize=9.5, color=INK_SECONDARY)
    ax.set_title("Figure 1 — what the internals know that the page does not",
                 fontsize=11.5, color=INK_PRIMARY, loc="left", pad=12)
    ax.set_xticks(ks)

    methods = sorted({e["delta_ci_method"] for e in pts})
    contributors = analysis.get("baselines_contributing") or []
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    _caption(fig, [
        f"Band: 95% bootstrap CI ({', '.join(methods)}).",
        f"S_text = max over {', '.join(contributors) if contributors else 'no'} text reader(s).",
    ])
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def figure_baseline_comparison(analysis: dict, path: str = FIG2_PATH) -> Optional[str]:
    """Figure 2 — every racer's AUC against k, on ONE axis (never two)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    entries = sorted(analysis["per_k"].values(), key=lambda e: e["k_percent"])
    if not entries:
        print("WARNING: analysis has no k points — skipping Figure 2.")
        return None
    ks = [e["k_percent"] for e in entries]

    # Fixed slot order — colour follows the reader, never its rank, so a rerun
    # in which a baseline drops out does not repaint the survivors.
    contributing = analysis.get("baselines_contributing") or []
    ordered = [n for n in (*TEXT_BASELINE_NAMES, CRUDE_FLOOR_NAME) if n in contributing]
    series: List[Tuple[str, List[Optional[float]]]] = [
        ("probe (internals)", [e["s_probe"] for e in entries])
    ]
    for name in ordered:
        series.append((name, [e["text_baselines"].get(name, {}).get("auc") for e in entries]))

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=FIG_DPI, facecolor=SURFACE)
    _style_axes(ax)

    drawn: List[Tuple[str, float, float, str]] = []   # label, x_end, y_end, colour
    for i, (label, vals) in enumerate(series):
        color = SERIES_COLORS[i % len(SERIES_COLORS)]
        xy = [(k, v) for k, v in zip(ks, vals) if v is not None]
        if not xy:
            continue
        xs, ys = zip(*xy)
        if len(xs) > 1:
            ax.plot(xs, ys, linewidth=2, color=color, label=label, zorder=3 + i)
            ax.plot(xs, ys, "o", markersize=7, color=color, markeredgecolor=SURFACE,
                    markeredgewidth=2, zorder=3 + i)
        else:
            ax.plot(xs, ys, "o", markersize=7, color=color, markeredgecolor=SURFACE,
                    markeredgewidth=2, label=label, zorder=3 + i)
        drawn.append((label, xs[-1], ys[-1], color))

    all_y = [v for _l, vals in series for v in vals if v is not None] + [0.5]
    y_lo, y_hi = min(all_y), max(all_y)
    pad = max(0.05 * (y_hi - y_lo), 0.02)
    ax.set_ylim(y_lo - pad, y_hi + pad)
    ax.axhline(0.5, color=INK_SECONDARY, linewidth=1.0, linestyle=":", zorder=1)
    ax.annotate("0.5 — chance", xy=(1.0, 0.5), xycoords=("axes fraction", "data"),
                xytext=(-4, 3), textcoords="offset points", ha="right", va="bottom",
                fontsize=8, color=INK_SECONDARY, zorder=5)

    # Direct labels at the right-hand end: two palette slots sit under 3:1 on a
    # light surface, so identity must never rest on colour alone.
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    label_y = _decollide([y for _l, _x, y, _c in drawn], min_gap=0.055 * span)
    for (label, x_end, _y, _color), y_lab in zip(drawn, label_y):
        ax.annotate(label, xy=(x_end, y_lab), xytext=(9, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK_PRIMARY, zorder=5)

    ax.set_xlabel("truncation point k (% of thinking tokens)", fontsize=9.5, color=INK_SECONDARY)
    ax.set_ylabel("held-out ROC-AUC", fontsize=9.5, color=INK_SECONDARY)
    ax.set_title("Figure 2 — the race, reader by reader", fontsize=11.5,
                 color=INK_PRIMARY, loc="left", pad=12)
    ax.set_xticks(ks)
    k_span = max(max(ks) - min(ks), 1)
    ax.set_xlim(min(ks) - 0.05 * k_span, max(ks) + 0.42 * k_span)
    # Legend below the plot so it can never sit on top of the data.
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_PRIMARY,
              loc="upper center", bbox_to_anchor=(0.5, -0.14),
              ncol=min(len(drawn), 3), handlelength=1.6, columnspacing=1.4)

    missing = analysis.get("baselines_never_available") or []
    # Only reserve footnote space when there IS a footnote.
    fig.tight_layout(rect=(0, 0.09, 1, 1) if missing else None)
    if missing:
        _caption(fig, [
            f"Not run: {', '.join(missing)}.",
            "S_text is a max over fewer readers than designed, so Δ is an UPPER bound.",
        ])
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------

def main() -> None:
    try:
        with open(CONFIG.results_path) as f:
            results = json.load(f)
    except OSError:
        raise SystemExit(
            "HALT: results.json not found — run train_probe (and text_floor) first."
        )

    baselines = load_all_baselines(results)
    missing = [n for n, p in baselines.items() if not p]
    if missing:
        print(f"NOTE: no results on disk for text baseline(s): {', '.join(missing)}. "
              f"S_text is a max over the remaining readers, which can only make Δ "
              f"LOOK BIGGER — this is recorded in analysis.json.")

    out = analyze(results, baselines, probe_scores_by_k=probe_test_scores(results))
    os.makedirs(CONFIG.output_dir, exist_ok=True)
    with open(ANALYSIS_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"analysis: -> {ANALYSIS_PATH}")

    for k in sorted(out["per_k"], key=int):
        e = out["per_k"][k]
        if e["delta"] is None:
            print(f"  k={k}%: S_probe={e['s_probe']} but NO text baseline — Δ undefined.")
            continue
        print(f"  k={k}%: S_probe={e['s_probe']}  S_text={e['s_text']} "
              f"(best: {e['s_text_source']})  Δ={e['delta']:+.4f} "
              f"CI95={e['delta_ci95']} [{e['delta_ci_method']}]")

    for p in (figure_delta_curve(out), figure_baseline_comparison(out)):
        if p:
            print(f"analysis: figure -> {p}")


if __name__ == "__main__":
    main()
