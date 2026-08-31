"""Run 010 — honestly-selected probe numbers.

The probe AUCs in RESULTS.md Run 005/007 are compromised two ways:

  1. `CONFIG.probe_C = 1.0` was never swept.
  2. `best_layer` is `argmax` over the three layers' **test** AUCs — model
     selection performed on the evaluation set.

Together those make the reported per-k probe curve (and in particular its
"non-monotonicity" 0.761 -> 0.647 -> 0.681) a selection artifact of unknown
size. This script recomputes the probe the defensible way and writes
`outputs/block2/honest_probe.json`.

Honesty contract, enforced structurally rather than by convention:

  * The recorded split is REUSED VERBATIM. `load_cut` reads
    `results.json -> split.train_problem_ids / test_problem_ids` and refuses
    to run if the two sides overlap, if a row's problem_id appears in
    neither, or if any recorded id is missing from the activations.
  * `select_by_cv` is the ONLY place a hyper-parameter is chosen, and its
    signature accepts training features/labels/groups only. It is physically
    incapable of reading a test row: no test array, no test index, and no
    path from which one could be loaded is passed in. `tests/
    test_honest_probe.py` pins this (both by signature and by showing the
    selection is invariant to arbitrary mutation of the held-out rows).
  * Selection uses StratifiedGroupKFold on problem_id inside the training
    split only, repeated over several shuffles to damp fold noise.

Nothing here modifies train_probe.py / config.py / analysis.py / redteam/.
CPU only, no network, no new data.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

EXPERIMENT_DIR = Path(__file__).resolve().parent
BLOCK2_DIR = EXPERIMENT_DIR / "outputs" / "block2"

K_VALUES: Tuple[int, ...] = (1, 10, 25, 50, 75, 90)
LAYERS: Tuple[int, ...] = (9, 18, 27)

#: C sweep, log-spaced across 1e-5 .. 1e2 (the brief's floor and ceiling).
C_GRID: Tuple[float, ...] = tuple(float(c) for c in np.logspace(-5, 2, 15))

CV_SPLITS = 5
CV_REPEATS = 5
SEED = 0
N_BOOTSTRAP = 2000


# --------------------------------------------------------------------------
# data loading — the recorded split, reused verbatim
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Cut:
    """One truncation cut's activations plus the split recorded for it."""

    k: int
    problem_ids: np.ndarray          # (n,) str
    labels: np.ndarray               # (n,) bool
    layer_acts: Dict[int, np.ndarray]  # layer -> (n, d) float32
    train_idx: np.ndarray            # row indices, from the RECORDED split
    test_idx: np.ndarray
    results: dict = field(repr=False, default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.labels)


def split_indices_from_results(
    problem_ids: Sequence[str], results: Mapping
) -> Tuple[np.ndarray, np.ndarray]:
    """Row indices for the split RECORDED in results.json — never re-derived.

    Raises (not asserts: must survive ``python -O``) if the recorded split is
    not a clean partition of the rows we hold.
    """
    try:
        train_pids = set(results["split"]["train_problem_ids"])
        test_pids = set(results["split"]["test_problem_ids"])
    except (KeyError, TypeError) as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            "results.json has no split.train_problem_ids / test_problem_ids block; "
            "refusing to invent a split."
        ) from exc

    overlap = train_pids & test_pids
    if overlap:
        raise RuntimeError(
            f"recorded split has problem_ids on BOTH sides: {sorted(overlap)[:10]}"
        )

    pids = [str(p) for p in problem_ids]
    unassigned = sorted({p for p in pids if p not in train_pids and p not in test_pids})
    if unassigned:
        raise RuntimeError(
            f"problem_ids absent from the recorded split: {unassigned[:10]} — "
            f"results.json and the activations are out of sync."
        )
    held = set(pids)
    missing = sorted((train_pids | test_pids) - held)
    if missing:
        raise RuntimeError(
            f"recorded split names problem_ids that are not in the activations: {missing[:10]}"
        )

    train_idx = np.array([i for i, p in enumerate(pids) if p in train_pids], dtype=int)
    test_idx = np.array([i for i, p in enumerate(pids) if p in test_pids], dtype=int)
    return train_idx, test_idx


def load_cut(k: int, block_dir: Path = BLOCK2_DIR, layers: Sequence[int] = LAYERS) -> Cut:
    """Load one cut's acts.npz and pair it with that cut's recorded split."""
    cut_dir = Path(block_dir) / f"k{k}"
    with open(cut_dir / "results.json") as f:
        results = json.load(f)
    data = np.load(cut_dir / "acts.npz", allow_pickle=False)

    problem_ids = np.array([str(p) for p in data["problem_ids"]])
    labels = data["labels"].astype(bool)
    layer_acts = {int(L): np.asarray(data[f"acts_layer{L}"], dtype=np.float64) for L in layers}
    train_idx, test_idx = split_indices_from_results(problem_ids, results)
    return Cut(
        k=k,
        problem_ids=problem_ids,
        labels=labels,
        layer_acts=layer_acts,
        train_idx=train_idx,
        test_idx=test_idx,
        results=results,
    )


def feature_sets(cut: Cut, rows: np.ndarray) -> Dict[str, np.ndarray]:
    """Named candidate feature matrices restricted to ``rows``.

    ``concat_9_18_27`` is the all-three-layers variant; it is selected by the
    same honest procedure as the single layers.
    """
    out: Dict[str, np.ndarray] = {f"layer_{L}": X[rows] for L, X in cut.layer_acts.items()}
    out["concat_" + "_".join(str(L) for L in sorted(cut.layer_acts))] = np.concatenate(
        [cut.layer_acts[L][rows] for L in sorted(cut.layer_acts)], axis=1
    )
    return out


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------


def make_probe(C: float, seed: int = SEED):
    """The existing probe recipe (StandardScaler -> logistic), C exposed."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, C=C, random_state=seed),
    )


def fit_score(X_train, y_train, X_eval, C: float, seed: int = SEED) -> np.ndarray:
    clf = make_probe(C, seed)
    clf.fit(X_train, y_train)
    return clf.predict_proba(X_eval)[:, 1]


# --------------------------------------------------------------------------
# CV selection — TRAIN ONLY. No test array reaches this function.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Selection:
    name: str
    C: float
    cv_auc: float
    cv_table: Dict[str, Dict[str, float]]


def cv_auc_for(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: Sequence[str],
    C: float,
    n_splits: int = CV_SPLITS,
    n_repeats: int = CV_REPEATS,
    seed: int = SEED,
) -> float:
    """Mean over repeats of the pooled out-of-fold AUC, inside the train split.

    StratifiedGroupKFold on problem_id: a problem never straddles a fold, the
    same invariant the outer split enforces. (At this cut each problem has
    exactly one row, so grouping is a no-op in practice — it is kept so the
    code stays correct if rows are ever pooled across cuts.)
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold

    y_train = np.asarray(y_train).astype(bool)
    groups = np.asarray(groups_train)
    aucs: List[float] = []
    for rep in range(n_repeats):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed + rep
        )
        oof = np.full(len(y_train), np.nan, dtype=float)
        for fit_rows, held_rows in splitter.split(X_train, y_train, groups):
            oof[held_rows] = fit_score(
                X_train[fit_rows], y_train[fit_rows], X_train[held_rows], C, seed=seed
            )
        scored = ~np.isnan(oof)
        if len(set(y_train[scored].tolist())) < 2:
            continue
        aucs.append(float(roc_auc_score(y_train[scored], oof[scored])))
    if not aucs:
        return float("nan")
    return float(np.mean(aucs))


def select_by_cv(
    train_feature_sets: Mapping[str, np.ndarray],
    y_train: np.ndarray,
    groups_train: Sequence[str],
    c_grid: Sequence[float] = C_GRID,
    n_splits: int = CV_SPLITS,
    n_repeats: int = CV_REPEATS,
    seed: int = SEED,
) -> Selection:
    """Pick (feature set, C) by cross-validation INSIDE THE TRAINING SPLIT.

    THE honesty invariant of this module: the parameters are training
    features, training labels and training groups. There is no test matrix,
    no test index array, and no filesystem path here — this function cannot
    observe a held-out row even in principle. Ties are broken toward the
    stronger regularizer (smaller C), then alphabetically by feature set, so
    the choice is deterministic.
    """
    y_train = np.asarray(y_train).astype(bool)
    if len(set(y_train.tolist())) < 2:
        raise RuntimeError("training split is single-class; cannot cross-validate.")
    for name, X in train_feature_sets.items():
        if len(X) != len(y_train):
            raise RuntimeError(
                f"feature set {name!r} has {len(X)} rows but y_train has {len(y_train)} — "
                f"a non-training matrix was passed to select_by_cv."
            )
    if len(groups_train) != len(y_train):
        raise RuntimeError("groups_train and y_train lengths disagree.")

    table: Dict[str, Dict[str, float]] = {}
    best: Tuple[float, float, str] | None = None  # (-auc, C, name) -> min()
    for name in sorted(train_feature_sets):
        X = train_feature_sets[name]
        row: Dict[str, float] = {}
        for C in c_grid:
            auc = cv_auc_for(X, y_train, groups_train, C, n_splits, n_repeats, seed)
            row[f"{C:.6g}"] = round(auc, 4)
            if np.isnan(auc):
                continue
            key = (-auc, C, name)
            if best is None or key < best:
                best = key
        table[name] = row

    if best is None:  # pragma: no cover - defensive
        raise RuntimeError("no (feature set, C) combination produced a finite CV AUC.")
    neg_auc, C, name = best
    return Selection(name=name, C=float(C), cv_auc=float(-neg_auc), cv_table=table)


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------


def bootstrap_ci(
    y_test: np.ndarray,
    scores: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> Tuple[float, float]:
    """Percentile bootstrap over test rows; single-class resamples are redrawn."""
    from sklearn.metrics import roc_auc_score

    y_test = np.asarray(y_test).astype(bool)
    scores = np.asarray(scores, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y_test)
    aucs: List[float] = []
    attempts = 0
    max_attempts = 50 * n_bootstrap
    while len(aucs) < n_bootstrap:
        attempts += 1
        if attempts > max_attempts:  # pragma: no cover - defensive
            raise SystemExit("HALT: bootstrap could not draw two-class resamples.")
        idx = rng.integers(0, n, size=n)
        if len(set(y_test[idx].tolist())) < 2:
            continue
        aucs.append(roc_auc_score(y_test[idx], scores[idx]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def paired_bootstrap_delta(
    y: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> Tuple[float, float, float]:
    """(delta, lo, hi) for AUC(a) - AUC(b) on the SAME rows, resampled jointly."""
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y).astype(bool)
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    point = float(roc_auc_score(y, a) - roc_auc_score(y, b))
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas: List[float] = []
    attempts = 0
    while len(deltas) < n_bootstrap:
        attempts += 1
        if attempts > 50 * n_bootstrap:  # pragma: no cover - defensive
            raise SystemExit("HALT: paired bootstrap could not draw two-class resamples.")
        idx = rng.integers(0, n, size=n)
        if len(set(y[idx].tolist())) < 2:
            continue
        deltas.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    return point, float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def honest_probe_for_cut(
    cut: Cut,
    c_grid: Sequence[float] = C_GRID,
    n_splits: int = CV_SPLITS,
    n_repeats: int = CV_REPEATS,
    seed: int = SEED,
    n_bootstrap: int = N_BOOTSTRAP,
) -> dict:
    """Honest single-layer and concat results for one cut, plus the old number."""
    from sklearn.metrics import roc_auc_score

    y_train = cut.labels[cut.train_idx]
    y_test = cut.labels[cut.test_idx]
    groups_train = cut.problem_ids[cut.train_idx]

    train_sets = feature_sets(cut, cut.train_idx)
    test_sets = feature_sets(cut, cut.test_idx)
    single = {k: v for k, v in train_sets.items() if k.startswith("layer_")}
    concat_key = next(k for k in train_sets if k.startswith("concat_"))

    # --- honest selection over the three single layers ---------------------
    sel = select_by_cv(single, y_train, groups_train, c_grid, n_splits, n_repeats, seed)
    scores = fit_score(train_sets[sel.name], y_train, test_sets[sel.name], sel.C, seed)
    auc = float(roc_auc_score(y_test, scores))
    lo, hi = bootstrap_ci(y_test, scores, n_bootstrap, seed)

    # --- honest selection of C for the concatenated variant ----------------
    sel_c = select_by_cv(
        {concat_key: train_sets[concat_key]}, y_train, groups_train,
        c_grid, n_splits, n_repeats, seed,
    )
    scores_c = fit_score(train_sets[concat_key], y_train, test_sets[concat_key], sel_c.C, seed)
    auc_c = float(roc_auc_score(y_test, scores_c))
    lo_c, hi_c = bootstrap_ci(y_test, scores_c, n_bootstrap, seed)

    # --- contrast: what was reported, and the full test-selected ceiling ---
    old_layer = cut.results.get("best_layer")
    old_auc = cut.results.get("per_layer", {}).get(old_layer, {}).get("auc")
    per_layer_test = {}
    ceiling = -1.0
    ceiling_at = None
    for name in sorted(train_sets):
        best_here = -1.0
        for C in c_grid:
            a = float(roc_auc_score(y_test, fit_score(
                train_sets[name], y_train, test_sets[name], C, seed)))
            if a > best_here:
                best_here = a
            if a > ceiling:
                ceiling, ceiling_at = a, {"features": name, "C": float(C)}
        per_layer_test[name] = round(best_here, 4)

    return {
        "k": cut.k,
        "n_train": int(len(cut.train_idx)),
        "n_test": int(len(cut.test_idx)),
        "n_test_pos": int(y_test.sum()),
        "n_test_neg": int((~y_test).sum()),
        "honest": {
            "selected_features": sel.name,
            "selected_C": sel.C,
            "cv_auc_train": round(sel.cv_auc, 4),
            "test_auc": round(auc, 4),
            "test_auc_ci95": [round(lo, 4), round(hi, 4)],
        },
        "honest_concat": {
            "selected_features": concat_key,
            "selected_C": sel_c.C,
            "cv_auc_train": round(sel_c.cv_auc, 4),
            "test_auc": round(auc_c, 4),
            "test_auc_ci95": [round(lo_c, 4), round(hi_c, 4)],
        },
        "reported_test_selected": {
            "best_layer": old_layer,
            "auc": old_auc,
            "auc_ci95": cut.results.get("per_layer", {}).get(old_layer, {}).get("auc_ci95"),
            "C": cut.results.get("config", {}).get("probe_C"),
            "how_selected": "argmax of per-layer AUC on the TEST set, C fixed at 1.0 (never swept)",
        },
        "test_selected_ceiling": {
            "auc": round(ceiling, 4),
            "at": ceiling_at,
            "best_per_feature_set": per_layer_test,
            "note": "upper bound if BOTH layer and C were tuned on test; reported for contrast only",
        },
        "cv_table": sel.cv_table | sel_c.cv_table,
        "_test_scores": scores.tolist(),
        "_test_problem_ids": [str(p) for p in cut.problem_ids[cut.test_idx]],
        "_test_labels": y_test.tolist(),
    }


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def run(
    ks: Sequence[int] = K_VALUES,
    block_dir: Path = BLOCK2_DIR,
    n_bootstrap: int = N_BOOTSTRAP,
    n_repeats: int = CV_REPEATS,
) -> dict:
    t0 = time.time()
    per_k: Dict[str, dict] = {}
    scores_by_k: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for k in ks:
        cut = load_cut(k, block_dir)
        res = honest_probe_for_cut(cut, n_bootstrap=n_bootstrap, n_repeats=n_repeats)
        scores_by_k[k] = (
            np.asarray(res.pop("_test_labels"), dtype=bool),
            np.asarray(res.pop("_test_scores"), dtype=float),
        )
        res.pop("_test_problem_ids", None)
        per_k[str(k)] = res
        h = res["honest"]
        print(
            f"  k={k:>2}  honest {h['selected_features']:>16s} C={h['selected_C']:<10.3g} "
            f"AUC={h['test_auc']:.4f} [{h['test_auc_ci95'][0]:.3f},{h['test_auc_ci95'][1]:.3f}]"
            f"   concat AUC={res['honest_concat']['test_auc']:.4f}"
            f"   (reported {res['reported_test_selected']['auc']})"
        )

    # --- does the reported non-monotonicity survive? -----------------------
    nonmono = {}
    pairs = [(10, 25), (50, 25), (10, 50)]
    for a, b in pairs:
        if a in scores_by_k and b in scores_by_k:
            y_a, s_a = scores_by_k[a]
            y_b, s_b = scores_by_k[b]
            if not np.array_equal(y_a, y_b):  # pragma: no cover - defensive
                raise RuntimeError("test label vectors differ between cuts; cannot pair.")
            d, lo, hi = paired_bootstrap_delta(y_a, s_a, s_b, n_bootstrap, SEED)
            nonmono[f"k{a}_minus_k{b}"] = {
                "delta_auc": round(d, 4),
                "ci95": [round(lo, 4), round(hi, 4)],
                "excludes_zero": bool(lo > 0 or hi < 0),
            }

    honest_curve = [per_k[str(k)]["honest"]["test_auc"] for k in ks if str(k) in per_k]
    reported_curve = [per_k[str(k)]["reported_test_selected"]["auc"] for k in ks if str(k) in per_k]

    out = {
        "schema_version": 1,
        "analysis": "honest_probe",
        "run": "010",
        "metric": "roc_auc",
        "what_this_fixes": [
            "CONFIG.probe_C = 1.0 was never swept",
            "best_layer was argmax of AUC on the TEST set (selection leakage)",
        ],
        "protocol": {
            "split": "REUSED VERBATIM from each cut's results.json (split.train_problem_ids / test_problem_ids)",
            "selection": "StratifiedGroupKFold on problem_id, inside the training split only",
            "cv_splits": CV_SPLITS,
            "cv_repeats": n_repeats,
            "cv_statistic": "mean over repeats of pooled out-of-fold ROC-AUC",
            "C_grid": [float(c) for c in C_GRID],
            "candidates": ["layer_9", "layer_18", "layer_27", "concat_9_18_27"],
            "estimator": "StandardScaler -> LogisticRegression(max_iter=5000)",
            "tie_break": "smaller C, then feature-set name",
            "n_bootstrap": n_bootstrap,
            "seed": SEED,
        },
        "per_k": per_k,
        "nonmonotonicity_check": {
            "reported_curve_k10_k25_k50": [
                per_k.get("10", {}).get("reported_test_selected", {}).get("auc"),
                per_k.get("25", {}).get("reported_test_selected", {}).get("auc"),
                per_k.get("50", {}).get("reported_test_selected", {}).get("auc"),
            ],
            "honest_curve_k10_k25_k50": [
                per_k.get("10", {}).get("honest", {}).get("test_auc"),
                per_k.get("25", {}).get("honest", {}).get("test_auc"),
                per_k.get("50", {}).get("honest", {}).get("test_auc"),
            ],
            "paired_bootstrap": nonmono,
        },
        "summary": {
            "honest_curve": honest_curve,
            "reported_curve": reported_curve,
            "honest_range": [min(honest_curve), max(honest_curve)] if honest_curve else None,
            "reported_range": [min(reported_curve), max(reported_curve)] if reported_curve else None,
        },
        "package_versions": _package_versions(),
        "elapsed_s": round(time.time() - t0, 1),
    }
    return out


def _package_versions() -> Dict[str, str]:
    import sklearn

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
    }


def _print_table(out: dict) -> None:
    print()
    print("HONEST PROBE (Run 010) — selection by CV inside the training split only")
    print(
        f"{'k':>4} {'honest AUC':>11} {'CI95':>18} {'layer':>10} {'C':>9} "
        f"{'concat AUC':>11} | {'reported':>9} {'(test-sel layer)':>17} {'test ceiling':>13}"
    )
    for k in K_VALUES:
        r = out["per_k"].get(str(k))
        if not r:
            continue
        h, c, o = r["honest"], r["honest_concat"], r["reported_test_selected"]
        ci = f"[{h['test_auc_ci95'][0]:.3f}, {h['test_auc_ci95'][1]:.3f}]"
        print(
            f"{k:>4} {h['test_auc']:>11.4f} {ci:>18} "
            f"{h['selected_features'].replace('layer_', 'L'):>10} {h['selected_C']:>9.3g} "
            f"{c['test_auc']:>11.4f} | {o['auc']:>9} "
            f"{str(o['best_layer']).replace('layer_', 'L'):>17} "
            f"{r['test_selected_ceiling']['auc']:>13.4f}"
        )
    nm = out["nonmonotonicity_check"]
    print()
    print(f"  reported k10/k25/k50: {nm['reported_curve_k10_k25_k50']}")
    print(f"  honest   k10/k25/k50: {nm['honest_curve_k10_k25_k50']}")
    for name, d in nm["paired_bootstrap"].items():
        print(
            f"  paired {name}: {d['delta_auc']:+.4f} "
            f"[{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] "
            f"{'EXCLUDES 0' if d['excludes_zero'] else 'includes 0'}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--block-dir", default=str(BLOCK2_DIR))
    ap.add_argument("--out", default=None)
    ap.add_argument("--k", nargs="*", type=int, default=list(K_VALUES))
    ap.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    ap.add_argument("--cv-repeats", type=int, default=CV_REPEATS)
    args = ap.parse_args(argv)

    block_dir = Path(args.block_dir)
    out_path = Path(args.out) if args.out else block_dir / "honest_probe.json"
    out = run(args.k, block_dir, args.n_bootstrap, args.cv_repeats)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    _print_table(out)
    print(f"\nhonest_probe: -> {out_path}  ({out['elapsed_s']}s)")


if __name__ == "__main__":
    main()
