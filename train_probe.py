"""Stage 4: logistic probe per layer, held-out AUC vs shuffled-label floor.

Reads acts.npz, writes results.json — the Gate 1 go/no-go number.

Invariants enforced IN CODE (not just tested):
  * GroupShuffleSplit on problem_id; the train/test problem_id intersection is
    checked empty (raise, not assert) before any model is fit.
  * Metric is ROC-AUC, never accuracy (class imbalance from the model's base
    success rate lets a probe cheat).
  * Multiple-comparisons honesty: we report the BEST layer's AUC, so the
    shuffled-label floor is the distribution of the max-across-layers AUC
    under permuted train labels — same selection, same statistic.
"""

from __future__ import annotations

import dataclasses
import json
import platform
import sys
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    from experiment.config import CONFIG, lineage
except ImportError:
    from config import CONFIG, lineage


def group_split(
    problem_ids: Sequence[str],
    labels: np.ndarray,
    test_fraction: float,
    seed: int,
    max_retries: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split row indices by problem_id so no problem straddles the boundary.

    Retries split seeds until both classes appear on both sides (AUC is
    undefined otherwise); halts with a clear message if impossible.
    """
    from sklearn.model_selection import GroupShuffleSplit

    groups = np.asarray(problem_ids)
    for attempt in range(max_retries):
        gss = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed + attempt)
        train_idx, test_idx = next(gss.split(np.zeros(len(groups)), labels, groups))

        # THE invariant: a problem appears in exactly one split.
        # raise, not assert: must survive `python -O`.
        overlap = set(groups[train_idx]) & set(groups[test_idx])
        if overlap:
            raise RuntimeError(f"problem_ids in BOTH splits: {sorted(overlap)}")

        if len(set(labels[train_idx].tolist())) == 2 and len(set(labels[test_idx].tolist())) == 2:
            return train_idx, test_idx

    raise SystemExit(
        f"HALT: after {max_retries} split seeds, could not build a train/test split with both "
        f"classes on both sides (labels: {int(labels.sum())} True / {int((~labels).sum())} False "
        f"of {len(labels)}). The 20-problem sample is too skewed — regenerate with a different "
        f"seed or more problems."
    )


def fit_and_auc(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
    seed: int, C: float = CONFIG.probe_C,
) -> Tuple[float, np.ndarray]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=C, random_state=seed),
    )
    clf.fit(X_train, y_train)
    scores = clf.predict_proba(X_test)[:, 1]
    return float(roc_auc_score(y_test, scores)), scores


def bootstrap_ci(
    y_test: np.ndarray, scores: np.ndarray, n_bootstrap: int, seed: int,
    max_attempts: int | None = None,
) -> Tuple[float, float]:
    from sklearn.metrics import roc_auc_score

    if max_attempts is None:
        max_attempts = 50 * n_bootstrap  # cap: never loop forever on a skewed test set
    rng = np.random.default_rng(seed)
    aucs: List[float] = []
    n = len(y_test)
    attempts = 0
    while len(aucs) < n_bootstrap:
        attempts += 1
        if attempts > max_attempts:
            raise SystemExit(
                f"HALT: bootstrap drew {attempts - 1} resamples but only {len(aucs)} had both "
                f"classes (need {n_bootstrap}). The test set (n={n}, "
                f"{int(y_test.sum())} True) is too small/skewed for a CI — regenerate with "
                f"more problems or a different seed."
            )
        idx = rng.integers(0, n, size=n)
        if len(set(y_test[idx].tolist())) < 2:
            continue  # resample had one class; AUC undefined
        aucs.append(roc_auc_score(y_test[idx], scores[idx]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def shuffled_label_floor_max(
    Xs_train: Sequence[np.ndarray], y_train: np.ndarray,
    Xs_test: Sequence[np.ndarray], y_test: np.ndarray,
    n_seeds: int, seed: int, C: float = CONFIG.probe_C,
) -> List[float]:
    """Floor of the MAX-across-layers AUC under permuted train labels.

    Train on permuted train labels, evaluate against TRUE test labels — any
    structure found this way is leakage/chance. Because the reported statistic
    is the best layer's AUC, each floor sample applies the same selection:
    one shared permutation per seed, fit every layer, keep the max. Comparing
    a best-of-3 probe against a single-probe floor would be multiple-
    comparisons dishonesty.
    """
    floors: List[float] = []
    rng = np.random.default_rng(seed)
    for s in range(n_seeds):
        y_shuf = rng.permutation(y_train)
        if len(set(y_shuf.tolist())) < 2:
            continue
        layer_aucs = [
            fit_and_auc(X_tr, y_shuf, X_te, y_test, seed=seed + s, C=C)[0]
            for X_tr, X_te in zip(Xs_train, Xs_test)
        ]
        floors.append(max(layer_aucs))
    return floors


def verdict_for(
    best_auc: float, floor_mean: float, floor_p95: float,
    frac_floor_below: float, n_test: int,
) -> str:
    """The Gate 1 decision rule as a pure, table-testable function.

    GO       — best-layer AUC beats the max-statistic floor's p95.
    MARGINAL — above the floor mean but not its p95; tiny n_test → human call.
    NO-GO    — does not even clear the floor mean.
    """
    if best_auc > floor_p95:
        return "GO (AUC beats shuffled-label floor p95)"
    if best_auc > floor_mean:
        return (
            f"MARGINAL (AUC above floor mean, beats {frac_floor_below:.0%} of floor seeds, "
            f"but not above floor p95 — n_test={n_test} is tiny; human call)"
        )
    return "NO-GO (AUC does not exceed the shuffled-label floor mean)"


def check_min_included(n_rows: int, min_rows: int, exclusion_counts: dict | None) -> None:
    """HALT (I/O matrix row 7) rather than fit a probe on degenerate data."""
    if n_rows < min_rows:
        raise SystemExit(
            f"HALT: only {n_rows} included rows after exclusions (< {min_rows}). "
            f"Refusing to fit a probe on degenerate data. "
            f"Per-reason exclusion counts: {exclusion_counts or 'unavailable'}. "
            f"Regenerate with more problems or a larger thinking budget."
        )


def read_exclusion_counts(prefixes_path: str) -> dict | None:
    """Pull the per-reason exclusion counts from the stage-2 meta line."""
    try:
        with open(prefixes_path) as f:
            first = json.loads(f.readline())
        if first.get("record_type") == "meta":
            return first.get("exclusion_counts")
    except (OSError, json.JSONDecodeError):
        pass
    return None


def package_versions() -> Dict[str, str]:
    import sklearn

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
    }


def main() -> None:
    t0 = time.time()
    data = np.load(CONFIG.acts_path, allow_pickle=False)
    problem_ids = data["problem_ids"]
    labels = data["labels"].astype(bool)
    acts_hash = str(data["config_hash"])

    if acts_hash != CONFIG.config_hash():
        print(f"WARNING: acts.npz built under config {acts_hash}, "
              f"current is {CONFIG.config_hash()}", file=sys.stderr)

    exclusion_counts = read_exclusion_counts(CONFIG.prefixes_path)
    check_min_included(len(labels), CONFIG.min_included_rows, exclusion_counts)

    train_idx, test_idx = group_split(
        problem_ids, labels, CONFIG.test_fraction, CONFIG.seed, CONFIG.max_split_retries
    )
    y_train, y_test = labels[train_idx], labels[test_idx]

    per_layer = {}
    Xs_train, Xs_test = [], []
    for L in CONFIG.layers:
        X = data[f"acts_layer{L}"]
        X_train, X_test = X[train_idx], X[test_idx]
        Xs_train.append(X_train)
        Xs_test.append(X_test)

        auc, scores = fit_and_auc(X_train, y_train, X_test, y_test, CONFIG.seed)
        ci_low, ci_high = bootstrap_ci(y_test, scores, CONFIG.n_bootstrap, CONFIG.seed)
        per_layer[f"layer_{L}"] = {
            "auc": round(auc, 4),
            "auc_ci95": [round(ci_low, 4), round(ci_high, 4)],
        }

    best_layer = max(per_layer, key=lambda k: per_layer[k]["auc"])
    best_auc = per_layer[best_layer]["auc"]

    # One floor for the one reported statistic: max AUC across layers per
    # shuffle seed (multiple-comparisons honesty).
    floors = shuffled_label_floor_max(
        Xs_train, y_train, Xs_test, y_test, CONFIG.n_shuffle_seeds, CONFIG.seed
    )
    # Thresholds computed ONCE; verdict, results.json, and the printed line
    # all reuse these exact numbers.
    floor_mean = float(np.mean(floors))
    floor_p95 = float(np.percentile(floors, 95))
    frac_floor_below = float(np.mean([f < best_auc for f in floors]))
    verdict = verdict_for(best_auc, floor_mean, floor_p95, frac_floor_below, len(test_idx))

    results = {
        "metric": "roc_auc",
        "truncation_k_percent": CONFIG.truncation_k_percent,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "class_balance": {"true": int(labels.sum()), "false": int((~labels).sum())},
        "split": {
            "train_problem_ids": sorted({str(problem_ids[i]) for i in train_idx}),
            "test_problem_ids": sorted({str(problem_ids[i]) for i in test_idx}),
        },
        "per_layer": per_layer,
        "best_layer": best_layer,
        "shuffled_floor": {
            "statistic": "max_auc_across_layers",
            "n_seeds": len(floors),
            "mean": round(floor_mean, 4),
            "p95": round(floor_p95, 4),
            # fraction of shuffled-label seeds the real best probe beats —
            # with a tiny test set this is more informative than p95 alone
            "frac_seeds_below_best_auc": round(frac_floor_below, 4),
        },
        "verdict": verdict,
        "exclusion_counts": exclusion_counts,
        "config": dataclasses.asdict(CONFIG),
        "package_versions": package_versions(),
        "elapsed_s": round(time.time() - t0, 1),
        "lineage": {
            **lineage(CONFIG.acts_path),
            "acts_config_hash": acts_hash,
        },
    }
    with open(CONFIG.results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"train_probe: results -> {CONFIG.results_path}")
    print(f"GATE 1 RESULT: best {best_layer} AUC={best_auc} "
          f"(CI95 {per_layer[best_layer]['auc_ci95']}) vs shuffled-label floor "
          f"(max-across-layers, {len(floors)} seeds) "
          f"mean={round(floor_mean, 4)} p95={round(floor_p95, 4)} -> {verdict}")


if __name__ == "__main__":
    main()
