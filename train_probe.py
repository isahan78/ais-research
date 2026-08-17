"""Stage 4: logistic probe per layer, held-out AUC vs shuffled-label floor.

Reads acts.npz, writes results.json — the Gate 1 go/no-go number.

Invariants enforced IN CODE (not just tested):
  * GroupShuffleSplit on problem_id; the train/test problem_id intersection is
    asserted empty before any model is fit.
  * Metric is ROC-AUC, never accuracy (class imbalance from the model's base
    success rate lets a probe cheat).
"""

from __future__ import annotations

import json
import sys
import time
from typing import List, Sequence, Tuple

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
        overlap = set(groups[train_idx]) & set(groups[test_idx])
        assert not overlap, f"problem_ids in BOTH splits: {sorted(overlap)}"

        if len(set(labels[train_idx].tolist())) == 2 and len(set(labels[test_idx].tolist())) == 2:
            return train_idx, test_idx

    raise SystemExit(
        f"HALT: after {max_retries} split seeds, could not build a train/test split with both "
        f"classes on both sides (labels: {int(labels.sum())} True / {int((~labels).sum())} False "
        f"of {len(labels)}). The 20-problem sample is too skewed — regenerate with a different "
        f"seed or more problems."
    )


def fit_and_auc(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, seed: int
) -> Tuple[float, np.ndarray]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=1.0, random_state=seed),
    )
    clf.fit(X_train, y_train)
    scores = clf.predict_proba(X_test)[:, 1]
    return float(roc_auc_score(y_test, scores)), scores


def bootstrap_ci(
    y_test: np.ndarray, scores: np.ndarray, n_bootstrap: int, seed: int
) -> Tuple[float, float]:
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    aucs: List[float] = []
    n = len(y_test)
    while len(aucs) < n_bootstrap:
        idx = rng.integers(0, n, size=n)
        if len(set(y_test[idx].tolist())) < 2:
            continue  # resample had one class; AUC undefined
        aucs.append(roc_auc_score(y_test[idx], scores[idx]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def shuffled_label_floor(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
    n_seeds: int, seed: int
) -> List[float]:
    """Train on permuted train labels, evaluate against TRUE test labels.

    Any structure the probe finds under shuffled labels is leakage/chance;
    the distribution of these AUCs is the floor the real probe must beat.
    """
    floors = []
    rng = np.random.default_rng(seed)
    for s in range(n_seeds):
        y_shuf = rng.permutation(y_train)
        if len(set(y_shuf.tolist())) < 2:
            continue
        auc, _ = fit_and_auc(X_train, y_shuf, X_test, y_test, seed=seed + s)
        floors.append(auc)
    return floors


def main() -> None:
    t0 = time.time()
    data = np.load(CONFIG.acts_path, allow_pickle=False)
    problem_ids = data["problem_ids"]
    labels = data["labels"].astype(bool)
    acts_hash = str(data["config_hash"])

    if acts_hash != CONFIG.config_hash():
        print(f"WARNING: acts.npz built under config {acts_hash}, "
              f"current is {CONFIG.config_hash()}", file=sys.stderr)

    train_idx, test_idx = group_split(
        problem_ids, labels, CONFIG.test_fraction, CONFIG.seed, CONFIG.max_split_retries
    )
    y_train, y_test = labels[train_idx], labels[test_idx]

    per_layer = {}
    for L in CONFIG.layers:
        X = data[f"acts_layer{L}"]
        X_train, X_test = X[train_idx], X[test_idx]

        auc, scores = fit_and_auc(X_train, y_train, X_test, y_test, CONFIG.seed)
        ci_low, ci_high = bootstrap_ci(y_test, scores, CONFIG.n_bootstrap, CONFIG.seed)
        floors = shuffled_label_floor(
            X_train, y_train, X_test, y_test, CONFIG.n_shuffle_seeds, CONFIG.seed
        )

        per_layer[f"layer_{L}"] = {
            "auc": round(auc, 4),
            "auc_ci95": [round(ci_low, 4), round(ci_high, 4)],
            "floor_mean": round(float(np.mean(floors)), 4),
            "floor_p95": round(float(np.percentile(floors, 95)), 4),
            "floor_n_seeds": len(floors),
            # fraction of shuffled-label seeds the real probe beats — with a
            # tiny test set this is more informative than p95 alone
            "floor_frac_below_auc": round(float(np.mean([f < auc for f in floors])), 4),
            "beats_floor_p95": bool(auc > np.percentile(floors, 95)),
        }

    results = {
        "metric": "roc_auc",
        "truncation_k_percent": CONFIG.truncation_k_percent,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "class_balance": {"true": int(labels.sum()), "false": int((~labels).sum())},
        "per_layer": per_layer,
        "elapsed_s": round(time.time() - t0, 1),
        "lineage": {
            **lineage(CONFIG.acts_path),
            "acts_config_hash": acts_hash,
        },
    }
    with open(CONFIG.results_path, "w") as f:
        json.dump(results, f, indent=2)

    best_layer = max(per_layer, key=lambda k: per_layer[k]["auc"])
    best = per_layer[best_layer]
    if best["beats_floor_p95"]:
        verdict = "GO (AUC beats shuffled-label floor p95)"
    elif best["auc"] > best["floor_mean"]:
        verdict = (f"MARGINAL (AUC above floor mean, beats {best['floor_frac_below_auc']:.0%} "
                   f"of floor seeds, but not above floor p95 — n_test={len(test_idx)} is tiny; "
                   f"human call)")
    else:
        verdict = "NO-GO (AUC does not exceed the shuffled-label floor mean)"
    print(f"train_probe: results -> {CONFIG.results_path}")
    print(f"GATE 1 RESULT: best {best_layer} AUC={best['auc']} "
          f"(CI95 {best['auc_ci95']}) vs shuffled-label floor "
          f"mean={best['floor_mean']} p95={best['floor_p95']} -> {verdict}")


if __name__ == "__main__":
    main()
