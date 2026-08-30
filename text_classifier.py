"""Text baseline 3 of 3: TRAINED TEXT CLASSIFIER — TF-IDF + logistic regression.

This is the baseline most likely to be strawmanned, and the one 2507.12428
reports the probe beating by ~13 F1. An untuned bag-of-words on default
settings loses to almost anything, and every point it loses is a free point of
Δ. So it is tuned properly, and the tuning is auditable:

HOW IT IS TUNED (and why you can believe the number)
  * Features: a union of WORD n-grams (content: "wait, let me recheck",
    "therefore the answer is") and CHARACTER n-grams inside word boundaries
    (morphology and formatting the word tokenizer throws away — LaTeX
    fragments, option letters, hedging suffixes). Both sublinear-tf, both
    min_df-filtered, both capped so the fit stays honest on ~200 documents.
  * Grid: C x word n-gram range x char n-gram range x class weighting
    (`PARAM_GRID` below) — the four knobs that actually move a TF-IDF+LR.
  * Selection: `StratifiedGroupKFold` cross-validation, scored by ROC-AUC,
    run ENTIRELY INSIDE THE TRAINING SPLIT. The grid never sees a test row.
    Grouping by `problem_id` matters twice over: it keeps the CV folds honest
    when the k-grid puts several cuts of one problem in the training set, and
    it mirrors the probe's own group split exactly.
  * The winning configuration is refit on the whole training split and scored
    once on the held-out problems. One number, one look at the test set.

Information parity with the probe (EXPERIMENT.md §7) is exact: same rows, same
split read back from results.json, same prefixes — the only difference is that
the probe reads the activations and this reads the page.

CPU-only. Run after train_probe:  python -m experiment.text_classifier
"""

from __future__ import annotations

import json
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from experiment import analysis, forced_answer
    from experiment.config import CONFIG, lineage
except ImportError:  # run as a plain script from inside experiment/
    import analysis  # type: ignore
    import forced_answer  # type: ignore
    from config import CONFIG, lineage  # type: ignore


BASELINE_NAME = "text_classifier"

# --- tunables (module-level: config.py belongs to another stage) ------------

CV_FOLDS = 5
MAX_ITER = 5000
WORD_MAX_FEATURES = 50_000
CHAR_MAX_FEATURES = 50_000
WORD_MIN_DF = 2          # a term seen in one document cannot generalise
CHAR_MIN_DF = 3

# Small, deliberate grid: the knobs that matter for TF-IDF + LR, and nothing
# else. Kept small on purpose — n_train is ~200, so a large grid would start
# selecting on CV noise rather than on signal.
PARAM_GRID: Dict[str, list] = {
    "features__word__ngram_range": [(1, 1), (1, 2)],
    "features__char__ngram_range": [(2, 4), (3, 5)],
    "clf__C": [0.01, 0.1, 0.3, 1.0, 3.0, 10.0],
    "clf__class_weight": [None, "balanced"],
}


def build_pipeline(seed: int = CONFIG.seed):
    """TF-IDF(word) + TF-IDF(char_wb) -> logistic regression.

    Step names are load-bearing: `PARAM_GRID` addresses them.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, Pipeline

    return Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "word",
                            TfidfVectorizer(
                                analyzer="word",
                                ngram_range=(1, 2),
                                min_df=WORD_MIN_DF,
                                sublinear_tf=True,
                                max_features=WORD_MAX_FEATURES,
                                lowercase=True,
                            ),
                        ),
                        (
                            "char",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                ngram_range=(3, 5),
                                min_df=CHAR_MIN_DF,
                                sublinear_tf=True,
                                max_features=CHAR_MAX_FEATURES,
                                lowercase=True,
                            ),
                        ),
                    ]
                ),
            ),
            ("clf", LogisticRegression(max_iter=MAX_ITER, random_state=seed)),
        ]
    )


def n_cv_splits(y_train: Sequence[bool], groups_train: Sequence[str], folds: int = CV_FOLDS) -> int:
    """The largest usable fold count: bounded by the rarer class and by the
    number of distinct problems, because folds must be group-disjoint AND
    contain both classes for ROC-AUC to be defined."""
    y = np.asarray(y_train, dtype=bool)
    n_min_class = int(min(y.sum(), (~y).sum()))
    n_groups = len(set(groups_train))
    return max(2, min(folds, n_min_class, n_groups))


def _cv_splitter(n_splits: int, seed: int):
    try:
        from sklearn.model_selection import StratifiedGroupKFold

        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    except ImportError:  # pragma: no cover - very old sklearn
        from sklearn.model_selection import GroupKFold

        return GroupKFold(n_splits=n_splits)


def tune(
    texts_train: Sequence[str],
    y_train: Sequence[bool],
    groups_train: Sequence[str],
    seed: int = CONFIG.seed,
    param_grid: Optional[Dict[str, list]] = None,
    folds: int = CV_FOLDS,
    n_jobs: int = -1,
):
    """Grid-search the classifier INSIDE the training split. Returns the
    fitted `GridSearchCV`.

    Structural honesty: this function is only ever handed training arrays. It
    has no parameter through which a test row could reach it, so "did the grid
    search see the test set?" is answerable by reading the signature, not by
    trusting the caller. `tests/test_baselines.py` also checks it empirically,
    by poisoning the test rows with a unique token and asserting that token
    never appears in the fitted vocabulary.
    """
    from sklearn.model_selection import GridSearchCV

    y = np.asarray(y_train, dtype=bool)
    if len(set(y.tolist())) < 2:
        raise SystemExit(
            f"HALT: the training split has a single class ({int(y.sum())} True of "
            f"{len(y)}) — a text classifier cannot be fit."
        )
    n_splits = n_cv_splits(y, groups_train, folds)
    search = GridSearchCV(
        estimator=build_pipeline(seed),
        param_grid=param_grid if param_grid is not None else PARAM_GRID,
        scoring="roc_auc",
        cv=_cv_splitter(n_splits, seed),
        refit=True,          # winner is refit on the FULL training split
        n_jobs=n_jobs,
        error_score="raise",
    )
    search.fit(list(texts_train), y, groups=np.asarray(list(groups_train)))
    return search


def fit_and_score(
    texts: Sequence[str],
    labels: Sequence[bool],
    groups: Sequence[str],
    train_idx: Sequence[int],
    test_idx: Sequence[int],
    seed: int = CONFIG.seed,
    param_grid: Optional[Dict[str, list]] = None,
    folds: int = CV_FOLDS,
    n_jobs: int = -1,
) -> Tuple[np.ndarray, dict]:
    """(test_scores, tuning_report). Slices train/test HERE, so `tune` cannot
    be handed a test row even by accident."""
    texts = list(texts)
    y = np.asarray(labels, dtype=bool)
    train_idx = list(train_idx)
    test_idx = list(test_idx)
    overlap = set(train_idx) & set(test_idx)
    if overlap:
        raise RuntimeError(f"train/test row indices overlap: {sorted(overlap)[:10]}")

    search = tune(
        [texts[i] for i in train_idx],
        y[train_idx],
        [groups[i] for i in train_idx],
        seed=seed,
        param_grid=param_grid,
        folds=folds,
        n_jobs=n_jobs,
    )
    scores = search.best_estimator_.predict_proba([texts[i] for i in test_idx])[:, 1]
    report = {
        "best_params": {k: list(v) if isinstance(v, tuple) else v
                        for k, v in search.best_params_.items()},
        "best_cv_auc": round(float(search.best_score_), 4),
        "cv_folds": int(search.cv.get_n_splits()) if hasattr(search.cv, "get_n_splits") else None,
        "grid_size": int(len(search.cv_results_["params"])),
        "cv_auc_range": [
            round(float(np.nanmin(search.cv_results_["mean_test_score"])), 4),
            round(float(np.nanmax(search.cv_results_["mean_test_score"])), 4),
        ],
        "n_train": len(train_idx),
        "tuning_scope": "cross-validation within the training split only",
    }
    return scores, report


# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()
    try:
        with open(CONFIG.results_path) as f:
            results = json.load(f)
    except OSError:
        raise SystemExit(
            "HALT: results.json not found — run train_probe first; the text "
            "classifier must train and test on the probe's exact problem split."
        )
    try:
        from experiment.text_floor import split_indices_from_results
    except ImportError:
        from text_floor import split_indices_from_results  # type: ignore

    rows, default_k = forced_answer.load_included_prefix_rows()
    if not rows:
        raise SystemExit("HALT: no included rows in prefixes.jsonl.")

    per_k: Dict[str, dict] = {}
    for k, krows in sorted(forced_answer.group_rows_by_k(rows, default_k).items()):
        texts = forced_answer.load_prefix_texts(krows, k)
        labels = [bool(r["label"]) for r in krows]
        groups = [r["problem_id"] for r in krows]
        train_idx, test_idx = split_indices_from_results(groups, results)
        if len(test_idx) == 0:
            print(f"WARNING: k={k}% has no test rows; skipping.", file=sys.stderr)
            continue

        scores, report = fit_and_score(texts, labels, groups, train_idx, test_idx)
        point = analysis.score_at_k(
            [labels[i] for i in test_idx],
            scores,
            row_keys=[forced_answer.row_key(krows[i], k) for i in test_idx],
        )
        point["notes"] = report
        per_k[str(k)] = point
        print(f"text_classifier[k={k}%]: best CV AUC={report['best_cv_auc']} "
              f"over {report['grid_size']} configs "
              f"({report['cv_folds']}-fold, train only) -> best_params={report['best_params']}")

    if not per_k:
        raise SystemExit("HALT: no scorable k.")

    path = analysis.write_baseline_json(
        BASELINE_NAME,
        per_k,
        notes={
            "description": "TF-IDF word+char n-grams -> logistic regression, "
                           "grid-searched by grouped CV inside the training split",
            "param_grid": {k: [list(v) if isinstance(v, tuple) else v for v in vs]
                           for k, vs in PARAM_GRID.items()},
            **lineage(CONFIG.prefixes_path),
        },
    )
    for k, p in sorted(per_k.items(), key=lambda kv: int(kv[0])):
        print(f"text_classifier[k={k}%]: AUC={p['auc']} CI95={p['auc_ci95']} "
              f"(n_test={p['n_test']}) -> {path}")
    print(f"text_classifier: done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
