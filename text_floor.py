"""Stage 5: crude text-side floor — Gate 1 must ask the real question.

Beating shuffled noise only proves the pipeline works; the project's question
is whether the probe beats what the TEXT already gives away. This is the
deliberately crude Gate 1 version (decision C): a logistic regression on two
scalars any spectator can read off the prefix without a GPU —

    (prefix token count, per-problem difficulty scalar)

where the difficulty scalar is the MATH-500 `level` when the dataset has one
and the prompt length in tokens otherwise (MMLU-Pro has no difficulty field;
prompt length is the only per-problem difficulty proxy readable off the page
without a GPU).

evaluated on the SAME train/test problem split the probe used (read back from
results.json, so the two numbers are computed on identical held-out problems).
The three real text baselines (LLM judge, trained text classifier,
forced-answer) are deferred work — do not grow this file into them.

Reads prefixes.jsonl (features/labels), traces.jsonl (problem level), and
results.json (the probe's split); writes the text-floor block back into
results.json. No GPU, no torch.
"""

from __future__ import annotations

import json
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    from experiment.config import CONFIG, lineage
    from experiment.generate_traces import parse_level
    from experiment.train_probe import bootstrap_ci, fit_and_auc
except ImportError:
    from config import CONFIG, lineage
    from generate_traces import parse_level
    from train_probe import bootstrap_ci, fit_and_auc


def build_feature_rows(
    prefix_rows: Sequence[dict], level_by_pid: Dict[str, float]
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """(prefix token count, per-problem difficulty scalar) per included row.

    Pure and CPU-only so the feature contract is unit-testable. Raises on a
    missing scalar rather than silently imputing one.
    """
    pids: List[str] = []
    feats: List[List[float]] = []
    labels: List[bool] = []
    for r in prefix_rows:
        pid = r["problem_id"]
        if pid not in level_by_pid or level_by_pid[pid] is None:
            raise RuntimeError(
                f"{pid}: no problem level/difficulty scalar found in traces.jsonl "
                f"— cannot build text floor"
            )
        pids.append(pid)
        feats.append([float(len(r["prefix_token_ids"])), float(level_by_pid[pid])])
        labels.append(bool(r["label"]))
    return pids, np.array(feats, dtype=np.float64), np.array(labels, dtype=bool)


def problem_difficulty_scalars(traces_path: str) -> Tuple[Dict[str, float], str]:
    """Per-problem difficulty scalar for the crude floor, plus its feature name.

    MATH-500 rows carry `level`; MMLU-Pro rows do not (level is None). Rather
    than dropping the second feature — which would quietly weaken the floor the
    probe has to beat, i.e. inflate the headline gap — fall back to the prompt
    length in tokens: still GPU-free, still readable off the page, still
    constant per problem.
    """
    scalars: Dict[str, float] = {}
    name = "problem_level"
    with open(traces_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("record_type") == "meta":
                continue
            level = parse_level(rec["level"]) if rec.get("level") is not None else None
            if level is None:
                name = "prompt_token_count"
                level = len(rec.get("prompt_token_ids") or []) or None
            scalars[rec["problem_id"]] = None if level is None else float(level)
    return scalars, name


# Back-compat alias: earlier callers imported load_levels.
def load_levels(traces_path: str) -> Dict[str, float]:
    return problem_difficulty_scalars(traces_path)[0]


def split_indices_from_results(
    pids: Sequence[str], results: dict
) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruct the probe's exact row split from the recorded problem ids."""
    train_pids = set(results["split"]["train_problem_ids"])
    test_pids = set(results["split"]["test_problem_ids"])
    train_idx = np.array([i for i, p in enumerate(pids) if p in train_pids], dtype=int)
    test_idx = np.array([i for i, p in enumerate(pids) if p in test_pids], dtype=int)
    missing = [p for p in pids if p not in train_pids and p not in test_pids]
    if missing:
        raise RuntimeError(
            f"problem_ids absent from the probe's recorded split: {sorted(set(missing))} — "
            f"results.json and prefixes.jsonl are out of sync; rerun train_probe."
        )
    return train_idx, test_idx


def main() -> None:
    t0 = time.time()
    try:
        with open(CONFIG.results_path) as f:
            results = json.load(f)
    except OSError:
        raise SystemExit(
            "HALT: results.json not found — run train_probe first; the text floor "
            "must reuse the probe's exact train/test split."
        )

    prefix_rows = []
    with open(CONFIG.prefixes_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("record_type") != "meta" and rec["included"]:
                prefix_rows.append(rec)

    level_by_pid, difficulty_feature = problem_difficulty_scalars(CONFIG.traces_path)
    pids, X, y = build_feature_rows(prefix_rows, level_by_pid)
    train_idx, test_idx = split_indices_from_results(pids, results)

    auc, scores = fit_and_auc(X[train_idx], y[train_idx], X[test_idx], y[test_idx], CONFIG.seed)
    ci_low, ci_high = bootstrap_ci(y[test_idx], scores, CONFIG.n_bootstrap, CONFIG.seed)

    results["text_floor"] = {
        "features": ["prefix_token_count", difficulty_feature],
        "auc": round(auc, 4),
        "auc_ci95": [round(ci_low, 4), round(ci_high, 4)],
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "lineage": lineage(CONFIG.prefixes_path),
    }
    with open(CONFIG.results_path, "w") as f:
        json.dump(results, f, indent=2)

    best = results["per_layer"][results["best_layer"]]
    print(f"text_floor: AUC={results['text_floor']['auc']} "
          f"CI95={results['text_floor']['auc_ci95']} "
          f"(features: prefix token count + {difficulty_feature}) -> {CONFIG.results_path} "
          f"({time.time() - t0:.1f}s)")
    print(f"text_floor: probe best {results['best_layer']} AUC={best['auc']} vs "
          f"text floor AUC={results['text_floor']['auc']} — "
          f"{'probe above crude text floor' if best['auc'] > auc else 'probe does NOT beat even the crude text floor'}")


if __name__ == "__main__":
    main()
