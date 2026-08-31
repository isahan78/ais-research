"""Run 010b — length-controlled reader comparison.

Run 007 found `corr(prefix thinking tokens at k, FULL trace thinking tokens)
= 0.99999999`: cutting at a fixed *fraction* of the trace hands every reader
the trace's eventual length, which no real-time monitor could know. Prefix
length alone scores ~0.70.

This script asks how much of each reader's AUC is that leak, using only
committed artifacts:

  * Within-stratum AUC — the TEST rows are cut into length terciles (and
    quartiles) by prefix thinking-token count, and every reader is re-scored
    inside each stratum, where length is nearly constant.
  * A stratum-pooled AUC — the Mantel-Haenszel-style estimator that pools
    only *within-stratum* concordant pairs:
        pooled = sum_s AUC_s * n_pos_s * n_neg_s / sum_s n_pos_s * n_neg_s
    i.e. the probability a random correct trace outscores a random incorrect
    one *of comparable length*. A simple unweighted mean is reported too.
  * Partial correlation of each reader's score with the label, controlling
    for log prefix length (Pearson on residuals, plus the rank version).

Readers
  probe_honest  — the honestly-selected layer + C from honest_probe.json
                  (Run 010), refit on the training split only.
  tfidf         — the tuned TF-IDF classifier's STORED test scores
                  (outputs/block2/k*/baseline_text_classifier.json).
  crude_floor   — the Gate-1 floor: logistic on (prefix token count, prompt
                  token count), refit on the training split only.
  length_only   — the reference: 1 feature, log1p(prefix thinking tokens),
                  fit on the training split only.

Split reuse and train-only fitting are enforced by `honest_probe.load_cut` /
`split_indices_from_results`; nothing here touches a test row before scoring.
CPU only, no network, no new data.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

try:
    from experiment.honest_probe import (
        BLOCK2_DIR, K_VALUES, SEED, fit_score, load_cut,
        split_indices_from_results,
    )
except ImportError:  # running as a script from inside experiment/
    from honest_probe import (  # type: ignore
        BLOCK2_DIR, K_VALUES, SEED, fit_score, load_cut,
        split_indices_from_results,
    )

N_BOOTSTRAP = 2000
READER_ORDER = ("probe_honest", "tfidf", "crude_floor", "length_only")


# --------------------------------------------------------------------------
# stratification
# --------------------------------------------------------------------------


def quantile_strata(values: Sequence[float], n_strata: int) -> np.ndarray:
    """Assign each row to a length stratum, 0 = shortest.

    Rank-based so ties never straddle a boundary: rows are ordered by value
    (ties broken by first appearance) and cut at equal counts. Every row lands
    in exactly one stratum; equal values always land together only if that
    keeps bins non-empty, so the result is always ``n_strata`` contiguous,
    near-equal, length-monotone bins.
    """
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n_strata < 1:
        raise ValueError("n_strata must be >= 1")
    if n == 0:
        return np.zeros(0, dtype=int)
    if n_strata == 1:
        return np.zeros(n, dtype=int)

    order = np.argsort(v, kind="stable")
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(n)
    # equal-count cut on ranks: floor(rank * n_strata / n)
    strata = (ranks * n_strata) // n
    return strata.astype(int)


def stratum_bounds(values: Sequence[float], strata: np.ndarray) -> List[List[float]]:
    v = np.asarray(values, dtype=float)
    out = []
    for g in range(int(strata.max()) + 1) if len(strata) else []:
        m = strata == g
        out.append([float(v[m].min()), float(v[m].max())] if m.any() else [float("nan")] * 2)
    return out


# --------------------------------------------------------------------------
# AUC machinery
# --------------------------------------------------------------------------


def auc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y).astype(bool)
    if len(set(y.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y, np.asarray(s, dtype=float)))


def stratified_auc(y: np.ndarray, s: np.ndarray, strata: np.ndarray) -> Tuple[float, float, dict]:
    """(pair-weighted pooled AUC, unweighted mean AUC, per-stratum detail).

    Pair-weighted pooling is the Mantel-Haenszel analogue for the AUC: each
    stratum contributes its concordance over its own n_pos * n_neg pairs, so
    the pooled number is a genuine within-length concordance probability.
    Degenerate strata (one class only) contribute nothing and are recorded.
    """
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=float)
    strata = np.asarray(strata, dtype=int)
    num = 0.0
    den = 0.0
    per: Dict[str, dict] = {}
    plain: List[float] = []
    for g in sorted(set(strata.tolist())):
        m = strata == g
        yg, sg = y[m], s[m]
        n_pos, n_neg = int(yg.sum()), int((~yg).sum())
        a = auc(yg, sg)
        per[str(g)] = {
            "n": int(m.sum()),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "auc": None if np.isnan(a) else round(a, 4),
        }
        if n_pos and n_neg:
            num += a * n_pos * n_neg
            den += n_pos * n_neg
            plain.append(a)
    pooled = float(num / den) if den else float("nan")
    return pooled, (float(np.mean(plain)) if plain else float("nan")), per


def _stratified_resample(strata: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Resample row indices with replacement WITHIN each stratum."""
    idx = []
    for g in sorted(set(strata.tolist())):
        rows = np.flatnonzero(strata == g)
        idx.append(rng.integers(0, len(rows), size=len(rows)).astype(int))
        idx[-1] = rows[idx[-1]]
    return np.concatenate(idx) if idx else np.zeros(0, dtype=int)


def pooled_ci(
    y: np.ndarray, s: np.ndarray, strata: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP, seed: int = SEED,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals: List[float] = []
    for _ in range(n_bootstrap):
        idx = _stratified_resample(strata, rng)
        p, _m, _d = stratified_auc(y[idx], s[idx], strata[idx])
        if not np.isnan(p):
            vals.append(p)
    if len(vals) < 10:  # pragma: no cover - defensive
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def pooled_delta_ci(
    y: np.ndarray, s_a: np.ndarray, s_b: np.ndarray, strata: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP, seed: int = SEED,
) -> Tuple[float, float, float]:
    """Paired stratified bootstrap for pooled AUC(a) - pooled AUC(b)."""
    point = stratified_auc(y, s_a, strata)[0] - stratified_auc(y, s_b, strata)[0]
    rng = np.random.default_rng(seed)
    vals: List[float] = []
    for _ in range(n_bootstrap):
        idx = _stratified_resample(strata, rng)
        pa = stratified_auc(y[idx], s_a[idx], strata[idx])[0]
        pb = stratified_auc(y[idx], s_b[idx], strata[idx])[0]
        if not (np.isnan(pa) or np.isnan(pb)):
            vals.append(pa - pb)
    if len(vals) < 10:  # pragma: no cover - defensive
        return float(point), float("nan"), float("nan")
    return float(point), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# --------------------------------------------------------------------------
# partial correlation
# --------------------------------------------------------------------------


def _residualize(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    Z = np.column_stack([np.ones(len(z)), np.asarray(z, dtype=float)])
    beta, *_ = np.linalg.lstsq(Z, np.asarray(x, dtype=float), rcond=None)
    return np.asarray(x, dtype=float) - Z @ beta


def partial_corr(x: Sequence[float], y: Sequence[float], z: Sequence[float]) -> float:
    """Pearson corr(x, y) with a linear effect of z removed from both."""
    rx, ry = _residualize(np.asarray(x, float), z), _residualize(np.asarray(y, float), z)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _rank(a: Sequence[float]) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="stable")
    r = np.empty(len(a), dtype=float)
    r[order] = np.arange(len(a), dtype=float)
    # average ties
    uniq, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(uniq))
    np.add.at(sums, inv, r)
    return (sums / counts)[inv]


def partial_spearman(x: Sequence[float], y: Sequence[float], z: Sequence[float]) -> float:
    return partial_corr(_rank(x), _rank(y), _rank(z))


# --------------------------------------------------------------------------
# reader scores
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CutRows:
    k: int
    test_pids: List[str]
    y_test: np.ndarray
    prefix_thinking_tokens: np.ndarray   # the stratification variable
    prefix_total_tokens: np.ndarray
    full_thinking_tokens: np.ndarray
    scores: Dict[str, np.ndarray]
    meta: dict


def read_prefix_rows(path: Path) -> Dict[str, dict]:
    rows: Dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("record_type") == "meta" or not r.get("included"):
                continue
            rows[str(r["problem_id"])] = r
    return rows


def build_cut_rows(
    k: int,
    block_dir: Path,
    honest_sel: Mapping[str, object] | None,
    seed: int = SEED,
) -> CutRows:
    cut_dir = Path(block_dir) / f"k{k}"
    cut = load_cut(k, block_dir)
    test_pids = [str(p) for p in cut.problem_ids[cut.test_idx]]
    y_test = cut.labels[cut.test_idx]
    y_train = cut.labels[cut.train_idx]

    pref = read_prefix_rows(cut_dir / "prefixes.jsonl")
    all_pids = [str(p) for p in cut.problem_ids]
    missing = [p for p in all_pids if p not in pref]
    if missing:
        raise RuntimeError(f"k{k}: prefixes.jsonl missing rows for {missing[:5]}")
    for p in all_pids:
        if bool(pref[p]["label"]) != bool(cut.labels[all_pids.index(p)]):
            raise RuntimeError(f"k{k}: label mismatch between prefixes.jsonl and acts.npz at {p}")

    def col(pids, fn):
        return np.array([fn(pref[p]) for p in pids], dtype=float)

    scores: Dict[str, np.ndarray] = {}

    # --- probe, honestly selected in Run 010, refit on train only ----------
    if honest_sel is not None:
        name = str(honest_sel["selected_features"])
        C = float(honest_sel["selected_C"])
        if name.startswith("concat"):
            X = np.concatenate([cut.layer_acts[L] for L in sorted(cut.layer_acts)], axis=1)
        else:
            X = cut.layer_acts[int(name.split("_")[1])]
        scores["probe_honest"] = fit_score(
            X[cut.train_idx], y_train, X[cut.test_idx], C, seed=seed
        )

    # --- tuned TF-IDF: STORED scores, realigned to our test ordering -------
    with open(cut_dir / "baseline_text_classifier.json") as f:
        tf = json.load(f)
    block = tf["per_k"][str(k)]
    by_pid = {}
    for key, sc, lab in zip(block["test_row_keys"], block["test_scores"], block["test_labels"]):
        by_pid[str(key).split("@")[0]] = (float(sc), bool(lab))
    if set(by_pid) != set(test_pids):
        raise RuntimeError(f"k{k}: TF-IDF test rows do not match the recorded test split")
    for p, yv in zip(test_pids, y_test):
        if by_pid[p][1] != bool(yv):
            raise RuntimeError(f"k{k}: TF-IDF label disagrees with acts.npz at {p}")
    scores["tfidf"] = np.array([by_pid[p][0] for p in test_pids], dtype=float)

    # --- crude Gate-1 floor: refit here (its per-row scores were not stored)
    tr_pids = [str(p) for p in cut.problem_ids[cut.train_idx]]
    f_crude = lambda r: [float(len(r["prefix_token_ids"])), float(len(r["prompt_token_ids"]))]
    X_cr_tr = np.array([f_crude(pref[p]) for p in tr_pids], dtype=float)
    X_cr_te = np.array([f_crude(pref[p]) for p in test_pids], dtype=float)
    scores["crude_floor"] = fit_score(X_cr_tr, y_train, X_cr_te, C=1.0, seed=seed)

    # --- length-only reference: 1 feature, train-fitted --------------------
    f_len = lambda r: [float(np.log1p(r["n_kept_thinking_tokens"]))]
    X_len_tr = np.array([f_len(pref[p]) for p in tr_pids], dtype=float)
    X_len_te = np.array([f_len(pref[p]) for p in test_pids], dtype=float)
    scores["length_only"] = fit_score(X_len_tr, y_train, X_len_te, C=1.0, seed=seed)

    prefix_thinking = col(test_pids, lambda r: r["n_kept_thinking_tokens"])
    prefix_total = col(test_pids, lambda r: len(r["prefix_token_ids"]))
    full_thinking = col(test_pids, lambda r: r["n_thinking_tokens"])
    all_pref_th = col(all_pids, lambda r: r["n_kept_thinking_tokens"])
    all_full_th = col(all_pids, lambda r: r["n_thinking_tokens"])

    meta = {
        "corr_prefix_vs_full_thinking_tokens_all_rows": round(
            float(np.corrcoef(all_pref_th, all_full_th)[0, 1]), 10
        ),
        "corr_prefix_vs_full_thinking_tokens_test_rows": round(
            float(np.corrcoef(prefix_thinking, full_thinking)[0, 1]), 10
        ),
        "probe_selection": dict(honest_sel) if honest_sel else None,
    }
    return CutRows(
        k=k, test_pids=test_pids, y_test=y_test,
        prefix_thinking_tokens=prefix_thinking,
        prefix_total_tokens=prefix_total,
        full_thinking_tokens=full_thinking,
        scores=scores, meta=meta,
    )


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def analyse_cut(
    rows: CutRows, strata_counts: Sequence[int] = (3, 4),
    n_bootstrap: int = N_BOOTSTRAP, seed: int = SEED,
) -> dict:
    y = rows.y_test
    loglen = np.log1p(rows.prefix_thinking_tokens)
    out: dict = {
        "k": rows.k,
        "n_test": int(len(y)),
        "n_pos": int(y.sum()),
        "n_neg": int((~y).sum()),
        "length_variable": "n_kept_thinking_tokens (prefix thinking tokens)",
        "length_range": [float(rows.prefix_thinking_tokens.min()),
                         float(rows.prefix_thinking_tokens.max())],
        **rows.meta,
        "readers": {},
        "strata": {},
    }

    for name in READER_ORDER:
        if name not in rows.scores:
            continue
        s = rows.scores[name]
        out["readers"][name] = {
            "auc_unstratified": round(auc(y, s), 4),
            "corr_score_vs_loglength": round(float(np.corrcoef(s, loglen)[0, 1]), 4),
            "corr_score_vs_label": round(float(np.corrcoef(s, y.astype(float))[0, 1]), 4),
            "partial_corr_score_label_given_loglength": round(
                partial_corr(s, y.astype(float), loglen), 4),
            "partial_spearman_score_label_given_loglength": round(
                partial_spearman(s, y.astype(float), loglen), 4),
        }

    for n_strata in strata_counts:
        strata = quantile_strata(rows.prefix_thinking_tokens, n_strata)
        block: dict = {
            "n_strata": n_strata,
            "bounds_thinking_tokens": stratum_bounds(rows.prefix_thinking_tokens, strata),
            "readers": {},
        }
        for name in READER_ORDER:
            if name not in rows.scores:
                continue
            s = rows.scores[name]
            pooled, mean_auc, per = stratified_auc(y, s, strata)
            lo, hi = pooled_ci(y, s, strata, n_bootstrap, seed)
            block["readers"][name] = {
                "pooled_auc": round(pooled, 4),
                "pooled_auc_ci95": [round(lo, 4), round(hi, 4)],
                "mean_stratum_auc": round(mean_auc, 4),
                "per_stratum": per,
                "leak_attributable": round(
                    float(out["readers"][name]["auc_unstratified"] - pooled), 4),
            }
        if "probe_honest" in block["readers"] and "tfidf" in block["readers"]:
            d, lo, hi = pooled_delta_ci(
                y, rows.scores["probe_honest"], rows.scores["tfidf"], strata, n_bootstrap, seed
            )
            block["delta_probe_minus_tfidf"] = {
                "pooled": round(d, 4),
                "ci95": [round(lo, 4), round(hi, 4)],
                "excludes_zero": bool(lo > 0 or hi < 0),
                "unstratified": round(
                    auc(y, rows.scores["probe_honest"]) - auc(y, rows.scores["tfidf"]), 4),
            }
        out["strata"][str(n_strata)] = block
    return out


def run(
    ks: Sequence[int] = K_VALUES,
    block_dir: Path = BLOCK2_DIR,
    honest_path: Path | None = None,
    n_bootstrap: int = N_BOOTSTRAP,
) -> dict:
    t0 = time.time()
    block_dir = Path(block_dir)
    honest_path = Path(honest_path) if honest_path else block_dir / "honest_probe.json"
    honest = None
    if honest_path.exists():
        with open(honest_path) as f:
            honest = json.load(f)

    per_k: Dict[str, dict] = {}
    for k in ks:
        sel = None
        if honest:
            sel = honest["per_k"][str(k)]["honest"]
        rows = build_cut_rows(k, block_dir, sel)
        per_k[str(k)] = analyse_cut(rows, n_bootstrap=n_bootstrap)

    return {
        "schema_version": 1,
        "analysis": "length_control",
        "run": "010b",
        "metric": "roc_auc",
        "question": (
            "How much of each reader's AUC is the k%-truncation protocol's length leak, "
            "and once length is controlled does the probe still lose to tuned TF-IDF?"
        ),
        "protocol": {
            "split": "REUSED VERBATIM from each cut's results.json",
            "stratification": "equal-count rank terciles/quartiles of the TEST rows by prefix thinking tokens",
            "pooled_estimator": "pair-weighted (Mantel-Haenszel-style) within-stratum AUC",
            "bootstrap": "stratified resampling within strata, percentile CI",
            "n_bootstrap": n_bootstrap,
            "probe": "layer + C selected honestly in Run 010 (honest_probe.json), refit on train only",
            "tfidf": "stored test scores from baseline_text_classifier.json (no refit)",
            "crude_floor": "logistic on (prefix token count, prompt token count), train-fit",
            "length_only": "logistic on log1p(prefix thinking tokens), train-fit",
            "seed": SEED,
        },
        "per_k": per_k,
        "package_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "elapsed_s": round(time.time() - t0, 1),
    }


def _print_table(out: dict, n_strata: int = 3) -> None:
    print()
    print(f"LENGTH-CONTROLLED AUC (Run 010b) — {n_strata} equal-count length strata of the test rows")
    print(f"{'k':>4} {'reader':>13} {'raw AUC':>8} {'pooled':>8} {'pooled CI95':>18} "
          f"{'leak':>7} | {'per-stratum AUC (short->long)':>34} {'r(sc,lab|len)':>14}")
    for k in sorted(out["per_k"], key=int):
        r = out["per_k"][k]
        blk = r["strata"][str(n_strata)]
        for name in READER_ORDER:
            if name not in blk["readers"]:
                continue
            b = blk["readers"][name]
            per = " ".join(
                f"{(b['per_stratum'][str(g)]['auc'] if b['per_stratum'][str(g)]['auc'] is not None else float('nan')):.3f}"
                for g in range(n_strata)
            )
            pc = r["readers"][name]["partial_corr_score_label_given_loglength"]
            ci = f"[{b['pooled_auc_ci95'][0]:.3f}, {b['pooled_auc_ci95'][1]:.3f}]"
            print(f"{k:>4} {name:>13} {r['readers'][name]['auc_unstratified']:>8.4f} "
                  f"{b['pooled_auc']:>8.4f} {ci:>18} {b['leak_attributable']:>+7.4f} | "
                  f"{per:>34} {pc:>14.3f}")
        d = blk.get("delta_probe_minus_tfidf")
        if d:
            print(f"{'':>4} {'DELTA p-tf':>13} {d['unstratified']:>+8.4f} {d['pooled']:>+8.4f} "
                  f"[{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]"
                  f"   {'EXCLUDES 0' if d['excludes_zero'] else 'includes 0'}")
        print()


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--block-dir", default=str(BLOCK2_DIR))
    ap.add_argument("--honest", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--k", nargs="*", type=int, default=list(K_VALUES))
    ap.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    ap.add_argument("--strata", type=int, default=3)
    args = ap.parse_args(argv)

    block_dir = Path(args.block_dir)
    out_path = Path(args.out) if args.out else block_dir / "length_control.json"
    out = run(args.k, block_dir, args.honest, args.n_bootstrap)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    _print_table(out, args.strata)
    print(f"length_control: -> {out_path}  ({out['elapsed_s']}s)")


if __name__ == "__main__":
    main()
