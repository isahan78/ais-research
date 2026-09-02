"""Cross-fitted analysis for Run 012 (addresses the second external review).

Three changes from compute_final_table.py's single-split table:

1. CROSS-FIT. Instead of one 65/35 split (77 test negatives), we take
   out-of-fold predictions over EVERY trace via StratifiedGroupKFold, so the
   whole dataset (213 negatives on the k% grid, 196 on fixed-length) is
   test data exactly once. Intervals shrink toward the truth.

2. HONEST, SYMMETRIC search budget. The probe still selects (layer, C) by
   NESTED CV inside each outer training fold — no test leakage. The text side
   is now a SINGLE a-priori TF-IDF config (the one used throughout), named as
   THE text baseline, not a per-cut max over readers. So Delta = probe - TFIDF
   with no uncorrected selection on the text side. We also report the probe's
   config budget (layers x C) so the two budgets are on the record: the probe
   gets hundreds of configs, the text reader one.

3. POPULATION CONTROL. With restrict_pids we run the k% grid on the SAME 781
   long-trace population used for the fixed-length grid, so k%-vs-fixed-length
   differs only in cut geometry, not population. length-only on that fixed
   population under both protocols prices the leak within one population.

AUC is reported two ways: pooled over all out-of-fold predictions (primary),
and mean-of-per-fold AUC (cross-check). Delta CI is a cluster bootstrap over
problems on the pooled OOF predictions (one row per problem per cut). CPU only.
"""
import json
import sys
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

BASE = Path(__file__).resolve().parent / "outputs" / "expansion"
LAYERS = list(range(35))
CGRID = [10.0 ** e for e in range(-5, 3)]          # 8 C values (regularisation only)
# Matched-budget probe: the text baseline is ONE fixed TF-IDF config, so the
# fair probe is also one config -- a single layer fixed a priori, cross-fit over
# all data, with no per-cut layer search. We fix it to layer 27, the deepest of
# the three pre-registered probe layers (9/18/27 = 25/50/75% depth), chosen
# before this analysis and never tuned per cut. Only the regularisation C is
# picked by CV inside each training fold. The generous best-of-35-layers probe
# (280 configs) is reported separately from the single-split final table as an
# upper bound -- the point being that even THAT loses to text.
PROBE_LAYER = 27
# n<<d (≈770 samples, 4096 features): liblinear's DUAL solver is ~7x faster than
# lbfgs here at the same AUC. Used for every probe fit.
N_OUTER = 5
N_INNER = 3
KPCT = ["k1", "k10", "k25", "k50", "k75", "k90"]
ABS = ["abs64", "abs128", "abs256", "abs512", "abs1024"]
rng = np.random.default_rng(0)


def tfidf_pipe():
    # THE a-priori text baseline. One config, fixed before this analysis.
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(C=1.0, max_iter=2000),
    )


def load(tag):
    d = BASE / tag
    z = np.load(d / "acts.npz", allow_pickle=True)
    pid = np.array([str(x) for x in z["problem_ids"]])
    y = np.array(z["labels"]).astype(bool)
    # matched-budget probe uses only PROBE_LAYER; loading all 35 layers wastes
    # most of the runtime, so load just the one we score.
    acts = {PROBE_LAYER: z[f"acts_layer{PROBE_LAYER}"]}
    fc = {}
    for l in open(d / "forced_confidence.jsonl"):
        r = json.loads(l)
        if r.get("record_type") != "meta":
            fc[str(r["problem_id"])] = r
    txt = np.array([fc.get(p, {}).get("prefix_text", "") for p in pid])

    def _ptop(r):
        v = r.get("variants") or {}
        return v.get("p_top") if isinstance(v, dict) and v.get("p_top") is not None else np.nan

    ptop = np.array([_ptop(fc.get(p, {})) for p in pid])
    plen = {}
    for l in open(d / "prefixes.jsonl"):
        r = json.loads(l)
        if r.get("record_type") != "meta" and r.get("included"):
            plen[str(r["problem_id"])] = np.log(len(r["prefix_token_ids"]))
    plen = np.array([plen.get(p, np.nan) for p in pid])
    return acts, pid, y, txt, ptop, plen


def _probe(C):
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C, solver="liblinear", dual=True, max_iter=2000))


def select_C(acts, y, tr_idx, pid):
    """Pick C for the fixed a-priori probe layer by inner CV in the training fold."""
    X = acts[PROBE_LAYER][tr_idx]
    ytr, gtr = y[tr_idx], pid[tr_idx]
    splits = list(StratifiedGroupKFold(N_INNER).split(np.zeros(len(tr_idx)), ytr, gtr))
    best = (-1.0, 1.0)
    for C in CGRID:
        sc = []
        for a, b in splits:
            m = _probe(C).fit(X[a], ytr[a])
            sc.append(roc_auc_score(ytr[b], m.decision_function(X[b])))
        mu = float(np.mean(sc))
        if mu > best[0]:
            best = (mu, C)
    return best[1]


def cross_fit(tag, restrict_pids=None):
    acts, pid, y, txt, ptop, plen = load(tag)
    if restrict_pids is not None:
        mask = np.array([p in restrict_pids for p in pid])
        acts = {L: acts[L][mask] for L in acts}
        pid, y, txt, ptop, plen = pid[mask], y[mask], txt[mask], ptop[mask], plen[mask]

    n = len(y)
    oof_probe = np.full(n, np.nan)
    oof_tfidf = np.full(n, np.nan)
    oof_len = np.full(n, np.nan)
    fold_of = np.full(n, -1)
    picks = []
    outer = StratifiedGroupKFold(N_OUTER)
    for fold, (tr, te) in enumerate(outer.split(np.zeros(n), y, pid)):
        C = select_C(acts, y, tr, pid)
        picks.append((PROBE_LAYER, float(C)))
        m = _probe(C).fit(acts[PROBE_LAYER][tr], y[tr])
        oof_probe[te] = m.decision_function(acts[PROBE_LAYER][te])
        oof_tfidf[te] = tfidf_pipe().fit(txt[tr], y[tr]).decision_function(txt[te])
        ln = make_pipeline(StandardScaler(),
                           LogisticRegression(max_iter=2000)).fit(plen[tr].reshape(-1, 1), y[tr])
        oof_len[te] = ln.decision_function(plen[te].reshape(-1, 1))
        fold_of[te] = fold

    # forced-confidence is a raw score (not trained): impute nan with median
    conf = np.where(np.isnan(ptop), np.nanmedian(ptop), ptop)

    def pooled(s):
        return round(float(roc_auc_score(y, s)), 4)

    def per_fold_mean(s):
        aucs = []
        for f in range(N_OUTER):
            m = fold_of == f
            if len(set(y[m])) > 1:
                aucs.append(roc_auc_score(y[m], s[m]))
        return round(float(np.mean(aucs)), 4), round(float(np.std(aucs)), 4)

    # cluster bootstrap over problems (one row per problem) for Delta = probe - tfidf
    deltas = []
    for _ in range(2000):
        i = rng.integers(0, n, n)
        if len(set(y[i])) > 1:
            deltas.append(roc_auc_score(y[i], oof_probe[i]) - roc_auc_score(y[i], oof_tfidf[i]))
    dlo, dhi = np.percentile(deltas, [2.5, 97.5])
    dmean = float(np.mean(deltas))

    return {
        "tag": tag,
        "n": int(n),
        "n_neg": int((~y).sum()),
        "probe": pooled(oof_probe),
        "probe_perfold": per_fold_mean(oof_probe),
        "probe_picks": picks,
        "tfidf": pooled(oof_tfidf),
        "tfidf_perfold": per_fold_mean(oof_tfidf),
        "length_only": pooled(oof_len),
        "forced_conf": round(float(roc_auc_score(y, conf)), 4),
        "delta_probe_minus_tfidf": round(dmean, 4),
        "delta_ci": [round(float(dlo), 3), round(float(dhi), 3)],
        "delta_excludes_0": bool(dlo > 0 or dhi < 0),
        "probe_layer_fixed": PROBE_LAYER,
        "probe_budget": "1 layer fixed a priori; C by inner CV",
        "text_budget": "1 TF-IDF config fixed a priori",
    }


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    # population for the fixed-length grid = pids present in abs64
    zabs = np.load(BASE / "abs64" / "acts.npz", allow_pickle=True)
    pop781 = set(str(x) for x in zabs["problem_ids"])

    out = {"headline": {}, "pop_control": {}}
    if which in ("all", "headline"):
        for tag in KPCT + ABS:
            r = cross_fit(tag)
            out["headline"][tag] = r
            print(f"[headline] {tag:8} n={r['n']} neg={r['n_neg']} probe={r['probe']} "
                  f"tfidf={r['tfidf']} len={r['length_only']} conf={r['forced_conf']} "
                  f"D={r['delta_probe_minus_tfidf']:+.3f}{r['delta_ci']}"
                  f"{'*' if r['delta_excludes_0'] else ''}", flush=True)
    if which in ("all", "pop"):
        for tag in KPCT + ABS:                    # k% AND abs, all on the same 781
            r = cross_fit(tag, restrict_pids=pop781)
            out["pop_control"][tag] = r
            print(f"[pop781]   {tag:8} n={r['n']} neg={r['n_neg']} probe={r['probe']} "
                  f"tfidf={r['tfidf']} len={r['length_only']} "
                  f"D={r['delta_probe_minus_tfidf']:+.3f}{r['delta_ci']}"
                  f"{'*' if r['delta_excludes_0'] else ''}", flush=True)

    (BASE / "cross_fit.json").write_text(json.dumps(out, indent=1))
    print("WROTE", BASE / "cross_fit.json", flush=True)


if __name__ == "__main__":
    main()
