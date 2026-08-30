"""Post-hoc diagnostics on Block 2 outputs. CPU only, no new runs, no GPU.

Written 2026-08-30 during review of Run 005. Every number Tyler reported in
that review comes from this file — run it and check them.

    cd experiment/outputs/block2 && python ../../diagnostics.py

Three questions, in order of how much they change the write-up:

1. `decompose_forced_answer` — the forced-answer score is BINARY
   (`forced_correct`), so its ROC-AUC is just (TPR+TNR)/2. If the forced
   answer usually EQUALS the final answer, that AUC is measuring answer
   agreement, not prediction. Reported alongside the agreement rate and the
   balanced accuracy restricted to rows where forced != final — the subset
   where the baseline is doing genuine prediction rather than copying.

2. `gain_over_question_only` — raw AUC rewards whoever best reads question
   difficulty, which is available at k=1 before any reasoning exists. The
   quantity the experiment is actually about is the GAIN over that baseline.
   Paired bootstrap, because every k shares one test split.

3. `orthogonality` — Spearman between probe and text-classifier scores, and an
   untuned 50/50 ensemble. An indicator only: the Delta = max(readers) framing
   structurally cannot ask whether the probe adds signal ON TOP of the text.
   A real incremental test fits the combiner inside the training split.

4. `subject_confound` (added 2026-08-30, second review pass) — MMLU-Pro spans
   14 subjects with wildly different error rates (train: math 0%, law 38%,
   CS 43%). A one-hot SUBJECT predictor needs no text and no activations. If
   it matches the readers, "question difficulty" sharpens to "subject
   identity" and every raw AUC in the headline table is mostly a subject
   detector. Fit on train subjects only; paired-bootstrapped against each
   reader on the shared test split.

A correction recorded against the first review pass: the `decompose` balanced
accuracy on the disagreement subset is PARTLY ARITHMETIC, not empirics. For a
CORRECT trace (label=True), forced != final implies forced != gold implies
forced_correct=False — deterministically. So TPR=0 on that subset by
construction, and for the positive class forced_correct == (forced == final)
EXACTLY: the "predictor" is the copy detector by identity, not by tendency.
This makes the answer-copy kill stronger, but the below-chance framing in the
first pass overstated what was measured. Also note: forced-answer's score is
binary, so its "AUC" is one operating point (balanced accuracy) — a continuous
upgrade would score the forced answer's token logprob, which needs a rerun.
"""
from __future__ import annotations

import json
import re

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

KS = [1, 10, 25, 50, 75, 90]
LAYERS = [9, 18, 27]
N_BOOT = 5000


def _split():
    res = json.load(open("k1/results.json"))
    return ([str(p) for p in res["split"]["train_problem_ids"]],
            [str(p) for p in res["split"]["test_problem_ids"]])


def _rows(path):
    return [r for r in (json.loads(l) for l in open(path)) if r.get("record_type") != "meta"]


def probe_scores(k, train_ids, test_ids):
    """Re-fit the probe exactly as train_probe.py does, on the same split."""
    d = np.load(f"k{k}/acts.npz", allow_pickle=False)
    pid = np.array([str(p) for p in d["problem_ids"]])
    y = d["labels"].astype(int)
    tr, te = np.isin(pid, train_ids), np.isin(pid, test_ids)
    best, best_auc = None, -1.0
    for layer in LAYERS:
        X = d[f"acts_layer{layer}"]
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, C=1.0, random_state=0))
        clf.fit(X[tr], y[tr])
        s = clf.predict_proba(X[te])[:, 1]
        auc = roc_auc_score(y[te], s)
        if auc > best_auc:
            best, best_auc = s, auc
    return pid[te], y[te], best


def tfidf_scores(k):
    d = json.load(open(f"k{k}/baseline_text_classifier.json"))["per_k"][str(k)]
    pid = np.array([rk.split("@")[0] for rk in d["test_row_keys"]])
    return pid, np.array(d["test_labels"]).astype(int), np.array(d["test_scores"])


def final_answer(trace):
    m = re.findall(r"\\boxed\{([^}]*)\}", trace.get("trace_text", ""))
    return m[-1].strip() if m else None


def decompose_forced_answer(test_ids):
    traces = {r["problem_id"]: r for r in _rows("gen/traces.jsonl")}
    print("FORCED-ANSWER: IS IT PREDICTING, OR COPYING THE FINAL ANSWER?")
    print(f"{'k':>3} {'AUC':>6} {'TPR':>6} {'TNR':>6} {'agree':>6} | "
          f"{'n(disagree)':>11} {'bal.acc':>8}")
    for k in [k for k in KS if k != 1]:
        rows = [r for r in _rows(f"k{k}/forced_answer.jsonl") if str(r["problem_id"]) in test_ids]
        pos = [r for r in rows if r["label"]]
        neg = [r for r in rows if not r["label"]]
        tpr = sum(r["forced_correct"] for r in pos) / len(pos)
        tnr = sum(not r["forced_correct"] for r in neg) / len(neg)
        matched = [r["forced_answer"] == final_answer(traces[r["problem_id"]]) for r in rows]
        agree = sum(bool(m) for m in matched) / len(matched)
        sub = [r for r, m in zip(rows, matched) if m is False]
        sp = [r for r in sub if r["label"]]
        sn = [r for r in sub if not r["label"]]
        if sp and sn:
            bal = (sum(r["forced_correct"] for r in sp) / len(sp)
                   + sum(not r["forced_correct"] for r in sn) / len(sn)) / 2
            bal_s = f"{bal:>8.3f}"
        else:
            bal_s = f"{'n/a':>8}"
        print(f"{k:>3} {(tpr + tnr) / 2:>6.3f} {tpr:>6.3f} {tnr:>6.3f} {agree:>6.3f} | "
              f"{len(sub):>11} {bal_s}")
    print("  AUC tracking `agree` means the baseline is an answer-copy, not a predictor.\n")


def _paired_gain(y, boots, s_k, s_1):
    d = np.array([roc_auc_score(y[b], s_k[b]) - roc_auc_score(y[b], s_1[b]) for b in boots])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return d.mean(), lo, hi, float((d > 0).mean())


def gain_over_question_only(order, y, probe, tfidf):
    rng = np.random.default_rng(0)
    n = len(y)
    boots = [b for b in (rng.integers(0, n, n) for _ in range(N_BOOT)) if len(set(y[b])) > 1]
    print(f"GAIN OVER THE k=1 QUESTION-ONLY BASELINE (paired bootstrap, {len(boots)} resamples)")
    print(f"{'k':>3} | {'probe':>6} {'gain':>7} {'CI95':>17} {'P>0':>5} | "
          f"{'tfidf':>6} {'gain':>7} {'CI95':>17} {'P>0':>5}")
    for k in KS:
        pa, ta = roc_auc_score(y, probe[k]), roc_auc_score(y, tfidf[k])
        if k == 1:
            print(f"{k:>3} | {pa:>6.3f} {'—':>7} {'(reference)':>17} {'—':>5} | "
                  f"{ta:>6.3f} {'—':>7} {'(reference)':>17} {'—':>5}")
            continue
        g = _paired_gain(y, boots, probe[k], probe[1])
        h = _paired_gain(y, boots, tfidf[k], tfidf[1])
        print(f"{k:>3} | {pa:>6.3f} {g[0]:>+7.3f} [{g[1]:>+6.3f},{g[2]:>+6.3f}] {g[3]:>5.2f} | "
              f"{ta:>6.3f} {h[0]:>+7.3f} [{h[1]:>+6.3f},{h[2]:>+6.3f}] {h[3]:>5.2f}")
    print()


def orthogonality(y, probe, tfidf):
    print("DO THE PROBE AND THE TEXT CLASSIFIER READ THE SAME THING?")
    print(f"{'k':>3} {'spearman':>9} {'probe':>7} {'tfidf':>7} {'ens':>7}")
    for k in KS:
        z = lambda v: (v - v.mean()) / (v.std() + 1e-12)
        print(f"{k:>3} {spearmanr(probe[k], tfidf[k]).statistic:>9.3f} "
              f"{roc_auc_score(y, probe[k]):>7.3f} {roc_auc_score(y, tfidf[k]):>7.3f} "
              f"{roc_auc_score(y, z(probe[k]) + z(tfidf[k])):>7.3f}")
    print("  `ens` is an UNTUNED 50/50 of z-scored scores — an indicator, not a result.\n")


def subject_confound(order, y, probe, tfidf):
    rows = {str(r["problem_id"]): r for r in _rows("gen/traces.jsonl")}
    train_ids, _ = _split()
    subjects = sorted({r["subject"] for r in rows.values()})
    onehot = lambda p: [1.0 if rows[p]["subject"] == s else 0.0 for s in subjects]
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(np.array([onehot(p) for p in train_ids]),
            np.array([int(rows[p]["correct"]) for p in train_ids]))
    subj = clf.predict_proba(np.array([onehot(p) for p in order]))[:, 1]
    print(f"SUBJECT-ONLY PREDICTOR ({len(subjects)} one-hot subjects, no text, no activations)")
    print(f"  AUC = {roc_auc_score(y, subj):.3f}   (readers' range: probe .62-.76, tfidf .73-.78)")
    rng = np.random.default_rng(0)
    n = len(y)
    boots = [b for b in (rng.integers(0, n, n) for _ in range(N_BOOT)) if len(set(y[b])) > 1]
    def vs(scores, name):
        d = np.array([roc_auc_score(y[b], scores[b]) - roc_auc_score(y[b], subj[b]) for b in boots])
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"  {name:<22} AUC={roc_auc_score(y, scores):.3f}  vs subject: {d.mean():+.3f} "
              f"[{lo:+.3f},{hi:+.3f}] P(>0)={(d > 0).mean():.2f}")
    for k in KS:
        vs(probe[k], f"probe k={k}")
    vs(tfidf[90], "tfidf k=90")
    z = lambda v: (v - v.mean()) / (v.std() + 1e-12)
    vs(z(probe[10]) + z(subj), "probe k=10 + subject")
    print("  If nothing clears the subject flag, raw AUCs here are subject detection.\n")


def main():
    train_ids, test_ids = _split()
    order, y, probe, tfidf = None, None, {}, {}
    for k in KS:
        pid, yk, s = probe_scores(k, train_ids, test_ids)
        if order is None:
            order, y = pid, yk
        elif list(pid) != list(order):
            raise SystemExit(f"HALT: k={k} test order differs from k=1 — splits are not shared.")
        probe[k] = s
        tp, ty, ts = tfidf_scores(k)
        ix = {p: i for i, p in enumerate(tp)}
        tfidf[k] = np.array([ts[ix[p]] for p in order])
        if not (np.array([ty[ix[p]] for p in order]) == y).all():
            raise SystemExit(f"HALT: k={k} text-classifier labels disagree with the probe's.")

    print("Probe AUCs re-derived by a second route — compare against results.json:")
    for k in KS:
        print(f"  k={k:>2}: {roc_auc_score(y, probe[k]):.4f}")
    print()
    decompose_forced_answer(set(test_ids))
    gain_over_question_only(order, y, probe, tfidf)
    orthogonality(y, probe, tfidf)
    subject_confound(order, y, probe, tfidf)


if __name__ == "__main__":
    main()
