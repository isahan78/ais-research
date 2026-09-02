"""Guards for the expanded-dataset final table (experiment/outputs/expansion).

Two invariants that MUST NOT silently break:

  1. n_test-is-ROWS: every cut's reported n_test equals the number of held-out
     problem rows (one row per problem_id) recorded in that cut's
     results.json split — NOT a token count. A bug that split the wrong array
     once produced n_test=166672 (tokens); this pins that it can never pass.

  2. SHARED-SPLIT: within a cut, every reader (probe activations, the TF-IDF /
     length prefixes, and the forced-confidence records) resolves to the exact
     SAME set of held-out problem_ids under the recorded split — so the four
     AUCs and the paired Δ are computed on identical test rows.

The tests read only artifacts already on disk and the project's own
`split_indices_from_results`. If the expansion artifacts are absent the tests
skip rather than fail, so the suite stays green on a fresh checkout.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from experiment.forced_answer import read_jsonl
from experiment.text_floor import split_indices_from_results

EXPERIMENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(EXPERIMENT_DIR, "outputs", "expansion")
FINAL_TABLE = os.path.join(BASE, "final_table.json")

K_CUTS = ["k1", "k10", "k25", "k50", "k75", "k90"]
ABS_CUTS = ["abs64", "abs128", "abs256", "abs512", "abs1024"]
ALL_CUTS = K_CUTS + ABS_CUTS


def _has_artifacts(cut: str) -> bool:
    d = os.path.join(BASE, cut)
    return all(
        os.path.exists(os.path.join(d, f))
        for f in ("results.json", "acts.npz", "prefixes.jsonl", "forced_confidence.jsonl")
    )


def _load_final_table():
    if not os.path.exists(FINAL_TABLE):
        pytest.skip("final_table.json not present — run the expansion driver first")
    with open(FINAL_TABLE) as f:
        return json.load(f)


def _recorded_split(cut: str):
    with open(os.path.join(BASE, cut, "results.json")) as f:
        results = json.load(f)
    return results


def _test_pid_set_via(pids, results) -> set:
    """The held-out problem_id set a reader with these row pids resolves to."""
    _train_idx, test_idx = split_indices_from_results(list(pids), results)
    return {str(pids[i]) for i in test_idx}


@pytest.mark.parametrize("cut", ALL_CUTS)
def test_n_test_is_rows_not_tokens(cut):
    if not _has_artifacts(cut):
        pytest.skip(f"{cut} artifacts absent")
    table = _load_final_table()
    if cut not in table:
        pytest.skip(f"{cut} not in final_table.json")

    results = _recorded_split(cut)
    train_pids = set(results["split"]["train_problem_ids"])
    test_pids = set(results["split"]["test_problem_ids"])

    # partition sanity
    assert not (train_pids & test_pids), f"{cut}: train/test problem_id overlap"

    n_test = table[cut]["n_test"]

    # (a) reported n_test == number of recorded held-out problems
    assert n_test == len(test_pids), (
        f"{cut}: n_test={n_test} != {len(test_pids)} recorded test problem_ids"
    )

    # (b) one row per problem in the activations, and that many are held out
    d = np.load(os.path.join(BASE, cut, "acts.npz"), allow_pickle=False)
    act_pids = np.array([str(p) for p in d["problem_ids"]])
    assert len(act_pids) == len(set(act_pids.tolist())), f"{cut}: >1 row per problem_id"
    n_held_rows = int(np.isin(act_pids, list(test_pids)).sum())
    assert n_test == n_held_rows, (
        f"{cut}: n_test={n_test} != {n_held_rows} held-out ROWS in acts.npz"
    )

    # (c) it is a ROW count, not a token count: rows <= population << tokens
    assert n_test <= len(act_pids), f"{cut}: n_test exceeds population (token-count bug)"
    band = (250, 360) if cut in K_CUTS else (230, 300)
    assert band[0] <= n_test <= band[1], (
        f"{cut}: n_test={n_test} outside the expected held-rows band {band}"
    )


@pytest.mark.parametrize("cut", ALL_CUTS)
def test_shared_split_identical_test_rows(cut):
    if not _has_artifacts(cut):
        pytest.skip(f"{cut} artifacts absent")
    results = _recorded_split(cut)
    recorded_test = set(results["split"]["test_problem_ids"])

    # probe side: activations
    d = np.load(os.path.join(BASE, cut, "acts.npz"), allow_pickle=False)
    probe_pids = [str(p) for p in d["problem_ids"]]

    # tfidf / length side: included prefix rows
    meta, rows = read_jsonl(os.path.join(BASE, cut, "prefixes.jsonl"))
    prefix_pids = [r["problem_id"] for r in rows if r.get("included")]

    # forced-confidence side: its records
    _mfc, frows = read_jsonl(os.path.join(BASE, cut, "forced_confidence.jsonl"))
    fc_pids = [r["problem_id"] for r in frows]

    probe_test = _test_pid_set_via(probe_pids, results)
    prefix_test = _test_pid_set_via(prefix_pids, results)
    fc_test = _test_pid_set_via(fc_pids, results)

    assert probe_test == recorded_test, f"{cut}: probe test rows != recorded split"
    assert prefix_test == recorded_test, f"{cut}: prefix (tfidf/length) test rows != recorded split"
    assert fc_test == recorded_test, f"{cut}: forced-confidence test rows != recorded split"

    # and, explicitly, all readers agree with one another
    assert probe_test == prefix_test == fc_test, (
        f"{cut}: readers disagree on the held-out test rows"
    )
