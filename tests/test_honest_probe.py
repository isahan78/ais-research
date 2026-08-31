"""Tests for honest_probe.py and length_control.py (Run 010).

The load-bearing invariant: model selection (layer AND C) must be blind to
the test split. Everything else is arithmetic.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from experiment import honest_probe as hp
from experiment import length_control as lc


class TestSelectionHonesty:
    def test_select_by_cv_signature_cannot_receive_test_rows(self):
        # The selection entry point takes training arrays only. If anyone adds
        # a test/eval parameter, this fails and the leak becomes visible.
        params = list(inspect.signature(hp.select_by_cv).parameters)
        for banned in ("X_test", "y_test", "test_rows", "eval_rows", "te"):
            assert banned not in params, f"selection can see {banned}"

    def test_cv_auc_source_never_references_test(self):
        src = inspect.getsource(hp.cv_auc_for) + inspect.getsource(hp.select_by_cv)
        # sklearn's CV folds are legitimately named "test" inside the training
        # split; what must NEVER appear is the experiment's held-out split.
        for banned in ("test_problem_ids", "X_test", "y_test", "split_indices_from_results"):
            assert banned not in src, f"selection touches {banned}"

    def test_selection_is_deterministic(self):
        rng = np.random.default_rng(0)
        X = {"layer_9": rng.normal(size=(80, 12)), "layer_18": rng.normal(size=(80, 12))}
        y = rng.random(80) > 0.6
        groups = np.array([f"p{i//2}" for i in range(80)])
        a = hp.select_by_cv(X, y, groups)
        b = hp.select_by_cv(X, y, groups)
        assert (a.name, a.C) == (b.name, b.C)


class TestLengthControl:
    def test_quantile_strata_are_balanced_and_ordered(self):
        vals = list(range(100))
        s = lc.quantile_strata(vals, 4)
        assert len(s) == 100 and set(s) == {0, 1, 2, 3}
        counts = np.bincount(s)
        assert counts.max() - counts.min() <= 1
        # ordering: larger values land in later strata
        assert s[0] == 0 and s[-1] == 3

    def test_partial_corr_removes_a_pure_confound(self):
        rng = np.random.default_rng(1)
        z = rng.normal(size=500)
        x = z + rng.normal(scale=0.05, size=500)      # x is basically z
        y = (z + rng.normal(scale=0.5, size=500)) > 0  # y driven by z
        raw = np.corrcoef(x, y.astype(float))[0, 1]
        part = lc.partial_corr(x, y.astype(float), z)
        assert abs(raw) > 0.4          # confounded correlation is large
        assert abs(part) < 0.15        # controlling for z kills it

    def test_stratified_auc_defeats_a_pure_length_reader(self):
        rng = np.random.default_rng(2)
        length = rng.uniform(1, 10, size=400)
        y = (length + rng.normal(scale=2.0, size=400)) > 5.5   # label ~ length
        s_len = length                                          # reader = length
        strata = lc.quantile_strata(length, 4)
        overall = lc.auc(np.asarray(y), np.asarray(s_len))
        within, _, _ = lc.stratified_auc(np.asarray(y), np.asarray(s_len), strata)
        assert overall > 0.68           # looks strong unstratified
        assert within < overall - 0.05  # advantage collapses within strata
