"""Invariant tests for the three text baselines and the Δ analysis.

Hard constraints on this file (it runs as a pre-flight gate before any GPU is
rented, on a laptop, offline):
  * no torch, no vllm, no transformers, no model weights;
  * no network — every API path is exercised through an injected fake;
  * no writes outside pytest's tmp_path.

What is tested is what could silently corrupt the headline number:
the forced-answer prompt construction, the grading of forced answers, the
judge's probability parser and cache key, the text classifier's tuning
honesty, and the Δ / CI arithmetic including the missing-baseline paths.
"""

from __future__ import annotations

import inspect
import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from experiment import analysis, forced_answer, llm_judge, text_classifier
from experiment.config import THINK_END_ID, THINK_START_ID


# ===========================================================================
# 0. The GPU-free / network-free contract
# ===========================================================================

HEAVY = ("torch", "vllm", "transformers", "anthropic", "matplotlib")


def test_baseline_modules_do_not_pull_in_gpu_or_api_stacks():
    """Importing any baseline module must not drag in torch/vllm/transformers/
    matplotlib.

    If this ever fails, the pre-flight pytest gate can no longer run on a
    laptop before the GPU is rented, and stops being a gate at all. Checked in
    a FRESH interpreter so the result cannot depend on test ordering.
    """
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    code = (
        "import sys\n"
        "import experiment.analysis, experiment.forced_answer, "
        "experiment.llm_judge, experiment.text_classifier\n"
        f"print([m for m in {HEAVY!r} if m in sys.modules])\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=repo_root, capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "[]", (
        f"heavy modules imported at module scope: {proc.stdout.strip()} — keep "
        f"them inside the function that needs them"
    )


# ===========================================================================
# 1. forced_answer — prompt construction and think-tag closure
# ===========================================================================

def test_build_forced_prompt_closes_the_think_tag_then_forces():
    prefix = [10, 11, THINK_START_ID, 12, 13]
    suffix = [90, 91]
    out = forced_answer.build_forced_prompt_ids(prefix, suffix)
    assert out == prefix + [THINK_END_ID] + suffix
    # the closing tag sits exactly between the prefix and the forcing string
    assert out.index(THINK_END_ID) == len(prefix)
    assert out[len(prefix) + 1 :] == suffix


def test_build_forced_prompt_rejects_a_prefix_that_already_ended_thinking():
    """A `</think>` in the prefix means the model already finished reasoning —
    the forced answer would no longer be an interruption, and the baseline
    would silently become a much easier task."""
    with pytest.raises(RuntimeError, match="already present"):
        forced_answer.build_forced_prompt_ids([1, 2, THINK_END_ID, 3], [9])


def test_build_forced_prompt_rejects_an_empty_prefix():
    with pytest.raises(RuntimeError, match="empty prefix"):
        forced_answer.build_forced_prompt_ids([], [9])


def test_forcing_suffix_opens_a_box_and_adds_no_information():
    s = forced_answer.FORCED_ANSWER_SUFFIX
    assert s.rstrip().endswith("\\boxed{"), "the suffix must open the box it will be graded from"
    for leak in ("A)", "the correct answer is A", "hint"):
        assert leak.lower() not in s.lower()


@pytest.mark.parametrize(
    "text,expected",
    [
        ("\\boxed{C}", "\\boxed{C}"),                                  # balanced: untouched
        ("\\boxed{C", "\\boxed{C}"),                                   # truncated: closed
        ("\\boxed{\\frac{1}{2}}", "\\boxed{\\frac{1}{2}}"),
        ("\\boxed{\\frac{1}{2}", "\\boxed{\\frac{1}{2}}"),             # nested + truncated
        ("no box here", "no box here"),
    ],
)
def test_close_boxed(text, expected):
    assert forced_answer.close_boxed(text) == expected


def test_assemble_forced_response_includes_the_suffix():
    """The suffix carries the opening `\\boxed{`; grading the generation alone
    would find no box and mark every row ungradeable."""
    out = forced_answer.assemble_forced_response(forced_answer.FORCED_ANSWER_SUFFIX, "C}")
    assert out.startswith(forced_answer.FORCED_ANSWER_SUFFIX)
    assert "\\boxed{C}" in out


# ===========================================================================
# 2. forced_answer — grading
# ===========================================================================

SUF = forced_answer.FORCED_ANSWER_SUFFIX


@pytest.mark.parametrize(
    "generated,gold,expected",
    [
        ("C}", "C", True),
        ("C} because option C matches.", "C", True),
        ("D}", "C", False),
        ("c}", "C", True),                      # case must not count as wrong
        ("C", "C", True),                       # ran out of tokens: closed by close_boxed
        ("**C**}", "C", True),                  # decoration stripped
        ("nonsense}", "C", False),              # not a valid option => wrong, not ungradeable
        ("\\frac{1}{2}}", "\\frac{1}{2}", True),   # MATH-500 shape still works
        ("\\dfrac{1}{2}}", "\\frac{1}{2}", True),
        ("3}", "\\frac{1}{2}", False),
    ],
)
def test_grade_forced_answer(generated, gold, expected):
    assert forced_answer.grade_forced_answer(SUF, generated, gold) is expected


def test_forced_answer_text_reports_what_the_model_committed_to():
    assert forced_answer.forced_answer_text(SUF, "B} — final.") == "B"


# ===========================================================================
# 3. forced_answer — shared prefix I/O helpers
# ===========================================================================

def _write_prefixes(tmp_path, rows, k=50):
    p = tmp_path / "prefixes.jsonl"
    with open(p, "w") as f:
        f.write(json.dumps({"record_type": "meta", "stage": "truncate", "k_percent": k}) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return str(p)


def test_load_included_prefix_rows_filters_excluded_and_reads_k(tmp_path):
    path = _write_prefixes(
        tmp_path,
        [
            {"problem_id": "p1", "included": True, "label": True, "prefix_token_ids": [1, 2]},
            {"problem_id": "p2", "included": False, "exclusion_reason": "ungradeable"},
        ],
        k=25,
    )
    rows, k = forced_answer.load_included_prefix_rows(path)
    assert [r["problem_id"] for r in rows] == ["p1"]
    assert k == 25


def test_row_key_and_grouping_survive_a_k_grid(tmp_path):
    rows = [
        {"problem_id": "p1", "included": True, "k_percent": 10, "prefix_token_ids": [1]},
        {"problem_id": "p1", "included": True, "k_percent": 50, "prefix_token_ids": [1, 2]},
        {"problem_id": "p2", "included": True, "prefix_token_ids": [3]},  # inherits file k
    ]
    assert forced_answer.row_key(rows[0], 50) == "p1@k10"
    assert forced_answer.row_key(rows[2], 50) == "p2@k50"
    grouped = forced_answer.group_rows_by_k(rows, 50)
    assert sorted(grouped) == [10, 50]
    assert len(grouped[50]) == 2


def test_load_prefix_texts_prefers_the_cached_decode(tmp_path):
    """The forced-answer stage stores the decoded prefix, so the laptop-side
    baselines need no tokenizer AND cannot decode it differently."""
    cache = tmp_path / "forced_answer.jsonl"
    with open(cache, "w") as f:
        f.write(json.dumps({"record_type": "meta"}) + "\n")
        f.write(json.dumps({"row_key": "p1@k50", "prefix_text": "CACHED TEXT"}) + "\n")
    rows = [{"problem_id": "p1", "prefix_token_ids": [1, 2, 3]}]

    def boom(_ids):
        raise AssertionError("must not need a tokenizer when the cache covers every row")

    assert forced_answer.load_prefix_texts(rows, 50, str(cache), decode=boom) == ["CACHED TEXT"]


def test_load_prefix_texts_falls_back_to_the_tokenizer_for_uncached_rows(tmp_path):
    rows = [{"problem_id": "p9", "prefix_token_ids": [7, 8]}]
    got = forced_answer.load_prefix_texts(
        rows, 50, str(tmp_path / "absent.jsonl"), decode=lambda ids: f"<{len(ids)}>"
    )
    assert got == ["<2>"]


# ===========================================================================
# 4. llm_judge — probability parsing
# ===========================================================================

@pytest.mark.parametrize(
    "reply,expected",
    [
        ("Looks solid.\nPROBABILITY: 0.73", 0.73),
        ("PROBABILITY: 0.0", 0.0),
        ("PROBABILITY: 1", 1.0),
        ("PROBABILITY: 73%", 0.73),
        ("probability = .8", 0.8),                       # case + '=' + leading dot
        ("PROBABILITY: 0.2\nPROBABILITY: 0.9", 0.9),      # last one wins
        ("I'd say roughly 0.35", 0.35),                   # no label: last number in range
    ],
)
def test_parse_probability_valid(reply, expected):
    assert llm_judge.parse_probability(reply) == pytest.approx(expected)


@pytest.mark.parametrize(
    "reply",
    [
        None,
        "",
        "It is hard to say.",              # no number at all
        "PROBABILITY: high",               # non-numeric
        "PROBABILITY: 1.5",                # out of range above
        "PROBABILITY: -0.2",               # out of range below
        "PROBABILITY: 150%",               # out of range after percent conversion
        "I'd say 73",                      # bare integer, no % — refuse to guess
    ],
)
def test_parse_probability_rejects_malformed_and_out_of_range(reply):
    assert llm_judge.parse_probability(reply) is None


def test_unparseable_rows_are_dropped_not_imputed():
    """Documented contract: imputing 0.5 would drag the judge toward chance and
    inflate Δ. The parser must return None so the caller drops the row."""
    assert llm_judge.parse_probability("no idea") is None


# ===========================================================================
# 5. llm_judge — cache key and cache round-trip
# ===========================================================================

def test_cache_key_depends_on_prefix_prompt_and_model_only():
    base = llm_judge.cache_key("PREFIX", "model-a")
    assert base == llm_judge.cache_key("PREFIX", "model-a")           # deterministic
    assert base != llm_judge.cache_key("PREFIX ", "model-a")          # prefix matters
    assert base != llm_judge.cache_key("PREFIX", "model-b")           # model matters
    assert base != llm_judge.cache_key("PREFIX", "model-a", "other prompt")  # prompt matters
    assert len(base) == 64 and all(c in "0123456789abcdef" for c in base)


def test_cache_key_is_not_confusable_across_field_boundaries():
    """Fields are NUL-separated, so "ab"+"c" must not collide with "a"+"bc"."""
    assert llm_judge.cache_key("ab", "c") != llm_judge.cache_key("a", "bc")


def test_judge_one_hits_the_cache_on_the_second_call(tmp_path):
    calls = []

    def fake_call(user_message, provider, model):
        calls.append(user_message)
        return "PROBABILITY: 0.61"

    first = llm_judge.judge_one("PFX", "openrouter", "m", str(tmp_path), call=fake_call)
    second = llm_judge.judge_one("PFX", "openrouter", "m", str(tmp_path), call=fake_call)
    assert first["probability"] == pytest.approx(0.61) and first["cached"] is False
    assert second["probability"] == pytest.approx(0.61) and second["cached"] is True
    assert len(calls) == 1, "a cached prefix must not be re-sent to the API"
    assert "PFX" in calls[0]


def test_judge_batch_is_offline_with_an_injected_call(tmp_path):
    out = llm_judge.judge_batch(
        ["a", "b", "c"], "openrouter", "m", str(tmp_path),
        concurrency=1, call=lambda msg, p, m: "PROBABILITY: 0.5",
    )
    assert [o["probability"] for o in out] == [0.5, 0.5, 0.5]


def test_clip_prefix_only_fires_past_the_limit():
    assert llm_judge.clip_prefix("short", max_chars=100) == ("short", False)
    clipped, was = llm_judge.clip_prefix("x" * 500, max_chars=100)
    assert was is True and "elided" in clipped


# ===========================================================================
# 6. llm_judge — provider detection and retry/backoff (no network)
# ===========================================================================

def test_detect_provider_degrades_gracefully_with_no_key():
    assert llm_judge.detect_provider({}) == (None, None)


def test_detect_provider_prefers_openrouter_then_anthropic_and_honours_the_override():
    assert llm_judge.detect_provider({"OPENROUTER_API_KEY": "k"})[0] == "openrouter"
    assert llm_judge.detect_provider({"ANTHROPIC_API_KEY": "k"})[0] == "anthropic"
    both = {"OPENROUTER_API_KEY": "k", "ANTHROPIC_API_KEY": "k"}
    assert llm_judge.detect_provider(both)[0] == "openrouter"
    assert llm_judge.detect_provider({**both, "JUDGE_PROVIDER": "anthropic"}) == (
        "anthropic",
        llm_judge.ANTHROPIC_MODEL,
    )


def test_call_with_retries_backs_off_then_succeeds(monkeypatch):
    attempts = {"n": 0}
    slept = []

    def flaky(_msg, _model):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("429 rate limited")
        return "PROBABILITY: 0.42"

    monkeypatch.setattr(llm_judge, "_call_openrouter", flaky)
    out = llm_judge.call_with_retries("m", "openrouter", "model", sleep=slept.append)
    assert llm_judge.parse_probability(out) == pytest.approx(0.42)
    assert attempts["n"] == 3
    assert len(slept) == 2 and slept[1] > slept[0], "backoff must increase"


def test_call_with_retries_gives_up_and_returns_none(monkeypatch):
    monkeypatch.setattr(
        llm_judge, "_call_openrouter",
        lambda *_a: (_ for _ in ()).throw(RuntimeError("down")),
    )
    assert llm_judge.call_with_retries("m", "openrouter", "model", max_retries=2,
                                       sleep=lambda _s: None) is None


# ===========================================================================
# 7. text_classifier — tuning honesty
# ===========================================================================

POISON = "ZZZPOISONTOKENZZZ"


def _synthetic_corpus():
    """12 problems x 2 classes, wordy enough to clear min_df. Every TEST row is
    poisoned with a unique token: if the grid search ever touched a test row,
    that token would end up in the fitted vocabulary."""
    texts, labels, groups = [], [], []
    for i in range(12):
        good = i % 2 == 0
        body = (
            "the reasoning is clean and the answer is clearly option alpha "
            if good
            else "hmm wait i am confused let me recheck this again and again "
        )
        texts.append(body * 6)
        labels.append(good)
        groups.append(f"p{i}")
    train_idx = list(range(8))
    test_idx = list(range(8, 12))
    for i in test_idx:
        texts[i] = texts[i] + " " + " ".join([POISON] * 5)
    return texts, labels, groups, train_idx, test_idx


TINY_GRID = {
    "features__word__ngram_range": [(1, 1), (1, 2)],
    "clf__C": [0.1, 1.0],
}


def test_tune_has_no_parameter_through_which_test_data_could_arrive():
    """Structural guarantee: the honesty of the tuning is readable off the
    signature, not dependent on the caller behaving."""
    params = set(inspect.signature(text_classifier.tune).parameters)
    assert not any("test" in p for p in params), params
    assert {"texts_train", "y_train", "groups_train"} <= params


def test_grid_search_never_sees_a_test_row():
    texts, labels, groups, train_idx, test_idx = _synthetic_corpus()
    scores, report = text_classifier.fit_and_score(
        texts, labels, groups, train_idx, test_idx,
        param_grid=TINY_GRID, folds=2, n_jobs=1,
    )
    assert len(scores) == len(test_idx)
    assert report["n_train"] == len(train_idx)
    assert report["tuning_scope"] == "cross-validation within the training split only"

    # Empirical check: refit the winner exactly as fit_and_score did and inspect
    # its vocabulary. The poison token appears in 4 test documents (>= min_df),
    # so it WOULD be present had any test row reached the fit.
    search = text_classifier.tune(
        [texts[i] for i in train_idx],
        [labels[i] for i in train_idx],
        [groups[i] for i in train_idx],
        param_grid=TINY_GRID, folds=2, n_jobs=1,
    )
    features = dict(search.best_estimator_.named_steps["features"].transformer_list)
    vocab = features["word"].vocabulary_
    assert not any(POISON.lower() in term for term in vocab), (
        "a test-only token reached the tuned vectorizer's vocabulary — the grid "
        "search saw held-out data"
    )


def test_fit_and_score_refuses_overlapping_train_and_test_indices():
    texts, labels, groups, train_idx, test_idx = _synthetic_corpus()
    with pytest.raises(RuntimeError, match="overlap"):
        text_classifier.fit_and_score(
            texts, labels, groups, train_idx, train_idx[:2] + test_idx,
            param_grid=TINY_GRID, folds=2, n_jobs=1,
        )


def test_tune_halts_rather_than_fit_a_single_class_training_split():
    with pytest.raises(SystemExit, match="single class"):
        text_classifier.tune(["a b c"] * 4, [True] * 4, ["p1", "p2", "p3", "p4"],
                             param_grid=TINY_GRID, folds=2, n_jobs=1)


def test_n_cv_splits_is_bounded_by_the_rarer_class_and_the_group_count():
    y = [True] * 10 + [False] * 2
    groups = [f"p{i}" for i in range(12)]
    assert text_classifier.n_cv_splits(y, groups, folds=5) == 2      # only 2 negatives
    y2 = [True] * 6 + [False] * 6
    assert text_classifier.n_cv_splits(y2, groups, folds=5) == 5
    assert text_classifier.n_cv_splits(y2, ["p1", "p2"] * 6, folds=5) == 2  # only 2 problems


def test_param_grid_actually_varies_the_documented_knobs():
    assert set(text_classifier.PARAM_GRID) == {
        "features__word__ngram_range",
        "features__char__ngram_range",
        "clf__C",
        "clf__class_weight",
    }
    assert len(text_classifier.PARAM_GRID["clf__C"]) >= 4, "not a real C sweep"


# ===========================================================================
# 8. analysis — S_text is the max, and the contributors are recorded
# ===========================================================================

def _pt(auc, ci=None, keys=None, labels=None, scores=None):
    return {
        "auc": auc,
        "auc_ci95": ci or [auc - 0.1, auc + 0.1],
        "test_row_keys": keys,
        "test_labels": labels,
        "test_scores": scores,
    }


def test_s_text_takes_the_max_not_the_mean():
    pts = {"llm_judge": _pt(0.62), "text_classifier": _pt(0.71), "forced_answer": _pt(0.55)}
    s, winner, contributors = analysis.s_text_at_k(pts)
    assert s == pytest.approx(0.71)
    assert winner == "text_classifier"
    assert contributors == ["forced_answer", "llm_judge", "text_classifier"]


def test_s_text_is_none_when_no_reader_is_available():
    assert analysis.s_text_at_k({}) == (None, None, [])


# ===========================================================================
# 9. analysis — Δ and CI maths on synthetic arrays
# ===========================================================================

def test_paired_bootstrap_delta_is_exactly_zero_for_identical_scores():
    rng = np.random.default_rng(0)
    y = np.array([True, False] * 12)
    s = rng.random(24)
    d, lo, hi, frac = analysis.paired_delta_bootstrap(y, s, s, n_bootstrap=100, seed=1)
    assert d == pytest.approx(0.0, abs=1e-12)
    assert lo == pytest.approx(0.0, abs=1e-12) and hi == pytest.approx(0.0, abs=1e-12)
    assert frac == 0.0


def test_paired_bootstrap_finds_a_positive_delta_when_the_probe_is_perfect():
    y = np.array([True] * 12 + [False] * 12)
    probe = y.astype(float)                       # perfect ranking: AUC 1.0
    text = np.full(24, 0.5)                       # no information: AUC 0.5
    d, lo, hi, frac = analysis.paired_delta_bootstrap(y, probe, text, n_bootstrap=200, seed=0)
    assert d == pytest.approx(0.5)
    assert lo > 0.3 and hi <= 0.5 + 1e-9
    assert frac == pytest.approx(1.0)


def test_paired_bootstrap_finds_a_negative_delta_when_the_text_reader_wins():
    y = np.array([True] * 12 + [False] * 12)
    probe = np.full(24, 0.5)
    text = y.astype(float)
    d, lo, hi, _frac = analysis.paired_delta_bootstrap(y, probe, text, n_bootstrap=200, seed=0)
    assert d == pytest.approx(-0.5)
    assert hi < 0


def test_marginal_fallback_ci_is_centred_on_delta_and_wider_than_either_input():
    d, lo, hi = analysis.delta_ci_from_marginals(0.80, [0.70, 0.90], 0.60, [0.50, 0.70])
    assert d == pytest.approx(0.20)
    assert (lo + hi) / 2 == pytest.approx(0.20)
    # each input CI is 0.20 wide; combining two independent SEs gives sqrt(2)x
    assert (hi - lo) == pytest.approx(0.20 * np.sqrt(2), rel=1e-6)


def test_align_scores_matches_rows_by_key_not_by_position():
    probe = _pt(0.9, keys=["b@k50", "a@k50"], labels=[False, True], scores=[0.1, 0.9])
    text = _pt(0.6, keys=["a@k50", "b@k50"], labels=[True, False], scores=[0.7, 0.3])
    y, ps, ts = analysis.align_scores(probe, text)
    assert list(y) == [True, False]               # sorted by key: a, b
    assert list(ps) == [0.9, 0.1]
    assert list(ts) == [0.7, 0.3]


def test_align_scores_returns_none_on_disjoint_or_missing_rows():
    probe = _pt(0.9, keys=["a@k50"], labels=[True], scores=[0.9])
    assert analysis.align_scores(probe, _pt(0.6, keys=["z@k50"], labels=[True], scores=[0.5])) is None
    assert analysis.align_scores(probe, _pt(0.6)) is None
    assert analysis.align_scores(None, _pt(0.6)) is None


def test_align_scores_raises_when_the_two_readers_disagree_about_a_label():
    probe = _pt(0.9, keys=["a@k50"], labels=[True], scores=[0.9])
    text = _pt(0.6, keys=["a@k50"], labels=[False], scores=[0.5])
    with pytest.raises(RuntimeError, match="not looking at the same data"):
        analysis.align_scores(probe, text)


# ===========================================================================
# 10. analysis — end-to-end with baselines present, partially present, absent
# ===========================================================================

RESULTS_SINGLE_K = {
    "truncation_k_percent": 50,
    "n_test": 20,
    "best_layer": "layer_18",
    "per_layer": {"layer_18": {"auc": 0.80, "auc_ci95": [0.70, 0.90]}},
}


def test_analyze_with_no_text_baseline_refuses_to_report_a_delta():
    out = analysis.analyze(
        RESULTS_SINGLE_K,
        {"forced_answer": {}, "llm_judge": {}, "text_classifier": {}},
        n_bootstrap=50,
    )
    e = out["per_k"]["50"]
    assert e["delta"] is None and e["s_text"] is None
    assert e["delta_ci_method"] == "none"
    assert "unaudited" in e["warning"]
    assert out["baselines_never_available"] == ["forced_answer", "llm_judge", "text_classifier"]


def test_analyze_records_which_baselines_were_missing():
    out = analysis.analyze(
        RESULTS_SINGLE_K,
        {"forced_answer": {50: _pt(0.65)}, "llm_judge": {}, "text_classifier": {}},
        n_bootstrap=50,
    )
    e = out["per_k"]["50"]
    assert e["s_text"] == pytest.approx(0.65)
    assert e["s_text_source"] == "forced_answer"
    assert e["baselines_present"] == ["forced_answer"]
    assert e["baselines_missing"] == ["llm_judge", "text_classifier"]
    assert out["baselines_never_available"] == ["llm_judge", "text_classifier"]


def test_analyze_uses_the_conservative_fallback_when_scores_are_unavailable():
    out = analysis.analyze(
        RESULTS_SINGLE_K, {"text_classifier": {50: _pt(0.60, ci=[0.50, 0.70])}}, n_bootstrap=50
    )
    e = out["per_k"]["50"]
    assert e["delta"] == pytest.approx(0.20)
    assert e["delta_ci_method"] == "independent_normal_approx_conservative"
    assert "frac_bootstrap_delta_gt_0" not in e


def test_analyze_uses_the_paired_bootstrap_when_both_sides_have_row_scores():
    keys = [f"p{i}@k50" for i in range(24)]
    labels = [True] * 12 + [False] * 12
    probe_scores = [1.0] * 12 + [0.0] * 12
    text_scores = [0.5] * 24
    probe_by_k = {50: {"test_row_keys": keys, "test_labels": labels,
                       "test_scores": probe_scores}}
    baselines = {
        "text_classifier": {
            50: _pt(0.50, keys=keys, labels=labels, scores=text_scores)
        }
    }
    out = analysis.analyze(RESULTS_SINGLE_K, baselines, probe_by_k, n_bootstrap=100)
    e = out["per_k"]["50"]
    assert e["delta_ci_method"] == "paired_bootstrap"
    # Δ comes from the refit probe's own scores, not from results.json's rounded AUC
    assert e["delta"] == pytest.approx(0.5)
    assert e["frac_bootstrap_delta_gt_0"] == pytest.approx(1.0)


def test_analyze_handles_a_multi_k_results_file():
    results = {
        "per_k": {
            "10": {"n_test": 20, "best_layer": "layer_9",
                   "per_layer": {"layer_9": {"auc": 0.60, "auc_ci95": [0.50, 0.70]}}},
            "90": {"n_test": 20, "best_layer": "layer_27",
                   "per_layer": {"layer_27": {"auc": 0.85, "auc_ci95": [0.75, 0.95]}}},
        }
    }
    out = analysis.analyze(
        results, {"llm_judge": {10: _pt(0.55), 90: _pt(0.84)}}, n_bootstrap=50
    )
    assert sorted(out["per_k"], key=int) == ["10", "90"]
    assert out["per_k"]["10"]["delta"] == pytest.approx(0.05)
    assert out["per_k"]["90"]["delta"] == pytest.approx(0.01)


def test_crude_floor_from_results_competes_as_a_text_baseline():
    results = dict(RESULTS_SINGLE_K)
    results["text_floor"] = {"auc": 0.58, "auc_ci95": [0.48, 0.68], "n_test": 20,
                             "features": ["prefix_token_count", "prompt_token_count"]}
    pts = analysis.crude_floor_points(results)
    assert pts[50]["auc"] == pytest.approx(0.58)
    assert pts[50]["test_scores"] is None      # no per-row scores => fallback CI path


def test_crude_floor_absent_is_not_an_error():
    assert analysis.crude_floor_points(RESULTS_SINGLE_K) == {}


# ===========================================================================
# 10b. analysis — refitting the probe to recover per-row scores
# ===========================================================================

def _write_acts(path, pids, labels, layers=(9, 18, 27), dim=8, seed=0):
    rng = np.random.default_rng(seed)
    y = np.asarray(labels, dtype=bool)
    kw = {"problem_ids": np.array(pids), "labels": y,
          "config_hash": np.array("deadbeef")}
    for L in layers:
        # a weakly separable feature so LogisticRegression has something to fit
        X = rng.normal(size=(len(pids), dim))
        X[:, 0] += y.astype(float) * 3.0
        kw[f"acts_layer{L}"] = X.astype(np.float32)
    np.savez_compressed(path, **kw)


def test_probe_test_scores_reads_the_right_acts_key_and_the_recorded_split(tmp_path):
    """results.json names the layer `layer_18`; acts.npz stores `acts_layer18`.
    A mismatch here would silently disable the paired Δ bootstrap and downgrade
    every interval to the conservative fallback without anyone noticing."""
    pids = [f"p{i:02d}" for i in range(20)]
    labels = [i % 3 != 0 for i in range(20)]
    acts = tmp_path / "acts.npz"
    _write_acts(acts, pids, labels)

    test_pids = sorted(pids[::4])
    train_pids = sorted(set(pids) - set(test_pids))
    results = {
        "truncation_k_percent": 50,
        "best_layer": "layer_18",
        "per_layer": {"layer_18": {"auc": 0.9, "auc_ci95": [0.8, 1.0]}},
        "split": {"train_problem_ids": train_pids, "test_problem_ids": test_pids},
        "n_test": len(test_pids),
    }
    got = analysis.probe_test_scores(results, acts_path=str(acts))
    assert set(got) == {50}
    point = got[50]
    assert point["test_row_keys"] == [f"{p}@k50" for p in test_pids]
    assert len(point["test_scores"]) == len(test_pids)
    assert 0.0 <= min(point["test_scores"]) and max(point["test_scores"]) <= 1.0


def test_probe_test_scores_is_optional_when_acts_are_absent():
    assert analysis.probe_test_scores(RESULTS_SINGLE_K, acts_path="/no/such/acts.npz") == {}


def test_analyze_pairs_the_refit_probe_with_the_winning_baseline(tmp_path):
    """The full preferred path: refit -> align -> paired bootstrap."""
    pids = [f"p{i:02d}" for i in range(20)]
    labels = [i % 3 != 0 for i in range(20)]
    acts = tmp_path / "acts.npz"
    _write_acts(acts, pids, labels)
    test_pids = sorted(pids[::4])
    results = {
        "truncation_k_percent": 50,
        "best_layer": "layer_18",
        "per_layer": {"layer_18": {"auc": 0.9, "auc_ci95": [0.8, 1.0]}},
        "split": {"train_problem_ids": sorted(set(pids) - set(test_pids)),
                  "test_problem_ids": test_pids},
        "n_test": len(test_pids),
    }
    probe_by_k = analysis.probe_test_scores(results, acts_path=str(acts))
    assert probe_by_k, "refit should have succeeded"
    keys = probe_by_k[50]["test_row_keys"]
    ylab = probe_by_k[50]["test_labels"]
    baselines = {"llm_judge": {50: _pt(0.5, keys=keys, labels=ylab,
                                       scores=[0.5] * len(keys))}}
    out = analysis.analyze(results, baselines, probe_by_k, n_bootstrap=100)
    assert out["per_k"]["50"]["delta_ci_method"] == "paired_bootstrap"
    assert "frac_bootstrap_delta_gt_0" in out["per_k"]["50"]


# ===========================================================================
# 11. analysis — the baseline file contract
# ===========================================================================

def test_score_at_k_reports_auc_never_accuracy(monkeypatch, tmp_path):
    y = [True] * 8 + [False] * 8
    s = [0.9] * 8 + [0.1] * 8
    point = analysis.score_at_k(y, s, row_keys=[f"r{i}" for i in range(16)],
                                n_bootstrap=50, seed=0)
    assert point["auc"] == pytest.approx(1.0)
    assert point["n_pos"] == 8 and point["n_neg"] == 8
    assert len(point["test_scores"]) == 16
    assert "accuracy" not in point


def test_score_at_k_halts_on_a_single_class_test_split():
    with pytest.raises(SystemExit, match="single class"):
        analysis.score_at_k([True] * 6, [0.5] * 6, n_bootstrap=10)


def test_baseline_json_round_trip(monkeypatch, tmp_path):
    monkeypatch.setattr(analysis, "baseline_path", lambda name: str(tmp_path / f"b_{name}.json"))
    per_k = {"50": {"auc": 0.7, "auc_ci95": [0.6, 0.8], "n_test": 20}}
    analysis.write_baseline_json("llm_judge", per_k, notes={"model": "x"})
    got = analysis.read_baseline_json("llm_judge")
    assert got["baseline"] == "llm_judge"
    assert got["metric"] == "roc_auc"
    assert got["per_k"]["50"]["auc"] == pytest.approx(0.7)
    assert got["schema_version"] == analysis.BASELINE_SCHEMA_VERSION


def test_read_baseline_json_is_none_when_absent(monkeypatch, tmp_path):
    monkeypatch.setattr(analysis, "baseline_path", lambda name: str(tmp_path / "nope.json"))
    assert analysis.read_baseline_json("llm_judge") is None


def test_read_baseline_json_rejects_a_stale_schema(monkeypatch, tmp_path):
    p = tmp_path / "b.json"
    p.write_text(json.dumps({"schema_version": 0, "baseline": "x", "per_k": {}}))
    monkeypatch.setattr(analysis, "baseline_path", lambda name: str(p))
    with pytest.raises(RuntimeError, match="schema_version"):
        analysis.read_baseline_json("x")


# ===========================================================================
# 12. figures — must degrade gracefully rather than draw a meaningless chart
# ===========================================================================

def test_figure_one_is_skipped_when_no_k_has_a_delta():
    out = analysis.analyze(RESULTS_SINGLE_K, {"llm_judge": {}}, n_bootstrap=50)
    assert analysis.figure_delta_curve(out, path="/nonexistent/should-not-be-written.png") is None
