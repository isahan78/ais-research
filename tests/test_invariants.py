"""Invariant tests for the Gate 1 smoke pipeline.

Runs on any machine — no GPU, no model weights, no network. Covers the two
invariants whose silent failure would fabricate a result (split leakage,
truncation landing outside the thinking span) and every row of the spec's
I/O & edge-case matrix that has CPU-testable logic.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiment import harvest_activations
from experiment import truncate as truncate_module
from experiment.config import CONFIG, THINK_END_ID, THINK_START_ID, lineage
from experiment.generate_traces import (
    check_incomplete_fraction,
    check_ungradeable_fraction,
    parse_level,
)
from experiment.grading import extract_boxed, grade, normalize_answer
from experiment.harvest_activations import (
    check_activation_vector,
    oom_guard,
    residual_index,
)
from experiment.text_floor import build_feature_rows, split_indices_from_results
from experiment.train_probe import (
    bootstrap_ci,
    check_min_included,
    fit_and_auc,
    group_split,
    shuffled_label_floor_max,
    verdict_for,
)
from experiment.truncate import build_prefix

PROMPT = [1000, 1001, 1002, 1003]


def make_trace(n_think: int, with_start: bool = True, with_end: bool = True,
               n_answer: int = 5):
    """Synthetic completion token ids: [<think>] thinking... [</think>] answer..."""
    ids = []
    if with_start:
        ids.append(THINK_START_ID)
    ids += [2000 + i for i in range(n_think)]
    if with_end:
        ids.append(THINK_END_ID)
    ids += [3000 + i for i in range(n_answer)]
    return ids


# --------------------------------------------------------------------------
# Invariant: truncation lands strictly inside the thinking span
# --------------------------------------------------------------------------

class TestTruncation:
    def test_happy_path_k50(self):
        trace = make_trace(n_think=10)
        r = build_prefix("p1", PROMPT, trace, label=True, k_percent=50)
        assert r.included and r.exclusion_reason is None
        assert r.n_thinking_tokens == 10 and r.n_kept_thinking_tokens == 5
        # prefix = prompt + <think> + first 5 thinking tokens
        assert r.prefix_token_ids == PROMPT + [THINK_START_ID] + [2000 + i for i in range(5)]
        assert r.prompt_token_ids == PROMPT

    @pytest.mark.parametrize("n_think", range(4, 40))
    @pytest.mark.parametrize("k", [1, 50, 99])
    def test_truncation_always_inside_thinking_span(self, n_think, k):
        trace = make_trace(n_think=n_think)
        r = build_prefix("p", PROMPT, trace, label=False, k_percent=k)
        assert r.included
        # never empty, never the full span: the prefix must end MID-thinking
        assert 1 <= r.n_kept_thinking_tokens <= n_think - 1
        # </think> must never appear anywhere in the prefix
        assert THINK_END_ID not in r.prefix_token_ids
        # prefix ends on a thinking token, and <think> is open before it
        tail = r.prefix_token_ids[len(PROMPT):]
        assert tail[0] == THINK_START_ID
        assert tail[-1] == 2000 + r.n_kept_thinking_tokens - 1

    def test_think_start_rendered_in_prompt(self):
        # Template variant: <think> lives in the prompt; completion starts
        # directly with thinking tokens.
        trace = make_trace(n_think=8, with_start=False)
        prompt = PROMPT + [THINK_START_ID]
        r = build_prefix("p", prompt, trace, label=True)
        assert r.included
        assert r.prefix_token_ids == prompt + [2000 + i for i in range(4)]
        assert THINK_END_ID not in r.prefix_token_ids

    def test_prefix_starts_with_prompt(self):
        trace = make_trace(n_think=12)
        r = build_prefix("p", PROMPT, trace, label=True)
        assert r.prefix_token_ids[: len(PROMPT)] == PROMPT


# --------------------------------------------------------------------------
# I/O matrix: malformed traces are excluded, never crash
# --------------------------------------------------------------------------

class TestExclusions:
    def test_no_think_end_token(self):
        # Generation hit max_tokens mid-thinking (I/O matrix row 2)
        trace = make_trace(n_think=50, with_end=False, n_answer=0)
        r = build_prefix("p", PROMPT, trace, label=True)
        assert not r.included
        assert r.exclusion_reason == "truncated_incomplete"
        assert r.prefix_token_ids is None

    @pytest.mark.parametrize("n_think", [0, 1, 2, 3])
    def test_thinking_too_short(self, n_think):
        # Degenerate trace (I/O matrix row 3); threshold is 4
        trace = make_trace(n_think=n_think)
        r = build_prefix("p", PROMPT, trace, label=True)
        assert not r.included
        assert r.exclusion_reason == "thinking_too_short"

    def test_ungradeable_label_none(self):
        # Parser could not extract an answer (I/O matrix row 4)
        trace = make_trace(n_think=20)
        r = build_prefix("p", PROMPT, trace, label=None)
        assert not r.included
        assert r.exclusion_reason == "ungradeable"

    def test_halt_when_too_many_ungradeable(self):
        with pytest.raises(SystemExit, match="HALT"):
            check_ungradeable_fraction(n_ungradeable=7, n_gradeable_pool=20)  # 35% > 30%

    def test_no_halt_at_or_below_threshold(self):
        check_ungradeable_fraction(n_ungradeable=6, n_gradeable_pool=20)  # 30%, not >30%
        check_ungradeable_fraction(n_ungradeable=0, n_gradeable_pool=0)   # all incomplete

    def test_halt_when_too_many_incomplete(self):
        # I/O matrix row 6: >1/3 hitting the token cap is label-correlated
        # survivor bias; the halt must name the remedy (raise max_new_tokens).
        with pytest.raises(SystemExit, match="HALT.*max_new_tokens"):
            check_incomplete_fraction(n_incomplete=7, n_total=20)  # 35% > 34%

    def test_no_incomplete_halt_at_or_below_threshold(self):
        check_incomplete_fraction(n_incomplete=6, n_total=20)  # 30% <= 34%
        check_incomplete_fraction(n_incomplete=0, n_total=0)

    def test_min_included_gate_halts_with_reason_counts(self):
        # I/O matrix row 7: fewer than min rows -> HALT with per-reason counts.
        counts = {"truncated_incomplete": 5, "ungradeable": 4}
        with pytest.raises(SystemExit, match="HALT.*truncated_incomplete"):
            check_min_included(n_rows=11, min_rows=12, exclusion_counts=counts)

    def test_min_included_gate_passes_at_threshold(self):
        check_min_included(n_rows=12, min_rows=12, exclusion_counts={})

    @pytest.mark.parametrize("raw,expected", [
        (4, 4), ("Level 5", 5), ("5", 5), ("level 3", 3), ("unknown", None),
    ])
    def test_parse_level(self, raw, expected):
        assert parse_level(raw) == expected


# --------------------------------------------------------------------------
# Invariant: no problem_id appears in both splits
# --------------------------------------------------------------------------

class TestGroupSplit:
    def test_split_is_disjoint_by_problem_id(self):
        rng = np.random.default_rng(0)
        # 30 problems x 3 rows each (simulates multiple truncation points per
        # problem — the exact leakage scenario the invariant prevents)
        problem_ids = [f"prob_{i}" for i in range(30) for _ in range(3)]
        labels = np.array([bool(i % 2) for i in range(30) for _ in range(3)])
        order = rng.permutation(len(problem_ids))
        problem_ids = [problem_ids[i] for i in order]
        labels = labels[order]

        train_idx, test_idx = group_split(problem_ids, labels, 0.3, seed=0, max_retries=50)
        train_probs = {problem_ids[i] for i in train_idx}
        test_probs = {problem_ids[i] for i in test_idx}
        assert train_probs & test_probs == set()
        assert len(train_idx) + len(test_idx) == len(problem_ids)
        # rows of the same problem all landed on one side
        for p in train_probs | test_probs:
            rows = {i for i in range(len(problem_ids)) if problem_ids[i] == p}
            assert rows <= set(train_idx.tolist()) or rows <= set(test_idx.tolist())

    def test_both_classes_on_both_sides(self):
        problem_ids = [f"p{i}" for i in range(20)]
        labels = np.array([i < 7 for i in range(20)])
        train_idx, test_idx = group_split(problem_ids, labels, 0.35, seed=0, max_retries=50)
        assert len(set(labels[train_idx].tolist())) == 2
        assert len(set(labels[test_idx].tolist())) == 2

    def test_halts_on_single_class_data(self):
        problem_ids = [f"p{i}" for i in range(20)]
        labels = np.zeros(20, dtype=bool)  # model got everything wrong
        with pytest.raises(SystemExit, match="HALT"):
            group_split(problem_ids, labels, 0.35, seed=0, max_retries=10)


# --------------------------------------------------------------------------
# Grading (I/O matrix row 4 mechanics)
# --------------------------------------------------------------------------

class TestGrading:
    def test_extract_boxed_nested(self):
        assert extract_boxed(r"so \boxed{\frac{1}{2}} done") == r"\frac{1}{2}"

    def test_extract_boxed_takes_last(self):
        assert extract_boxed(r"\boxed{3} wait no \boxed{7}") == "7"

    def test_extract_boxed_missing(self):
        assert extract_boxed("the answer is 42, no box") is None

    def test_grade_ungradeable_is_none(self):
        assert grade("no boxed answer here", "42") is None

    def test_grade_correct_and_incorrect(self):
        assert grade(r"answer: \boxed{42}", "42") is True
        assert grade(r"answer: \boxed{41}", "42") is False

    @pytest.mark.parametrize("a,b", [
        (r"\dfrac{1}{2}", r"\frac{1}{2}"),
        ("2.0", "2"),
        (r"\left(3,\ 4\right)", "(3,4)"),
        ("{16}", "16"),
        ("45^\\circ", "45"),
    ])
    def test_normalize_equivalences(self, a, b):
        assert normalize_answer(a) == normalize_answer(b)


# --------------------------------------------------------------------------
# Lineage: every output can be traced to config hash + input file
# --------------------------------------------------------------------------

class TestLineage:
    def test_config_hash_stable_and_short(self):
        h1, h2 = CONFIG.config_hash(), CONFIG.config_hash()
        assert h1 == h2
        assert len(h1) == 12
        int(h1, 16)  # valid hex

    def test_config_hash_path_independent(self):
        from experiment.config import Config
        import dataclasses
        moved = dataclasses.replace(Config(), output_dir="/somewhere/else")
        assert moved.config_hash() == CONFIG.config_hash()

    def test_config_hash_changes_with_settings(self):
        from experiment.config import Config
        import dataclasses
        changed = dataclasses.replace(Config(), truncation_k_percent=25)
        assert changed.config_hash() != CONFIG.config_hash()

    def test_lineage_stamp_fields(self):
        stamp = lineage("traces.jsonl")
        assert stamp["config_hash"] == CONFIG.config_hash()
        assert stamp["input_file"] == "traces.jsonl"


# --------------------------------------------------------------------------
# Config sanity (Code Map facts frozen into the config)
# --------------------------------------------------------------------------

class TestConfig:
    def test_layers_below_post_norm_trap(self):
        # hidden_states index L+1 must never be the last entry (37 entries total)
        for L in CONFIG.layers:
            assert L + 1 < CONFIG.num_decoder_layers  # strictly below index 36 == [-1]

    def test_single_truncation_point_only(self):
        # The full grid is deferred work; Gate 1 is k=50 only.
        assert CONFIG.truncation_k_percent == 50

    def test_think_token_ids(self):
        assert THINK_START_ID == 151667
        assert THINK_END_ID == 151668


# --------------------------------------------------------------------------
# I/O matrix row 5: GPU OOM during harvest -> crash with actionable message
# --------------------------------------------------------------------------

class TestOomHandling:
    def test_oom_message_is_actionable(self):
        msg = harvest_activations.oom_halt_message(RuntimeError("CUDA out of memory"))
        assert msg.startswith("HALT:")
        # Must name the remedies the spec's matrix requires.
        assert "reduce harvest batch size" in msg
        assert "logits_to_keep=1" in msg
        # Must preserve the original error for debugging.
        assert "CUDA out of memory" in msg

    def test_oom_guard_converts_torch_style_oom(self):
        # The REAL guard used in main(): torch.cuda.OutOfMemoryError is
        # class-named `OutOfMemoryError`; the guard recognizes it by name
        # (no torch needed) and converts it to SystemExit + remedy text.
        class OutOfMemoryError(Exception):  # same class name as torch's
            pass

        with pytest.raises(SystemExit) as exc:
            with oom_guard():
                raise OutOfMemoryError("Tried to allocate 2.00 GiB")

        assert "HALT: CUDA OOM during harvest" in str(exc.value)
        assert "no co-residency" in str(exc.value)

    def test_oom_guard_converts_message_style_oom(self):
        # Some paths surface OOM as a RuntimeError with the canonical message.
        with pytest.raises(SystemExit, match="HALT: CUDA OOM"):
            with oom_guard():
                raise RuntimeError("CUDA out of memory. Tried to allocate 512 MiB")

    def test_oom_guard_passes_other_errors_through(self):
        # Non-OOM failures must NOT be relabeled as OOM.
        with pytest.raises(ValueError, match="unrelated"):
            with oom_guard():
                raise ValueError("unrelated bug")

    def test_oom_guard_is_transparent_on_success(self):
        with oom_guard():
            x = 1 + 1
        assert x == 2

    def test_oom_guard_wraps_model_load(self):
        # from_pretrained is the first place a lingering vLLM co-residency
        # OOMs; the guard must cover it, not just the forward loop.
        import inspect

        src = inspect.getsource(harvest_activations.main)
        assert "with oom_guard():\n        model = AutoModel.from_pretrained(" in src

    def test_harvest_module_imports_without_torch(self):
        # Row 5 is only reachable if the module itself loads on a CPU box;
        # torch must stay lazily imported inside main().
        assert hasattr(harvest_activations, "oom_halt_message")
        assert "import torch" not in _module_toplevel_imports()


def _module_toplevel_imports() -> str:
    import inspect

    src = inspect.getsource(harvest_activations)
    head = src.split("def ", 1)[0]
    return head


# --------------------------------------------------------------------------
# residual_index: the [L+1]-never-[-1] rule as an executable table
# --------------------------------------------------------------------------

class TestResidualIndex:
    @pytest.mark.parametrize("layer,expected", [
        (0, 1),    # residual after layer 0 is hidden_states[1] ([0] = embeddings)
        (9, 10),
        (18, 19),
        (27, 28),
        (34, 35),  # deepest legal layer for 37 entries: 35 is not the last index
    ])
    def test_residual_index_table(self, layer, expected):
        assert residual_index(layer, n_hidden_states=37) == expected

    def test_configured_layers_map_off_by_one_above(self):
        # The exact indices main() will use for the configured layers.
        n = CONFIG.num_decoder_layers + 1
        for L in CONFIG.layers:
            assert residual_index(L, n) == L + 1

    @pytest.mark.parametrize("layer", [35, 36, 40])
    def test_residual_index_refuses_post_norm_trap(self, layer):
        # hidden_states[-1] is post-final-RMSNorm; mapping onto it must raise.
        with pytest.raises(ValueError, match="post-final-RMSNorm"):
            residual_index(layer, n_hidden_states=37)

    def test_residual_index_refuses_negative_layer(self):
        with pytest.raises(ValueError):
            residual_index(-1, n_hidden_states=37)


# --------------------------------------------------------------------------
# Activation sanity: finiteness / nonzero / shape are raises, not asserts
# --------------------------------------------------------------------------

class TestActivationChecks:
    def test_valid_vector_passes(self):
        check_activation_vector(np.ones(8, dtype=np.float32), "p", 9, hidden_size=8)

    def test_nan_raises(self):
        vec = np.ones(8, dtype=np.float32)
        vec[3] = np.nan
        with pytest.raises(RuntimeError, match="non-finite"):
            check_activation_vector(vec, "p", 9, hidden_size=8)

    def test_inf_raises(self):
        vec = np.ones(8, dtype=np.float32)
        vec[0] = np.inf
        with pytest.raises(RuntimeError, match="non-finite"):
            check_activation_vector(vec, "p", 9, hidden_size=8)

    def test_all_zero_raises(self):
        with pytest.raises(RuntimeError, match="all-zero"):
            check_activation_vector(np.zeros(8, dtype=np.float32), "p", 9, hidden_size=8)

    def test_wrong_shape_raises(self):
        with pytest.raises(RuntimeError, match="shape"):
            check_activation_vector(np.ones(7, dtype=np.float32), "p", 9, hidden_size=8)


# --------------------------------------------------------------------------
# Verdict: the go/no-go branch as an executable table
# --------------------------------------------------------------------------

class TestVerdict:
    @pytest.mark.parametrize("auc,floor_mean,floor_p95,expected", [
        (0.90, 0.50, 0.70, "GO"),        # clears p95
        (0.71, 0.50, 0.70, "GO"),        # just clears p95
        (0.70, 0.50, 0.70, "MARGINAL"),  # ties with p95 -> NOT a GO
        (0.65, 0.50, 0.70, "MARGINAL"),  # above mean, below p95
        (0.50, 0.50, 0.70, "NO-GO"),     # ties with mean -> NOT marginal
        (0.45, 0.50, 0.70, "NO-GO"),     # below mean
        (0.99, 0.995, 0.999, "NO-GO"),   # high AUC still loses to a higher floor
    ])
    def test_verdict_table(self, auc, floor_mean, floor_p95, expected):
        v = verdict_for(auc, floor_mean, floor_p95, frac_floor_below=0.5, n_test=7)
        assert v.startswith(expected), f"verdict_for({auc}, {floor_mean}, {floor_p95}) -> {v}"

    def test_marginal_reports_fraction_and_n(self):
        v = verdict_for(0.65, 0.50, 0.70, frac_floor_below=0.83, n_test=7)
        assert "83%" in v and "n_test=7" in v


# --------------------------------------------------------------------------
# Shuffled-label floor: known-answer behavior + max-statistic honesty
# --------------------------------------------------------------------------

def _probe_data(n=60, d=8, separable=False, seed=0):
    """Synthetic activations: alternating labels, optional separable feature."""
    rng = np.random.default_rng(seed)
    y = np.array([i % 2 == 0 for i in range(n)])
    X = rng.normal(size=(n, d))
    if separable:
        X[:, 0] += np.where(y, 3.0, -3.0)
    return X, y


class TestShuffledFloor:
    def test_floor_on_noise_is_near_chance(self):
        # Known answer: with pure-noise features the floor must sit near 0.5.
        X, y = _probe_data(separable=False)
        floors = shuffled_label_floor_max(
            [X[:40]], y[:40], [X[40:]], y[40:], n_seeds=40, seed=0
        )
        assert len(floors) == 40
        assert 0.35 < float(np.mean(floors)) < 0.65

    def test_floor_on_separable_data_stays_low_while_real_probe_is_high(self):
        # THE mutation-killer: if shuffling is disabled (y_shuf = y_train.copy()),
        # every "floor" seed becomes the real probe (~1.0 here) and both
        # assertions below fail.
        X, y = _probe_data(separable=True)
        X_tr, y_tr, X_te, y_te = X[:40], y[:40], X[40:], y[40:]

        real_auc, _ = fit_and_auc(X_tr, y_tr, X_te, y_te, seed=0)
        assert real_auc > 0.9  # sanity: the signal is really there

        floors = shuffled_label_floor_max([X_tr], y_tr, [X_te], y_te, n_seeds=40, seed=0)
        assert float(np.mean(floors)) < 0.75      # shuffling destroyed the signal
        assert real_auc > float(np.percentile(floors, 95))

    def test_max_statistic_floor_is_higher_than_single_layer_floor(self):
        # Multiple-comparisons honesty: best-of-3-layers selection lifts the
        # floor; comparing best-of-3 against a single-probe floor flatters
        # the probe. E[max of 3] > E[single] on independent noise.
        y = np.array([i % 2 == 0 for i in range(60)])
        layers = [np.random.default_rng(s).normal(size=(60, 8)) for s in (1, 2, 3)]
        tr, te = slice(0, 40), slice(40, 60)

        floor_1 = shuffled_label_floor_max(
            [layers[0][tr]], y[tr], [layers[0][te]], y[te], n_seeds=40, seed=0
        )
        floor_3 = shuffled_label_floor_max(
            [Xl[tr] for Xl in layers], y[tr], [Xl[te] for Xl in layers], y[te],
            n_seeds=40, seed=0,
        )
        assert float(np.mean(floor_3)) > float(np.mean(floor_1))


class TestBootstrapCap:
    def test_bootstrap_returns_ordered_ci(self):
        rng = np.random.default_rng(0)
        y = np.array([True] * 5 + [False] * 5)
        scores = rng.uniform(size=10)
        lo, hi = bootstrap_ci(y, scores, n_bootstrap=50, seed=0)
        assert 0.0 <= lo <= hi <= 1.0

    def test_bootstrap_halts_instead_of_spinning_on_degenerate_test_set(self):
        # Single-class test set: every resample is invalid; the old code
        # would loop forever. The cap must convert that into a HALT.
        y = np.ones(6, dtype=bool)
        scores = np.linspace(0.1, 0.9, 6)
        with pytest.raises(SystemExit, match="HALT"):
            bootstrap_ci(y, scores, n_bootstrap=10, seed=0, max_attempts=50)


# --------------------------------------------------------------------------
# Stage round-trip: synthetic traces.jsonl -> truncate.main -> stage-3 loader
# --------------------------------------------------------------------------

# Every key stage 3 (harvest) reads off an included prefix row.
STAGE3_REQUIRED_KEYS = {"problem_id", "included", "label", "prefix_token_ids", "prompt_token_ids"}


def _write_synthetic_traces(path) -> dict:
    """A traces.jsonl exercising every partition; returns expectations."""
    rows = [
        {"problem_id": "good_true_1", "prompt_token_ids": PROMPT,
         "trace_token_ids": make_trace(20), "correct": True},
        {"problem_id": "good_true_2", "prompt_token_ids": PROMPT,
         "trace_token_ids": make_trace(30), "correct": True},
        {"problem_id": "good_false_1", "prompt_token_ids": PROMPT,
         "trace_token_ids": make_trace(16), "correct": False},
        {"problem_id": "incomplete_1", "prompt_token_ids": PROMPT,
         "trace_token_ids": make_trace(50, with_end=False, n_answer=0), "correct": None},
        {"problem_id": "short_1", "prompt_token_ids": PROMPT,
         "trace_token_ids": make_trace(2), "correct": True},
        {"problem_id": "ungradeable_1", "prompt_token_ids": PROMPT,
         "trace_token_ids": make_trace(20), "correct": None},
    ]
    import json

    with open(path, "w") as f:
        f.write(json.dumps({"record_type": "meta", "stage": "generate_traces",
                            "config_hash": CONFIG.config_hash()}) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return {
        "included": {"good_true_1", "good_true_2", "good_false_1"},
        "excluded": {"incomplete_1": "truncated_incomplete",
                     "short_1": "thinking_too_short",
                     "ungradeable_1": "ungradeable"},
        "labels": {"good_true_1": True, "good_true_2": True, "good_false_1": False},
    }


class TestStageRoundTrip:
    @pytest.fixture()
    def roundtrip(self, tmp_path, monkeypatch):
        import dataclasses

        cfg = dataclasses.replace(
            CONFIG,
            traces_path=str(tmp_path / "traces.jsonl"),
            prefixes_path=str(tmp_path / "prefixes.jsonl"),
        )
        monkeypatch.setattr(truncate_module, "CONFIG", cfg)
        expected = _write_synthetic_traces(cfg.traces_path)
        truncate_module.main()
        meta, included_rows = harvest_activations.load_included_prefixes(cfg.prefixes_path)
        return cfg, expected, meta, included_rows

    def test_included_excluded_partition_end_to_end(self, roundtrip):
        cfg, expected, meta, included_rows = roundtrip
        assert {r["problem_id"] for r in included_rows} == expected["included"]
        assert meta["n_included"] == len(expected["included"])
        assert meta["exclusion_counts"] == {
            "truncated_incomplete": 1, "thinking_too_short": 1, "ungradeable": 1,
        }
        # excluded rows are still IN the file (with reasons), just not loaded
        import json

        all_rows = [json.loads(l) for l in open(cfg.prefixes_path)][1:]
        assert len(all_rows) == 6
        by_id = {r["problem_id"]: r for r in all_rows}
        for pid, reason in expected["excluded"].items():
            assert by_id[pid]["included"] is False
            assert by_id[pid]["exclusion_reason"] == reason

    def test_stage3_schema_keys_present(self, roundtrip):
        _, expected, meta, included_rows = roundtrip
        for r in included_rows:
            assert STAGE3_REQUIRED_KEYS <= set(r.keys()), (
                f"{r['problem_id']}: missing stage-3 keys "
                f"{STAGE3_REQUIRED_KEYS - set(r.keys())}"
            )
            assert r["config_hash"] == CONFIG.config_hash()
            assert r["label"] == expected["labels"][r["problem_id"]]

    def test_stage3_invariants_hold_on_roundtripped_rows(self, roundtrip):
        # The exact checks harvest_activations.main() re-runs before spending
        # GPU time: prompt-prefix alignment and no </think> in the prefix.
        _, _, meta, included_rows = roundtrip
        for r in included_rows:
            pids = r["prompt_token_ids"]
            assert r["prefix_token_ids"][: len(pids)] == pids
            assert THINK_END_ID not in r["prefix_token_ids"][len(pids):]
        # lineage: the meta line ties the output to config + input file
        assert meta["config_hash"] == CONFIG.config_hash()
        assert meta["input_file"].endswith("traces.jsonl")


# --------------------------------------------------------------------------
# Text floor: feature contract + split reuse
# --------------------------------------------------------------------------

class TestTextFloor:
    def _rows(self):
        return [
            {"problem_id": "a", "prefix_token_ids": list(range(10)), "label": True},
            {"problem_id": "b", "prefix_token_ids": list(range(25)), "label": False},
            {"problem_id": "c", "prefix_token_ids": list(range(7)), "label": True},
        ]

    def test_features_are_prefix_len_and_level(self):
        pids, X, y = build_feature_rows(self._rows(), {"a": 4, "b": 5, "c": 4})
        assert pids == ["a", "b", "c"]
        assert X.shape == (3, 2)
        assert X[:, 0].tolist() == [10.0, 25.0, 7.0]   # prefix token count
        assert X[:, 1].tolist() == [4.0, 5.0, 4.0]     # problem level
        assert y.tolist() == [True, False, True]

    def test_missing_level_raises(self):
        with pytest.raises(RuntimeError, match="no problem level"):
            build_feature_rows(self._rows(), {"a": 4, "c": 4})

    def test_split_reuses_probe_partition_exactly(self):
        pids = ["a", "b", "c", "d"]
        results = {"split": {"train_problem_ids": ["a", "c"], "test_problem_ids": ["b", "d"]}}
        train_idx, test_idx = split_indices_from_results(pids, results)
        assert train_idx.tolist() == [0, 2]
        assert test_idx.tolist() == [1, 3]

    def test_split_halts_on_out_of_sync_results(self):
        results = {"split": {"train_problem_ids": ["a"], "test_problem_ids": ["b"]}}
        with pytest.raises(RuntimeError, match="out of sync"):
            split_indices_from_results(["a", "b", "mystery"], results)
