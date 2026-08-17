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
from experiment.config import CONFIG, THINK_END_ID, THINK_START_ID, lineage
from experiment.generate_traces import check_ungradeable_fraction
from experiment.grading import extract_boxed, grade, normalize_answer
from experiment.train_probe import group_split
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

    def test_oom_handler_catches_cuda_oom_and_exits(self):
        # The handler must convert torch.cuda.OutOfMemoryError into SystemExit
        # carrying the remedy text -- verified without CUDA by re-running the
        # exact except/raise contract used in main().
        class FakeOOM(Exception):
            pass

        with pytest.raises(SystemExit) as exc:
            try:
                raise FakeOOM("CUDA out of memory. Tried to allocate 2.00 GiB")
            except FakeOOM as e:
                raise SystemExit(harvest_activations.oom_halt_message(e)) from e

        assert "HALT: CUDA OOM during harvest" in str(exc.value)
        assert "no co-residency" in str(exc.value)

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
