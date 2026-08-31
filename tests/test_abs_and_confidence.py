"""Invariant tests for the two Run-008/009 capabilities.

Same hard constraints as the rest of the suite (it is the pre-flight gate that
runs before any GPU is rented): no torch / vllm / transformers, no network, no
writes outside pytest's tmp_path.

What is tested is what could silently corrupt the two new headline numbers:

  1. `truncate_abs` — the cut lands EXACTLY N thinking tokens; the population is
     byte-identical at every N; no prefix contains `</think>`; the population
     filter runs BEFORE (and independently of) any per-cut decision; the config
     hash moves with the absolute-N setting; the emitted prefixes.jsonl is a
     superset of what harvest/train_probe/the text baselines already consume.

  2. `forced_confidence` — the score is a function of the prefix alone. Enforced
     both by inspection (no grading import, no gold-shaped parameter in the
     scoring path) and empirically: rescoring with deliberately CORRUPTED gold
     answers leaves every score and every AUC bit-identical.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import math
import pathlib
import re
import subprocess
import sys

import numpy as np
import pytest

from experiment import analysis, forced_confidence, truncate, truncate_abs
from experiment.config import CONFIG, THINK_END_ID, THINK_START_ID, Config

PROMPT = [1000, 1001, 1002, 1003]
ABS_GRID = (64, 128, 256, 512, 1024)


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


def trace_row(pid: str, n_think: int, correct, **kw) -> dict:
    return {
        "problem_id": pid,
        "prompt_token_ids": PROMPT,
        "trace_token_ids": make_trace(n_think, **kw),
        "correct": correct,
    }


def population_fixture_rows():
    """A traces.jsonl exercising every partition of the absolute-cut population.

    Deliberately built so that the naive implementation ("skip traces shorter
    than N") would produce a DIFFERENT, more-correct-skewed population at each
    N: the short traces here are all correct.
    """
    rows = [
        # --- in the population: >= 1024 thinking tokens, complete, gradeable --
        trace_row("long_true_1", 1024, True),     # exactly at the boundary
        trace_row("long_true_2", 2000, True),
        trace_row("long_false_1", 1500, False),
        trace_row("long_false_2", 1100, False),
        # --- out: too short. NOTE all of these are CORRECT, which is exactly
        #     why a per-cut length exclusion would be label-correlated. --------
        trace_row("short_true_100", 100, True),
        trace_row("short_true_300", 300, True),
        trace_row("short_true_1023", 1023, True),
        # --- out: for reasons that have nothing to do with N -----------------
        trace_row("incomplete_1", 4000, None, with_end=False, n_answer=0),
        trace_row("ungradeable_1", 1200, None),
        trace_row("degenerate_1", 2, True),
    ]
    return rows


IN_POPULATION = {"long_true_1", "long_true_2", "long_false_1", "long_false_2"}


def write_traces(path, rows) -> None:
    with open(path, "w") as f:
        f.write(json.dumps({"record_type": "meta", "stage": "generate_traces",
                            "config_hash": CONFIG.config_hash()}) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")


def abs_config(tmp_path, n: int) -> Config:
    return dataclasses.replace(
        CONFIG,
        truncation_abs_n=n,
        truncation_k_percent=n,       # the cut label, exactly as _from_env sets it
        abs_population_min_thinking=1024,
        traces_path=str(tmp_path / "traces.jsonl"),
        prefixes_path=str(tmp_path / f"prefixes_abs{n}.jsonl"),
    )


def run_truncate_abs(tmp_path, monkeypatch, n: int):
    from experiment import config as config_module

    cfg = abs_config(tmp_path, n)
    monkeypatch.setattr(truncate_abs, "CONFIG", cfg)
    # `lineage()` reads the module-global CONFIG, so patch that too or the meta
    # line and the row stamps would disagree about which config produced them.
    monkeypatch.setattr(config_module, "CONFIG", cfg)
    truncate_abs.main()
    rows = [json.loads(l) for l in open(cfg.prefixes_path)]
    return cfg, rows[0], rows[1:]


# ===========================================================================
# 0. The GPU-free / network-free contract
# ===========================================================================

HEAVY = ("torch", "vllm", "transformers", "anthropic", "matplotlib")


def test_new_modules_do_not_pull_in_gpu_or_api_stacks():
    """Both new modules must import clean on a laptop, in a FRESH interpreter."""
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    code = (
        "import sys\n"
        "import experiment.truncate_abs, experiment.forced_confidence\n"
        f"print([m for m in {HEAVY!r} if m in sys.modules])\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=repo_root, capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "[]", (
        f"heavy modules imported at module scope: {proc.stdout.strip()}"
    )


# ===========================================================================
# 1. truncate_abs — the cut lands EXACTLY N thinking tokens
# ===========================================================================

@pytest.mark.parametrize("n", ABS_GRID)
@pytest.mark.parametrize("n_think", [1024, 1025, 1500, 4000])
def test_cut_keeps_exactly_n_thinking_tokens(n, n_think):
    res = truncate_abs.build_abs_prefix(
        "p", PROMPT, make_trace(n_think), True, abs_n=n, population_min_thinking=1024
    )
    assert res.included
    assert res.n_kept_thinking_tokens == n
    # and the prefix really is prompt + <think> + exactly n thinking tokens
    assert len(res.prefix_token_ids) == len(PROMPT) + 1 + n
    assert res.prefix_token_ids[: len(PROMPT)] == PROMPT
    assert res.prefix_token_ids[len(PROMPT)] == THINK_START_ID
    assert res.prefix_token_ids[len(PROMPT) + 1:] == [2000 + i for i in range(n)]


@pytest.mark.parametrize("n", ABS_GRID)
def test_cut_is_exact_when_the_template_already_opened_think(n):
    """Completion begins directly with thinking tokens (no `<think>` emitted)."""
    res = truncate_abs.build_abs_prefix(
        "p", PROMPT, make_trace(1500, with_start=False), True,
        abs_n=n, population_min_thinking=1024,
    )
    assert res.included and res.n_kept_thinking_tokens == n
    assert len(res.prefix_token_ids) == len(PROMPT) + n


@pytest.mark.parametrize("n", ABS_GRID)
def test_no_prefix_ever_contains_the_think_end_token(tmp_path, monkeypatch, n):
    write_traces(tmp_path / "traces.jsonl", population_fixture_rows())
    _cfg, _meta, rows = run_truncate_abs(tmp_path, monkeypatch, n)
    included = [r for r in rows if r["included"]]
    assert included
    for r in included:
        assert THINK_END_ID not in r["prefix_token_ids"], (
            f"{r['problem_id']}: </think> ({THINK_END_ID}) leaked into the prefix"
        )
        # the same check harvest_activations.main() re-runs before spending GPU
        assert r["prefix_token_ids"][: len(r["prompt_token_ids"])] == r["prompt_token_ids"]
        assert THINK_END_ID not in r["prefix_token_ids"][len(r["prompt_token_ids"]):]


def test_prefix_lengths_are_identical_within_a_cut(tmp_path, monkeypatch):
    """The whole point: within a cut, length carries ZERO information.

    Run 007's leak was corr(prefix length, full trace length) = 0.99999999.
    Here the variance of the kept-token count is exactly zero, so that
    correlation is undefined rather than ~1 — the leak is structurally gone.
    """
    write_traces(tmp_path / "traces.jsonl", population_fixture_rows())
    for n in ABS_GRID:
        _cfg, _meta, rows = run_truncate_abs(tmp_path, monkeypatch, n)
        kept = {r["n_kept_thinking_tokens"] for r in rows if r["included"]}
        assert kept == {n}
        # full trace lengths still vary across the population, so this is not
        # a vacuous check
        full = {r["n_thinking_tokens"] for r in rows if r["included"]}
        assert len(full) > 1


# ===========================================================================
# 2. THE TRAP: the population is fixed once, identical at every N
# ===========================================================================

def test_population_filter_takes_no_cut_size_argument():
    """The structural guarantee, asserted by signature.

    If `population_status` ever grows an `n`/`abs_n` parameter, membership can
    depend on the cut, exclusions become label-correlated (short traces are
    disproportionately correct) and the cuts stop being comparable. That is the
    bias decision B removed; it must not come back through this door.
    """
    params = set(inspect.signature(truncate_abs.population_status).parameters)
    assert params == {
        "trace_token_ids", "label", "population_min_thinking", "min_thinking_tokens"
    }
    assert not (params & {"n", "abs_n", "cut", "k", "k_percent"})


def test_population_is_identical_across_every_n(tmp_path, monkeypatch):
    write_traces(tmp_path / "traces.jsonl", population_fixture_rows())
    populations = {}
    balances = {}
    for n in ABS_GRID:
        _cfg, meta, rows = run_truncate_abs(tmp_path, monkeypatch, n)
        populations[n] = tuple(sorted(r["problem_id"] for r in rows if r["included"]))
        balances[n] = (
            sum(1 for r in rows if r["included"] and r["label"]),
            sum(1 for r in rows if r["included"] and not r["label"]),
        )
        assert tuple(meta["population_problem_ids"]) == populations[n]

    assert len(set(populations.values())) == 1, populations
    assert set(populations[ABS_GRID[0]]) == IN_POPULATION
    assert len(set(balances.values())) == 1, balances
    # the population is long-trace-only, so the labels are NOT the dataset's
    assert balances[ABS_GRID[0]] == (2, 2)


def test_short_traces_are_excluded_once_by_the_population_not_per_cut(tmp_path, monkeypatch):
    """A 300-token trace must be absent even at N=64, which it could have served.

    Including it at small N and dropping it at large N is precisely the
    label-correlated survivor bias; the fixture makes all short traces correct
    so the naive version would inflate the positive rate at small N.
    """
    write_traces(tmp_path / "traces.jsonl", population_fixture_rows())
    _cfg, meta, rows = run_truncate_abs(tmp_path, monkeypatch, 64)
    by_id = {r["problem_id"]: r for r in rows}
    for pid in ("short_true_100", "short_true_300", "short_true_1023"):
        assert by_id[pid]["included"] is False
        assert by_id[pid]["exclusion_reason"] == "below_population_min"
    assert meta["exclusion_counts"]["below_population_min"] == 3


def test_every_exclusion_reason_is_population_level(tmp_path, monkeypatch):
    write_traces(tmp_path / "traces.jsonl", population_fixture_rows())
    _cfg, meta, rows = run_truncate_abs(tmp_path, monkeypatch, 256)
    reasons = {r["exclusion_reason"] for r in rows if not r["included"]}
    assert reasons <= set(truncate_abs.POPULATION_EXCLUSION_REASONS)
    assert meta["exclusion_counts"] == {
        "below_population_min": 3,
        "truncated_incomplete": 1,
        "ungradeable": 1,
        "thinking_too_short": 1,
    }


def test_population_report_agrees_with_the_per_row_build():
    rows = population_fixture_rows()
    rep = truncate_abs.population_report(rows)
    assert set(rep["population_problem_ids"]) == IN_POPULATION
    assert (rep["n_pos"], rep["n_neg"]) == (2, 2)
    for n in ABS_GRID:
        built = {
            r["problem_id"]
            for r in rows
            if truncate_abs.build_abs_prefix(
                r["problem_id"], r["prompt_token_ids"], r["trace_token_ids"],
                r["correct"], abs_n=n,
            ).included
        }
        assert built == IN_POPULATION


def test_underpowered_population_is_flagged_loudly(tmp_path, monkeypatch, capsys):
    """Under ~25 surviving negatives the grid is underpowered; say so."""
    rows = population_fixture_rows()
    rep = truncate_abs.population_report(rows)
    assert rep["underpowered"] is True          # 2 negatives in the fixture
    assert rep["min_negatives_for_power"] == 25

    write_traces(tmp_path / "traces.jsonl", rows)
    run_truncate_abs(tmp_path, monkeypatch, 128)
    err = capsys.readouterr().err
    assert "UNDERPOWERED" in err and "2 negatives" in err

    plenty = rows + [trace_row(f"neg_{i}", 1200, False) for i in range(30)]
    assert truncate_abs.population_report(plenty)["underpowered"] is False


def test_a_cut_larger_than_the_population_minimum_is_refused():
    """An N above max(N) would silently become a per-cut, length-correlated drop."""
    with pytest.raises(ValueError, match="population is fixed"):
        truncate_abs.build_abs_prefix(
            "p", PROMPT, make_trace(4000), True, abs_n=2048, population_min_thinking=1024
        )
    with pytest.raises(ValueError, match=">= 1"):
        truncate_abs.build_abs_prefix("p", PROMPT, make_trace(4000), True, abs_n=0)


def test_config_refuses_abs_n_above_the_population_minimum(monkeypatch):
    from experiment import config as config_module

    monkeypatch.setenv("EXPERIMENT_ABS_N", "2048")
    monkeypatch.delenv("EXPERIMENT_K", raising=False)
    monkeypatch.delenv("EXPERIMENT_ABS_POP_MIN", raising=False)
    with pytest.raises(SystemExit, match="exceeds the fixed population minimum"):
        config_module._from_env()


def test_config_refuses_both_cut_modes_at_once(monkeypatch):
    from experiment import config as config_module

    monkeypatch.setenv("EXPERIMENT_ABS_N", "256")
    monkeypatch.setenv("EXPERIMENT_K", "50")
    with pytest.raises(SystemExit, match="mutually exclusive"):
        config_module._from_env()


def test_fraction_stage_refuses_to_run_under_absolute_settings(tmp_path, monkeypatch):
    """truncate.py would read the cut LABEL (e.g. 1024) as "1024 percent"."""
    monkeypatch.setattr(
        truncate, "CONFIG", dataclasses.replace(CONFIG, truncation_abs_n=256,
                                                truncation_k_percent=256)
    )
    with pytest.raises(SystemExit, match="truncate_abs"):
        truncate.main()


# ===========================================================================
# 3. truncate_abs agrees with truncate.py about where the thinking span is
# ===========================================================================

@pytest.mark.parametrize("with_start", [True, False])
@pytest.mark.parametrize("n_think", [1024, 1500, 4000])
def test_span_extraction_matches_truncate_build_prefix(with_start, n_think):
    """Guards against the two stages drifting apart on token-boundary logic."""
    trace = make_trace(n_think, with_start=with_start)
    head, thinking = truncate_abs.thinking_span(trace)
    assert len(thinking) == n_think
    assert THINK_END_ID not in head + thinking

    # truncate.build_prefix at a k% that lands on a round number of tokens must
    # produce byte-identical output to the absolute cut at that same count.
    ref = truncate.build_prefix("p", PROMPT, trace, True, k_percent=50,
                                min_thinking_tokens=4)
    got = truncate_abs.build_abs_prefix(
        "p", PROMPT, trace, True, abs_n=ref.n_kept_thinking_tokens,
        population_min_thinking=ref.n_kept_thinking_tokens,
    )
    assert got.prefix_token_ids == ref.prefix_token_ids
    assert got.n_thinking_tokens == ref.n_thinking_tokens


def test_incomplete_trace_has_no_span():
    assert truncate_abs.thinking_span(make_trace(50, with_end=False, n_answer=0)) is None


# ===========================================================================
# 4. Config hash moves with the absolute-N setting
# ===========================================================================

def test_config_hash_changes_with_the_absolute_cut_setting():
    base = Config()
    assert base.truncation_abs_n is None
    seen = {base.config_hash()}
    for n in ABS_GRID:
        h = dataclasses.replace(base, truncation_abs_n=n).config_hash()
        assert h not in seen, f"abs_n={n} did not move the config hash"
        seen.add(h)


def test_config_hash_changes_with_the_population_minimum():
    base = Config()
    assert (
        dataclasses.replace(base, abs_population_min_thinking=512).config_hash()
        != base.config_hash()
    )
    assert (
        dataclasses.replace(base, abs_n_grid=(64, 128)).config_hash() != base.config_hash()
    )


def test_absolute_and_fractional_cuts_of_the_same_number_hash_differently():
    """`EXPERIMENT_K=64` and `EXPERIMENT_ABS_N=64` are different experiments."""
    frac = dataclasses.replace(Config(), truncation_k_percent=64)
    absolute = dataclasses.replace(Config(), truncation_k_percent=64, truncation_abs_n=64)
    assert frac.config_hash() != absolute.config_hash()


def test_env_override_sets_both_the_mode_and_the_cut_label(monkeypatch):
    from experiment import config as config_module

    monkeypatch.setenv("EXPERIMENT_ABS_N", "512")
    monkeypatch.delenv("EXPERIMENT_K", raising=False)
    monkeypatch.delenv("EXPERIMENT_ABS_POP_MIN", raising=False)
    cfg = config_module._from_env()
    assert cfg.truncation_abs_n == 512
    # the cut label, so probe row keys (`pid@k512`) and baseline row keys agree
    assert cfg.truncation_k_percent == 512
    assert cfg.config_hash() != Config().config_hash()


def test_row_config_hash_is_stamped_and_matches_the_run(tmp_path, monkeypatch):
    write_traces(tmp_path / "traces.jsonl", population_fixture_rows())
    cfg, meta, rows = run_truncate_abs(tmp_path, monkeypatch, 256)
    assert meta["config_hash"] == cfg.config_hash()
    assert all(r["config_hash"] == cfg.config_hash() for r in rows)
    assert meta["abs_n"] == 256 and meta["cut_mode"] == "absolute"


# ===========================================================================
# 5. Schema compatibility of prefixes.jsonl with the existing consumers
# ===========================================================================

STAGE3_REQUIRED_KEYS = {
    "problem_id", "included", "label", "prefix_token_ids", "prompt_token_ids"
}
TRUNCATE_ROW_KEYS = {
    "problem_id", "included", "exclusion_reason", "label", "prefix_token_ids",
    "prompt_token_ids", "n_thinking_tokens", "n_kept_thinking_tokens",
}


def test_prefixes_jsonl_is_a_superset_of_the_fractional_schema(tmp_path, monkeypatch):
    write_traces(tmp_path / "traces.jsonl", population_fixture_rows())
    _cfg, meta, rows = run_truncate_abs(tmp_path, monkeypatch, 256)
    for r in rows:
        assert TRUNCATE_ROW_KEYS <= set(r)
    for r in (r for r in rows if r["included"]):
        assert STAGE3_REQUIRED_KEYS <= set(r)
    assert meta["record_type"] == "meta"
    assert "k_percent" in meta and "n_included" in meta and "exclusion_counts" in meta


def test_harvest_activations_loads_the_absolute_prefixes(tmp_path, monkeypatch):
    from experiment import harvest_activations

    write_traces(tmp_path / "traces.jsonl", population_fixture_rows())
    cfg, _meta, _rows = run_truncate_abs(tmp_path, monkeypatch, 128)
    meta, included = harvest_activations.load_included_prefixes(cfg.prefixes_path)
    assert {r["problem_id"] for r in included} == IN_POPULATION
    assert meta["input_file"].endswith("traces.jsonl")


def test_text_baselines_read_the_cut_label_from_the_meta_line(tmp_path, monkeypatch):
    """`forced_answer`'s shared prefix I/O keys every row by the cut."""
    from experiment import forced_answer

    write_traces(tmp_path / "traces.jsonl", population_fixture_rows())
    cfg, _meta, _rows = run_truncate_abs(tmp_path, monkeypatch, 512)
    rows, default_k = forced_answer.load_included_prefix_rows(cfg.prefixes_path)
    assert default_k == 512
    assert {forced_answer.row_key(r, default_k) for r in rows} == {
        f"{pid}@k512" for pid in IN_POPULATION
    }
    # ... and the keys are unique across the grid, so a merged analysis cannot
    # collide two cuts of the same problem
    cfg64, _m, _r = run_truncate_abs(tmp_path, monkeypatch, 64)
    rows64, k64 = forced_answer.load_included_prefix_rows(cfg64.prefixes_path)
    assert not ({forced_answer.row_key(r, k64) for r in rows64} &
                {forced_answer.row_key(r, default_k) for r in rows})


def test_forced_answer_prompt_construction_accepts_an_absolute_prefix(tmp_path, monkeypatch):
    from experiment import forced_answer

    write_traces(tmp_path / "traces.jsonl", population_fixture_rows())
    cfg, _meta, _rows = run_truncate_abs(tmp_path, monkeypatch, 64)
    rows, _k = forced_answer.load_included_prefix_rows(cfg.prefixes_path)
    for r in rows:
        ids = forced_answer.build_forced_prompt_ids(r["prefix_token_ids"], [9, 9, 9])
        assert ids[len(r["prefix_token_ids"])] == THINK_END_ID


def test_main_halts_when_the_absolute_n_is_unset(tmp_path, monkeypatch):
    write_traces(tmp_path / "traces.jsonl", population_fixture_rows())
    monkeypatch.setattr(
        truncate_abs, "CONFIG",
        dataclasses.replace(CONFIG, truncation_abs_n=None,
                            traces_path=str(tmp_path / "traces.jsonl"),
                            prefixes_path=str(tmp_path / "p.jsonl")),
    )
    with pytest.raises(SystemExit, match="EXPERIMENT_ABS_N"):
        truncate_abs.main()


# ===========================================================================
# 6. forced_confidence — pure logprob handling
# ===========================================================================

class FakeLogprob:
    """Stand-in for vllm.sequence.Logprob."""

    def __init__(self, logprob, decoded_token=None):
        self.logprob = logprob
        self.decoded_token = decoded_token


def letter_dist(probs: dict, extra: dict | None = None) -> dict:
    """{'A': 0.7, 'B': 0.2} -> a vLLM-shaped {token_id: Logprob} map."""
    out = {}
    for letter, p in probs.items():
        out[100 + ord(letter)] = FakeLogprob(math.log(p), letter)
    for tid, (text, p) in (extra or {}).items():
        out[tid] = FakeLogprob(math.log(p), text)
    return out


@pytest.mark.parametrize("text,expected", [
    ("A", "A"), (" C", "C"), ("j", "J"), ("  b  ", "B"),
    ("AB", None), ("", None), (None, None), ("\\", None), ("1", None), ("K", None),
])
def test_normalize_answer_token(text, expected):
    assert forced_confidence.normalize_answer_token(text) == expected


def test_extract_answer_distribution_reads_the_emitted_token():
    lp = letter_dist({"C": 0.6, "A": 0.3, "B": 0.1})
    rec = forced_confidence.extract_answer_distribution([100 + ord("C")], [lp])
    assert rec["chosen_letter"] == "C"
    assert rec["answer_position"] == 0
    assert math.exp(rec["chosen_logprob"]) == pytest.approx(0.6)
    assert [r["token"] for r in rec["topk"]] == ["C", "A", "B"]   # sorted best-first
    assert set(rec["letter_logprobs"]) == {"A", "B", "C"}


def test_extract_answer_distribution_skips_a_leading_non_letter():
    stray = {7: FakeLogprob(math.log(0.9), " ")}
    lp0 = {**stray}
    lp1 = letter_dist({"D": 0.8, "E": 0.2})
    rec = forced_confidence.extract_answer_distribution(
        [7, 100 + ord("D")], [lp0, lp1]
    )
    assert rec["answer_position"] == 1 and rec["chosen_letter"] == "D"


def test_extract_answer_distribution_handles_a_missing_distribution():
    assert forced_confidence.extract_answer_distribution([], [{}]) is None
    assert forced_confidence.extract_answer_distribution([1], None) is None


def test_extract_answer_distribution_records_the_full_top_k():
    """Entropy/margin variants must be computable later with no GPU."""
    lp = letter_dist({l: p for l, p in zip("ABCDEFGHIJ", [0.55, 0.2, 0.1, 0.05,
                                                          0.04, 0.03, 0.01, 0.01,
                                                          0.005, 0.005])})
    rec = forced_confidence.extract_answer_distribution([100 + ord("A")], [lp])
    assert len(rec["topk"]) == 10
    assert len(rec["letter_logprobs"]) == 10


def test_confidence_variants_are_all_computable_from_the_record():
    lp = letter_dist({"A": 0.6, "B": 0.3, "C": 0.1})
    rec = forced_confidence.extract_answer_distribution([100 + ord("A")], [lp])
    v = forced_confidence.confidence_variants(rec)
    assert set(v) == set(forced_confidence.CONFIDENCE_VARIANTS)
    assert v["p_top"] == pytest.approx(0.6)
    assert v["logprob_top"] == pytest.approx(math.log(0.6))
    assert v["p_top_letter"] == pytest.approx(0.6)
    assert v["p_letter_norm"] == pytest.approx(0.6)          # letters sum to 1 here
    assert v["letter_margin"] == pytest.approx(0.3)
    expected_neg_h = sum(p * math.log(p) for p in (0.6, 0.3, 0.1))
    assert v["neg_letter_entropy"] == pytest.approx(expected_neg_h)


def test_confidence_variants_renormalise_when_letters_do_not_sum_to_one():
    lp = letter_dist({"A": 0.3, "B": 0.1}, extra={5: ("\n", 0.6)})
    rec = forced_confidence.extract_answer_distribution([100 + ord("A")], [lp])
    v = forced_confidence.confidence_variants(rec)
    assert v["p_top_letter"] == pytest.approx(0.3)
    assert v["p_letter_norm"] == pytest.approx(0.75)
    assert v["letter_margin"] == pytest.approx(0.5)


def test_confidence_variants_are_none_rather_than_imputed_when_absent():
    assert all(v is None for v in forced_confidence.confidence_variants(None).values())
    lonely = {"chosen_logprob": math.log(0.4), "letter_logprobs": {}}
    v = forced_confidence.confidence_variants(lonely)
    assert v["p_top"] == pytest.approx(0.4)
    assert v["p_top_letter"] is None and v["letter_margin"] is None


def test_higher_confidence_is_a_higher_score_correct_orientation():
    """A confident model must score ABOVE a hesitant one, or the AUC inverts."""
    confident = forced_confidence.extract_answer_distribution(
        [100 + ord("A")], [letter_dist({"A": 0.95, "B": 0.05})]
    )
    hesitant = forced_confidence.extract_answer_distribution(
        [100 + ord("A")], [letter_dist({"A": 0.35, "B": 0.33, "C": 0.32})]
    )
    for v in forced_confidence.CONFIDENCE_VARIANTS:
        cv = forced_confidence.confidence_variants(confident)[v]
        hv = forced_confidence.confidence_variants(hesitant)[v]
        assert cv > hv, f"variant {v} is oriented backwards"


# ===========================================================================
# 7. forced_confidence — ADMISSIBILITY: gold never enters the scoring path
# ===========================================================================

SCORING_PATH = (
    forced_confidence.score,
    forced_confidence.score_rows,
    forced_confidence.confidence_variants,
    forced_confidence.extract_answer_distribution,
    forced_confidence.letter_logprobs,
    forced_confidence.token_distribution,
    forced_confidence.normalize_answer_token,
    forced_confidence.selected_variant,
)


def _code_only(fn) -> str:
    """Function source with docstrings and comments stripped.

    The prose in `forced_confidence` mentions gold constantly — explaining why
    it is absent. Only the executable lines are evidence.
    """
    src = inspect.getsource(fn)
    src = re.sub(r'"""(?:.|\n)*?"""', "", src)
    src = re.sub(r"'''(?:.|\n)*?'''", "", src)
    src = "\n".join(line.split("#", 1)[0] for line in src.splitlines())
    return src.replace("gold_free", "")   # the self-describing artifact flag


def test_no_function_in_the_scoring_path_accepts_a_gold_parameter():
    for fn in SCORING_PATH:
        params = set(inspect.signature(fn).parameters)
        offenders = {p for p in params
                     if any(t in p.lower() for t in ("gold", "answer_key", "target",
                                                     "correct", "truth"))}
        assert not offenders, f"{fn.__name__} takes {offenders}"


def test_the_module_never_imports_or_calls_the_grader():
    for name in ("grade", "extract_boxed", "grading", "gold_answers_by_problem",
                 "grade_forced_answer", "forced_answer_text"):
        assert not hasattr(forced_confidence, name), (
            f"forced_confidence.{name} exists — the grader has crept back in"
        )
    for fn in SCORING_PATH:
        body = _code_only(fn).lower()
        assert "grade" not in body, f"{fn.__name__} executes a grader"
        assert "gold" not in body, f"{fn.__name__} touches gold"
    # traces.jsonl is where gold_answer lives; the scoring path must not open it
    assert "traces_path" not in _code_only(forced_confidence.score)
    assert "read_jsonl" in _code_only(forced_confidence.score)  # sanity


def test_the_generate_path_never_looks_up_a_gold_answer():
    body = _code_only(forced_confidence.main).lower()
    assert "gold" not in body, "the GPU stage is loading the answer key"
    assert "grade" not in body
    # ... unlike its withdrawn predecessor, whose generate path does both —
    # this asserts the check above can actually fail, i.e. is not vacuous
    from experiment import forced_answer

    assert "gold" in _code_only(forced_answer.main).lower()


# --- the empirical version of the same claim --------------------------------

def _write_confidence_fixture(tmp_path, probs_and_labels, cut=256):
    """A forced_confidence.jsonl + results.json + a traces.jsonl full of gold."""
    recs = []
    for i, (p, label) in enumerate(probs_and_labels):
        lp = letter_dist({"A": p, "B": round(1.0 - p, 6)})
        dist = forced_confidence.extract_answer_distribution([100 + ord("A")], [lp])
        recs.append({
            "row_key": f"pid{i}@k{cut}",
            "problem_id": f"pid{i}",
            "k_percent": cut,
            "label": label,
            "generated_text": "A}",
            "confidence": dist,
            "variants": forced_confidence.confidence_variants(dist),
            "prefix_text": "...thinking...",
            "config_hash": "deadbeef",
        })
    conf_path = tmp_path / "forced_confidence.jsonl"
    with open(conf_path, "w") as f:
        f.write(json.dumps({"record_type": "meta", "stage": "forced_confidence"}) + "\n")
        for r in recs:
            f.write(json.dumps(r) + "\n")

    pids = [r["problem_id"] for r in recs]
    # interleaved so BOTH classes land in both halves (score_at_k HALTs on a
    # single-class test split, correctly)
    results = {"split": {"train_problem_ids": pids[0::2],
                         "test_problem_ids": pids[1::2]},
               "truncation_k_percent": cut}
    (tmp_path / "results.json").write_text(json.dumps(results))
    return conf_path, recs


def _run_score(tmp_path, monkeypatch):
    cfg = dataclasses.replace(
        CONFIG,
        truncation_k_percent=256, truncation_abs_n=256,
        n_bootstrap=50,
        output_dir=str(tmp_path),
        traces_path=str(tmp_path / "traces.jsonl"),
        results_path=str(tmp_path / "results.json"),
    )
    monkeypatch.setattr(forced_confidence, "CONFIG", cfg)
    monkeypatch.setattr(forced_confidence, "FORCED_CONFIDENCE_PATH",
                        str(tmp_path / "forced_confidence.jsonl"))
    monkeypatch.setattr(analysis, "baseline_path",
                        lambda name: str(tmp_path / f"baseline_{name}.json"))
    forced_confidence.score()
    return json.loads((tmp_path / "baseline_forced_confidence.json").read_text())


CONF_FIXTURE = [
    (0.95, True), (0.90, True), (0.85, True), (0.80, True), (0.75, True), (0.70, True),
    (0.60, False), (0.55, False), (0.50, False), (0.45, False), (0.40, False), (0.35, False),
]


def test_score_is_unchanged_when_the_gold_answers_are_corrupted(tmp_path, monkeypatch):
    """The decisive admissibility test.

    Score once with a correct-looking traces.jsonl, then again after replacing
    every gold answer with garbage. If any part of the scoring path consulted
    the answer key — as the withdrawn forced_answer baseline did — the AUC would
    move. It must not move by a single bit.
    """
    _write_confidence_fixture(tmp_path, CONF_FIXTURE)

    def write_gold(value_for):
        with open(tmp_path / "traces.jsonl", "w") as f:
            f.write(json.dumps({"record_type": "meta", "stage": "generate_traces"}) + "\n")
            for i in range(len(CONF_FIXTURE)):
                f.write(json.dumps({"problem_id": f"pid{i}",
                                    "gold_answer": value_for(i)}) + "\n")

    write_gold(lambda i: "A")
    honest = _run_score(tmp_path, monkeypatch)

    write_gold(lambda i: "ZZZ_NOT_AN_ANSWER")
    corrupted = _run_score(tmp_path, monkeypatch)

    (tmp_path / "traces.jsonl").unlink()          # and with no answer key at all
    absent = _run_score(tmp_path, monkeypatch)

    for other in (corrupted, absent):
        assert other["per_k"]["256"]["auc"] == honest["per_k"]["256"]["auc"]
        assert other["per_k"]["256"]["test_scores"] == honest["per_k"]["256"]["test_scores"]
        assert other["per_k"]["256"]["auc_ci95"] == honest["per_k"]["256"]["auc_ci95"]
    assert honest["per_k"]["256"]["auc"] > 0.5    # sanity: the fixture is informative


def test_score_writes_the_shared_baseline_schema(tmp_path, monkeypatch):
    """Byte-compatible with baseline_text_classifier.json etc. (analysis.py)."""
    _write_confidence_fixture(tmp_path, CONF_FIXTURE)
    payload = _run_score(tmp_path, monkeypatch)

    assert payload["schema_version"] == analysis.BASELINE_SCHEMA_VERSION
    assert payload["baseline"] == "forced_confidence"
    assert payload["metric"] == "roc_auc"
    point = payload["per_k"]["256"]
    for key in ("auc", "auc_ci95", "n_test", "test_row_keys", "test_labels",
                "test_scores"):
        assert key in point, f"missing {key} — analysis.align_scores needs it"
    assert len(point["test_row_keys"]) == point["n_test"] == len(point["test_scores"])
    assert all(k.endswith("@k256") for k in point["test_row_keys"])
    assert payload["notes"]["gold_free"] is True
    assert point["notes"]["variant"] == forced_confidence.DEFAULT_VARIANT
    # every variant's AUC recorded, so entropy/margin need no second GPU run
    assert set(point["notes"]["variant_aucs"]) == set(forced_confidence.CONFIDENCE_VARIANTS)


def test_analysis_reads_the_new_baseline_and_pairs_it_by_row_key(tmp_path, monkeypatch):
    _write_confidence_fixture(tmp_path, CONF_FIXTURE)
    _run_score(tmp_path, monkeypatch)

    assert "forced_confidence" in analysis.TEXT_BASELINE_NAMES
    payload = analysis.read_baseline_json("forced_confidence")
    assert payload is not None
    point = payload["per_k"]["256"]

    # a probe scored on the same rows must pair cleanly (align_scores returns None
    # on a row-key mismatch, which would silently widen every CI)
    probe = {"test_row_keys": point["test_row_keys"],
             "test_labels": point["test_labels"],
             "test_scores": [0.5] * point["n_test"]}
    aligned = analysis.align_scores(probe, point)
    assert aligned is not None
    y, _ps, ts = aligned
    assert len(y) == point["n_test"] and len(ts) == point["n_test"]


def test_score_drops_rows_with_no_score_rather_than_imputing():
    rows = [
        {"variants": {"p_top": 0.9}, "problem_id": "a"},
        {"variants": {"p_top": None}, "problem_id": "b"},
        {"variants": {}, "confidence": None, "problem_id": "c"},
    ]
    kept, scores, dropped = forced_confidence.score_rows(rows, "p_top")
    assert [r["problem_id"] for r in kept] == ["a"]
    assert scores == [0.9] and dropped == 2


def test_selected_variant_rejects_an_unknown_override(monkeypatch):
    monkeypatch.setenv("EXPERIMENT_CONFIDENCE_VARIANT", "vibes")
    with pytest.raises(SystemExit, match="not one of"):
        forced_confidence.selected_variant()
    monkeypatch.setenv("EXPERIMENT_CONFIDENCE_VARIANT", "letter_margin")
    assert forced_confidence.selected_variant() == "letter_margin"


def test_logprob_variant_gives_the_same_auc_as_the_probability_variant(tmp_path, monkeypatch):
    """exp() is monotone, so the pre-registered "logprob" and "probability"
    readings of the same quantity must be the same number."""
    _write_confidence_fixture(tmp_path, CONF_FIXTURE)
    payload = _run_score(tmp_path, monkeypatch)
    aucs = payload["per_k"]["256"]["notes"]["variant_aucs"]
    assert aucs["p_top"] == aucs["logprob_top"]


def test_the_forced_prompt_is_built_from_the_prefix_alone():
    """Structural admissibility: the model sees prefix + </think> + suffix."""
    from experiment import forced_answer

    prefix = PROMPT + [THINK_START_ID] + [2000 + i for i in range(64)]
    suffix = [7, 8, 9]
    ids = forced_answer.build_forced_prompt_ids(prefix, suffix)
    assert ids == prefix + [THINK_END_ID] + suffix
    assert forced_confidence.FORCED_ANSWER_SUFFIX is forced_answer.FORCED_ANSWER_SUFFIX
