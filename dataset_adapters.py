"""Dataset adapters: raw HF row -> the one normalized record the pipeline uses.

Pure stdlib on purpose. This module must import (and be fully testable) on a
machine with no torch, no vllm, no `datasets`, no network — stage 1 imports it
next to the heavy libraries, but the invariant suite imports it alone.

Why this exists: the pipeline was written against MATH-500's schema
(`unique_id` / `problem` / `level` / `subject` / `answer`). MMLU-Pro was adopted
on measured evidence (RESULTS.md Run 004: 23% error, 0 ungradeable, 12,032
items vs MATH-500's ~11% error and 134 usable items), so the row->record
mapping is now the only dataset-specific code, and it is a pure function.

Contract — every adapter returns a NormalizedRecord with:
    problem_id  str  stable per-problem id (str even when the source is an int,
                     so downstream group-splitting keys never change type)
    prompt_text str  the user message handed to apply_chat_template ONCE
    gold_answer str  what grading.grade() compares the \\boxed{...} against
    meta        dict dataset-specific extras; always carries "level" and
                     "subject" keys (None where the dataset has no such field)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

# MMLU-Pro has up to 10 options per question.
OPTION_LABELS = "ABCDEFGHIJ"

MMLU_PRO_INSTRUCTION = (
    "Reason step by step, then give your final answer as a single letter "
    "in \\boxed{}, for example \\boxed{A}."
)


@dataclass(frozen=True)
class NormalizedRecord:
    problem_id: str
    prompt_text: str
    gold_answer: str
    meta: Dict[str, Any] = field(default_factory=dict)


class DatasetAdapter:
    """Base class. `kind` is what config.dataset_kind selects on."""

    kind: str = ""
    default_dataset_id: str = ""
    default_split: str = "test"

    def keep_row(self, row: Dict[str, Any], levels: Tuple[int, ...]) -> bool:
        """Row filter applied before sampling. Default: keep everything."""
        return True

    def adapt(self, row: Dict[str, Any]) -> NormalizedRecord:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# MATH-500 (the original dataset; kept selectable so Runs 001-003 reproduce)
# ---------------------------------------------------------------------------

def parse_level(value: Any) -> Optional[int]:
    """Normalize the MATH-500 `level` field (int, or strings like 'Level 4')."""
    import re

    if isinstance(value, int):
        return value
    m = re.search(r"\d+", str(value))
    return int(m.group()) if m else None


class Math500Adapter(DatasetAdapter):
    kind = "math500"
    default_dataset_id = "HuggingFaceH4/MATH-500"
    default_split = "test"

    def keep_row(self, row: Dict[str, Any], levels: Tuple[int, ...]) -> bool:
        # Decision A (2026-08-23): levels 4-5 only. Refuted for the science
        # (Run 003) but kept faithful for the math500 path.
        if not levels:
            return True
        return parse_level(row["level"]) in tuple(levels)

    def adapt(self, row: Dict[str, Any]) -> NormalizedRecord:
        return NormalizedRecord(
            problem_id=str(row["unique_id"]),
            prompt_text=str(row["problem"]),
            gold_answer=str(row["answer"]),
            meta={
                "level": row.get("level"),
                "subject": row.get("subject"),
                "dataset_kind": self.kind,
            },
        )


# ---------------------------------------------------------------------------
# MMLU-Pro (adopted 2026-08-26 on a measured base rate)
# ---------------------------------------------------------------------------

class MmluProAdapter(DatasetAdapter):
    kind = "mmlu_pro"
    default_dataset_id = "TIGER-Lab/MMLU-Pro"
    default_split = "test"

    # No level field exists; every row is kept and the level filter is a no-op.
    def keep_row(self, row: Dict[str, Any], levels: Tuple[int, ...]) -> bool:
        return True

    def adapt(self, row: Dict[str, Any]) -> NormalizedRecord:
        options = list(row["options"])
        if not options:
            raise ValueError(f"{row.get('question_id')}: MMLU-Pro row has no options")
        if len(options) > len(OPTION_LABELS):
            raise ValueError(
                f"{row.get('question_id')}: {len(options)} options exceeds the "
                f"{len(OPTION_LABELS)} labels A-{OPTION_LABELS[-1]}"
            )

        # Options are rendered verbatim and never filtered: the gold letter is
        # positional (answer_index), so dropping any entry would silently
        # re-letter the choices and mislabel the run.
        lines = [
            f"{OPTION_LABELS[i]}. {opt}" for i, opt in enumerate(options)
        ]
        prompt_text = (
            f"{str(row['question']).strip()}\n\n"
            f"Options:\n" + "\n".join(lines) + "\n\n"
            f"{MMLU_PRO_INSTRUCTION}"
        )

        gold = str(row.get("answer") or "").strip().upper()
        idx = row.get("answer_index")
        # NB: `"" in "ABCD"` is True, so the length check is load-bearing.
        if not (len(gold) == 1 and gold in OPTION_LABELS[: len(options)]):
            # Fall back to the positional index rather than grading against junk.
            if isinstance(idx, int) and 0 <= idx < len(options):
                gold = OPTION_LABELS[idx]
            else:
                raise ValueError(
                    f"{row.get('question_id')}: unusable answer {row.get('answer')!r} "
                    f"/ answer_index {idx!r} for {len(options)} options"
                )
        if isinstance(idx, int) and 0 <= idx < len(options) and OPTION_LABELS[idx] != gold:
            raise ValueError(
                f"{row.get('question_id')}: answer {gold!r} disagrees with "
                f"answer_index {idx} ({OPTION_LABELS[idx]!r})"
            )

        category = row.get("category")
        return NormalizedRecord(
            problem_id=str(row["question_id"]),
            prompt_text=prompt_text,
            gold_answer=gold,
            meta={
                "level": None,          # MMLU-Pro has no difficulty field
                "subject": category,    # kept under the old key for downstream code
                "category": category,
                "src": row.get("src"),
                "n_options": len(options),
                "answer_index": idx,
                "dataset_kind": self.kind,
            },
        )


ADAPTERS: Dict[str, DatasetAdapter] = {
    a.kind: a for a in (Math500Adapter(), MmluProAdapter())
}


def get_adapter(dataset_kind: str) -> DatasetAdapter:
    try:
        return ADAPTERS[dataset_kind]
    except KeyError:
        raise SystemExit(
            f"HALT: unknown dataset_kind {dataset_kind!r}; "
            f"expected one of {sorted(ADAPTERS)}"
        )
