"""Answer extraction and grading for MATH-500 traces.

Pure stdlib on purpose: the invariant tests import this on a machine with no
GPU, no torch, no model weights.

Grading contract (I/O matrix row 4): if the final answer cannot be extracted
or compared, the label is None ("ungradeable") — the caller excludes the row
and counts it, never crashes.
"""

from __future__ import annotations

import re
from typing import Optional

_BOXED = re.compile(r"\\boxed\s*")


def extract_boxed(text: str) -> Optional[str]:
    """Return the contents of the LAST \\boxed{...} in `text`, or None.

    Brace-matching by hand because the contents may nest braces
    (e.g. \\boxed{\\frac{1}{2}}).
    """
    last = None
    for m in _BOXED.finditer(text):
        i = m.end()
        if i >= len(text) or text[i] != "{":
            continue
        depth = 0
        for j in range(i, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    last = text[i + 1 : j]
                    break
        # unclosed brace: fall through, keep previous `last`
    return last


def normalize_answer(ans: str) -> str:
    """Conservative LaTeX/string normalization for exact-match comparison."""
    s = ans.strip()
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\;", "").replace("\\ ", "")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "")
    s = s.replace("\\%", "").replace("%", "")
    s = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", s)
    s = s.replace("$", "").replace(" ", "")
    # strip one layer of fully-wrapping braces: {answer} -> answer
    while len(s) >= 2 and s[0] == "{" and s[-1] == "}" and extract_depth_ok(s):
        s = s[1:-1]
    s = s.rstrip(".")
    # canonicalize numbers like "2.0" -> "2", "0.50" -> "0.5"
    try:
        f = float(s)
        if f == int(f):
            s = str(int(f))
        else:
            s = repr(f)
    except (ValueError, OverflowError):
        pass
    return s


def extract_depth_ok(s: str) -> bool:
    """True if the outermost braces of `s` wrap the whole string."""
    depth = 0
    for i, ch in enumerate(s):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and i != len(s) - 1:
                return False
    return depth == 0


def grade(response_text: str, gold_answer: str) -> Optional[bool]:
    """Grade the post-thinking response against the dataset's `answer` field.

    Returns True/False, or None when ungradeable (no \\boxed{...} found).
    """
    pred = extract_boxed(response_text)
    if pred is None:
        return None
    return normalize_answer(pred) == normalize_answer(gold_answer)
