"""Text baseline 2 of 3: LLM JUDGE — the "just ask an LLM" baseline.

Neel Nanda names this baseline explicitly, and the LessWrong "activation
oracles" post is built on it: filter out the cases a capable LLM could call
from the preceding text alone and ~95% of the apparent internal signal
evaporates. So this is not a checkbox — it is the reader that has, in the
published literature, done the most damage to probe results.

A frontier model is shown the EXACT decoded prefix the probe reads (prompt +
`<think>` + the first k% of thinking tokens) and asked for a calibrated
probability that the final answer will be correct. Those probabilities are
ranked by ROC-AUC on the probe's own held-out problems.

Design notes that matter for the number:
  * The judge is UNTRAINED on our data, so it never touches the training split
    and cannot leak. It is scored on the probe's test problems only — which is
    also what keeps the API bill proportional to n_test rather than n_total.
  * Every response is cached on disk under sha256(prefix + prompt + model), so
    a rerun, a crash, or a change to the scoring code costs nothing.
  * No API key ⇒ this stage prints why and exits 0. analysis.py then records
    the judge as missing and states plainly that S_text is a max over fewer
    readers, which can only make Δ look bigger.

Run:  python -m experiment.llm_judge          # after train_probe
      python -m experiment.llm_judge --all    # judge train rows too (costs more)
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from experiment import analysis, forced_answer
    from experiment.config import CONFIG, lineage
except ImportError:  # run as a plain script from inside experiment/
    import analysis  # type: ignore
    import forced_answer  # type: ignore
    from config import CONFIG, lineage  # type: ignore


BASELINE_NAME = "llm_judge"

# --- tunables (module-level: config.py belongs to another stage) ------------

# Provider is chosen from the environment; the model id is pinned here so the
# number in the write-up is traceable to a specific judge.
ANTHROPIC_MODEL = "claude-opus-5"
OPENROUTER_MODEL = "anthropic/claude-opus-5"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

JUDGE_EFFORT = "low"          # Anthropic `output_config.effort`; the judge reads a
                              # long trace and emits one number — depth buys little
JUDGE_MAX_TOKENS = 2048
JUDGE_TIMEOUT_S = 180
JUDGE_CONCURRENCY = 4         # polite parallelism; the cache makes reruns free
JUDGE_MAX_RETRIES = 5
JUDGE_BACKOFF_BASE_S = 2.0    # exponential: 2, 4, 8, 16, 32 (+ jitter)
JUDGE_CACHE_DIR = os.path.join(CONFIG.output_dir, "judge_cache")

# Safety valve only. Qwen3's 16k thinking budget is ~64k characters, well inside
# a 1M-token context, so this should never fire; if it does, the middle is cut
# and the row is flagged so a truncated judge input never passes silently.
JUDGE_MAX_PREFIX_CHARS = 400_000

# Above this many UNCACHED calls, refuse to spend money without an explicit
# --yes (or JUDGE_CONFIRM=1). Cheap insurance on a project whose whole GPU bill
# is ~$2.
JUDGE_CONFIRM_THRESHOLD = 200

JUDGE_SYSTEM = (
    "You are a careful forecaster. You will be shown a reasoning model's "
    "partial chain of thought on a multiple-choice or short-answer question. "
    "The trace is CUT OFF partway through; you never see the final answer. "
    "Your job is to predict whether that model's eventual final answer will be "
    "correct. Be calibrated: use the full 0-1 range, and do not default to 0.5. "
    "Reason briefly, then end your reply with a single final line of exactly "
    "the form:\nPROBABILITY: <number between 0 and 1>"
)

JUDGE_USER_TEMPLATE = (
    "Below is the prompt given to a reasoning model, followed by the first part "
    "of its thinking. Judge how likely it is that its final answer will be "
    "correct.\n\n"
    "=== BEGIN PARTIAL TRACE ===\n{prefix}\n=== END PARTIAL TRACE ===\n\n"
    "Remember: end with a line `PROBABILITY: <0-1>`."
)

# Everything that goes into the cache key alongside the prefix and model, so a
# prompt edit invalidates the cache instead of silently mixing two judges.
PROMPT_FINGERPRINT = JUDGE_SYSTEM + "\n---\n" + JUDGE_USER_TEMPLATE


# ---------------------------------------------------------------------------
# Pure helpers — parsing, cache keys, prompt assembly (unit-tested offline)
# ---------------------------------------------------------------------------

_PROB_LINE = re.compile(r"PROBABILITY\s*[:=]\s*(-?[0-9]*\.?[0-9]+)\s*(%?)", re.IGNORECASE)
_ANY_NUMBER = re.compile(r"(-?[0-9]*\.?[0-9]+)\s*(%?)")


def parse_probability(text: Optional[str]) -> Optional[float]:
    """Extract the judge's probability, or None if the reply is unusable.

    Contract: a value must land in [0, 1] after percent conversion. Anything
    else — no number, a bare "73" with no percent sign, 1.5, -0.2, a refusal —
    returns None and the row is DROPPED rather than imputed. Imputing 0.5 for
    unparseable replies would quietly drag the judge's AUC toward chance and
    inflate Δ, which is the exact failure this whole file exists to prevent.
    """
    if not text:
        return None
    matches = _PROB_LINE.findall(text)
    if not matches:
        matches = _ANY_NUMBER.findall(text)
        matches = matches[-1:] if matches else []
    else:
        matches = matches[-1:]
    if not matches:
        return None
    raw, pct = matches[0]
    try:
        val = float(raw)
    except ValueError:
        return None
    if pct == "%":
        val /= 100.0
    if not (0.0 <= val <= 1.0):
        return None
    return val


def clip_prefix(prefix_text: str, max_chars: int = JUDGE_MAX_PREFIX_CHARS) -> Tuple[str, bool]:
    """(text, was_clipped). Keeps the head and tail — the question is at the
    front and the most recent reasoning at the back; the middle is the least
    load-bearing part to lose."""
    if len(prefix_text) <= max_chars:
        return prefix_text, False
    head = max_chars // 2
    tail = max_chars - head
    return (
        prefix_text[:head]
        + "\n\n[... middle of the trace elided to fit the judge's context ...]\n\n"
        + prefix_text[-tail:],
        True,
    )


def cache_key(prefix_text: str, model: str, prompt_fingerprint: str = PROMPT_FINGERPRINT) -> str:
    """sha256 over prefix + prompt + model — the three things that change the
    answer. Nothing else (not the row id, not k) belongs in the key: two
    identical prefixes should hit the same cache entry."""
    h = hashlib.sha256()
    for part in (prefix_text, prompt_fingerprint, model):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def cache_path(key: str, cache_dir: Optional[str] = None) -> str:
    # `cache_dir=None` resolves the module global at CALL time, not at def
    # time, so the directory stays overridable (tests, a scratch rerun).
    return os.path.join(JUDGE_CACHE_DIR if cache_dir is None else cache_dir, f"{key}.json")


def cache_read(key: str, cache_dir: Optional[str] = None) -> Optional[dict]:
    p = cache_path(key, cache_dir)
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def cache_write(key: str, payload: dict, cache_dir: Optional[str] = None) -> None:
    """Atomic write. The temp name carries a per-writer suffix because two
    worker threads can be judging two rows whose prefixes are byte-identical
    (and therefore share a cache key); a shared temp name makes the second
    `os.replace` fail on a file the first one already moved."""
    final = cache_path(key, cache_dir)
    os.makedirs(os.path.dirname(final), exist_ok=True)
    tmp = f"{final}.{os.getpid()}.{threading.get_ident()}.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, final)


def build_user_message(prefix_text: str) -> str:
    return JUDGE_USER_TEMPLATE.format(prefix=prefix_text)


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

def detect_provider(env: Optional[Dict[str, str]] = None) -> Tuple[Optional[str], Optional[str]]:
    """(provider, model) from the environment, or (None, None) if no key.

    JUDGE_PROVIDER pins the choice when both keys are present.
    """
    env = os.environ if env is None else env
    forced = (env.get("JUDGE_PROVIDER") or "").strip().lower()
    has_or = bool(env.get("OPENROUTER_API_KEY"))
    has_an = bool(env.get("ANTHROPIC_API_KEY"))
    if forced == "openrouter" and has_or:
        return "openrouter", env.get("JUDGE_MODEL") or OPENROUTER_MODEL
    if forced == "anthropic" and has_an:
        return "anthropic", env.get("JUDGE_MODEL") or ANTHROPIC_MODEL
    if forced:
        return None, None
    if has_or:
        return "openrouter", env.get("JUDGE_MODEL") or OPENROUTER_MODEL
    if has_an:
        return "anthropic", env.get("JUDGE_MODEL") or ANTHROPIC_MODEL
    return None, None


def _call_anthropic(user_message: str, model: str) -> str:
    """Official Anthropic SDK. Imported lazily so the module stays importable
    (and testable) without the package installed."""
    try:
        import anthropic
    except ImportError:
        raise SystemExit(
            "HALT: ANTHROPIC_API_KEY is set but the `anthropic` package is not "
            "installed. `pip install anthropic`, or set OPENROUTER_API_KEY to use "
            "the stdlib HTTP path instead."
        )

    client = anthropic.Anthropic(timeout=JUDGE_TIMEOUT_S)
    # Thinking is adaptive by default on this model family; effort is the lever
    # that keeps a 1,500-call judging run affordable. Server-side refusal
    # fallbacks are deliberately NOT enabled: a refusal here simply yields an
    # unparseable reply, which this pipeline already drops explicitly, and the
    # extra beta header is one more way for the stage to fail on a fresh box.
    kwargs = dict(
        model=model,
        max_tokens=JUDGE_MAX_TOKENS,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": user_message}],
    )
    try:
        resp = client.messages.create(output_config={"effort": JUDGE_EFFORT}, **kwargs)
    except TypeError:
        # An older installed SDK does not know `output_config`. Losing the
        # effort dial costs money, not correctness — far better than retrying
        # a guaranteed failure five times and then producing no baseline.
        print("WARNING: installed `anthropic` SDK rejects output_config; "
              "running the judge without an effort setting.", file=sys.stderr)
        resp = client.messages.create(**kwargs)
    if getattr(resp, "stop_reason", None) == "refusal":
        return ""
    return "".join(
        b.text for b in resp.content if getattr(b, "type", None) == "text"
    )


def _call_openrouter(user_message: str, model: str) -> str:
    """OpenRouter is a separate service with its own (OpenAI-shaped) HTTP API
    and no Anthropic SDK, so this path is stdlib HTTP by necessity."""
    import urllib.error
    import urllib.request

    body = json.dumps(
        {
            "model": model,
            "max_tokens": JUDGE_MAX_TOKENS,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": user_message},
            ],
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        OPENROUTER_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=JUDGE_TIMEOUT_S) as r:
        payload = json.loads(r.read().decode("utf-8"))
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError(f"OpenRouter returned no choices: {payload}")
    return choices[0]["message"]["content"] or ""


def call_with_retries(
    user_message: str,
    provider: str,
    model: str,
    max_retries: int = JUDGE_MAX_RETRIES,
    sleep=time.sleep,
) -> Optional[str]:
    """One judge call with exponential backoff + jitter. Returns None if every
    attempt fails — the row is then dropped, never imputed."""
    fn = _call_anthropic if provider == "anthropic" else _call_openrouter
    for attempt in range(max_retries):
        try:
            return fn(user_message, model)
        except SystemExit:
            raise
        except Exception as exc:  # network, 429, 5xx, malformed body
            if attempt == max_retries - 1:
                print(f"WARNING: judge call failed after {max_retries} attempts: {exc}",
                      file=sys.stderr)
                return None
            delay = JUDGE_BACKOFF_BASE_S * (2**attempt) * (1.0 + random.random() * 0.25)
            print(f"WARNING: judge call failed ({exc}); retrying in {delay:.1f}s",
                  file=sys.stderr)
            sleep(delay)
    return None


def judge_one(
    prefix_text: str,
    provider: str,
    model: str,
    cache_dir: Optional[str] = None,
    call=None,
) -> dict:
    """{"probability", "raw", "cached"} for one prefix, cache-first."""
    call = call_with_retries if call is None else call
    text, clipped = clip_prefix(prefix_text)
    key = cache_key(text, model)
    hit = cache_read(key, cache_dir)
    if hit is not None:
        return {"probability": hit.get("probability"), "raw": hit.get("raw"),
                "cached": True, "clipped": clipped}
    raw = call(build_user_message(text), provider, model)
    prob = parse_probability(raw)
    record = {"probability": prob, "raw": raw, "model": model, "provider": provider}
    cache_write(key, record, cache_dir)
    return {"probability": prob, "raw": raw, "cached": False, "clipped": clipped}


def judge_batch(
    prefix_texts: Sequence[str],
    provider: str,
    model: str,
    cache_dir: Optional[str] = None,
    concurrency: int = JUDGE_CONCURRENCY,
    call=None,
) -> List[dict]:
    if concurrency <= 1 or len(prefix_texts) <= 1:
        return [judge_one(t, provider, model, cache_dir, call) for t in prefix_texts]
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(lambda t: judge_one(t, provider, model, cache_dir, call), prefix_texts))


# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    judge_all_rows = "--all" in argv
    assume_yes = "--yes" in argv or os.environ.get("JUDGE_CONFIRM") == "1"

    provider, model = detect_provider()
    if provider is None:
        print(
            "llm_judge: SKIPPED — no API key found.\n"
            "  Set OPENROUTER_API_KEY (stdlib HTTP path) or ANTHROPIC_API_KEY "
            "(needs `pip install anthropic`) and rerun.\n"
            "  analysis.py will record this baseline as missing; S_text then takes "
            "its max over fewer readers, which can only make Δ LOOK BIGGER. Say so "
            "in the write-up if the judge never ran."
        )
        return

    try:
        with open(CONFIG.results_path) as f:
            results = json.load(f)
    except OSError:
        raise SystemExit(
            "HALT: results.json not found — run train_probe first; the judge must "
            "be scored on the probe's exact held-out problems."
        )

    try:
        from experiment.text_floor import split_indices_from_results
    except ImportError:
        from text_floor import split_indices_from_results  # type: ignore

    rows, default_k = forced_answer.load_included_prefix_rows()
    if not rows:
        raise SystemExit("HALT: no included rows in prefixes.jsonl.")

    per_k: Dict[str, dict] = {}
    t0 = time.time()
    for k, krows in sorted(forced_answer.group_rows_by_k(rows, default_k).items()):
        pids = [r["problem_id"] for r in krows]
        _train_idx, test_idx = split_indices_from_results(pids, results)
        target_idx = list(range(len(krows))) if judge_all_rows else list(test_idx)
        target_rows = [krows[i] for i in target_idx]
        texts = forced_answer.load_prefix_texts(target_rows, k)

        n_uncached = sum(
            1 for t in texts if cache_read(cache_key(clip_prefix(t)[0], model)) is None
        )
        if n_uncached > JUDGE_CONFIRM_THRESHOLD and not assume_yes:
            raise SystemExit(
                f"HALT: k={k}% needs {n_uncached} uncached judge calls to "
                f"{provider}:{model}. That is real money. Rerun with --yes (or "
                f"JUDGE_CONFIRM=1) to proceed, or lower the row count."
            )
        print(f"llm_judge[k={k}%]: {len(texts)} rows, {n_uncached} uncached "
              f"-> {provider}:{model}")

        judged = judge_batch(texts, provider, model)

        # Rows the judge could not produce a usable probability for are dropped,
        # not imputed (see parse_probability).
        keep = [(r, j) for r, j in zip(target_rows, judged) if j["probability"] is not None]
        n_dropped = len(judged) - len(keep)
        by_pid = {r["problem_id"]: j["probability"] for r, j in keep}

        test_rows = [krows[i] for i in test_idx if krows[i]["problem_id"] in by_pid]
        if not test_rows:
            print(f"WARNING: k={k}% — no scorable test rows; skipping.", file=sys.stderr)
            continue
        y = [bool(r["label"]) for r in test_rows]
        s = [by_pid[r["problem_id"]] for r in test_rows]
        point = analysis.score_at_k(
            y, s, row_keys=[forced_answer.row_key(r, k) for r in test_rows]
        )
        point["notes"] = {
            "provider": provider,
            "model": model,
            "effort": JUDGE_EFFORT if provider == "anthropic" else None,
            "n_unparseable_dropped": n_dropped,
            "n_clipped_prefixes": sum(1 for _r, j in keep if j["clipped"]),
            "n_cache_hits": sum(1 for _r, j in keep if j["cached"]),
            "judged_rows": "all" if judge_all_rows else "test_split_only",
        }
        per_k[str(k)] = point

    if not per_k:
        raise SystemExit("HALT: the judge produced no scorable point at any k.")

    path = analysis.write_baseline_json(
        BASELINE_NAME,
        per_k,
        notes={
            "description": "frontier LLM reads the decoded prefix, returns P(final answer correct)",
            "prompt_sha256": hashlib.sha256(PROMPT_FINGERPRINT.encode()).hexdigest()[:16],
            "trained_on_our_data": False,
            **lineage(CONFIG.prefixes_path),
        },
    )
    for k, p in sorted(per_k.items(), key=lambda kv: int(kv[0])):
        print(f"llm_judge[k={k}%]: AUC={p['auc']} CI95={p['auc_ci95']} "
              f"(n_test={p['n_test']}) -> {path}")
    print(f"llm_judge: done in {time.time() - t0:.1f}s; cache at {JUDGE_CACHE_DIR}")


if __name__ == "__main__":
    main()
