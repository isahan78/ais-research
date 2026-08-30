"""End-to-end dry run of the three text baselines + analysis on synthetic
artifacts. No GPU, no network, no real API key. Scratch only."""
import json, os, random, sys, tempfile

sys.path.insert(0, "/Users/IsahanKhan/ais-research")

from experiment.config import CONFIG

TMP = tempfile.mkdtemp(prefix="baseline_e2e_")
for field, val in [
    ("output_dir", TMP),
    ("traces_path", os.path.join(TMP, "traces.jsonl")),
    ("prefixes_path", os.path.join(TMP, "prefixes.jsonl")),
    ("acts_path", os.path.join(TMP, "acts.npz")),
    ("results_path", os.path.join(TMP, "results.json")),
]:
    object.__setattr__(CONFIG, field, val)

from experiment import analysis, forced_answer, llm_judge, text_classifier

forced_answer.FORCED_ANSWER_PATH = os.path.join(TMP, "forced_answer.jsonl")
llm_judge.JUDGE_CACHE_DIR = os.path.join(TMP, "judge_cache")
analysis.ANALYSIS_PATH = os.path.join(TMP, "analysis.json")
analysis.FIG1_PATH = os.path.join(TMP, "fig1.png")
analysis.FIG2_PATH = os.path.join(TMP, "fig2.png")

rng = random.Random(0)
N = 40
K = 50
pids = [f"prob_{i:03d}" for i in range(N)]
labels = [rng.random() < 0.72 for _ in pids]

# --- synthetic traces + prefixes -------------------------------------------
with open(CONFIG.traces_path, "w") as f:
    f.write(json.dumps({"record_type": "meta", "stage": "generate_traces"}) + "\n")
    for pid, lab in zip(pids, labels):
        f.write(json.dumps({
            "problem_id": pid, "level": None, "subject": "math",
            "gold_answer": "C", "prompt_token_ids": [1] * 20,
            "trace_text": "", "correct": lab, "truncated_incomplete": False,
        }) + "\n")

with open(CONFIG.prefixes_path, "w") as f:
    f.write(json.dumps({"record_type": "meta", "stage": "truncate", "k_percent": K,
                        "n_included": N}) + "\n")
    for pid, lab in zip(pids, labels):
        f.write(json.dumps({
            "problem_id": pid, "included": True, "exclusion_reason": None,
            "label": lab, "prefix_token_ids": [rng.randint(5, 500) for _ in range(30)],
            "prompt_token_ids": [1] * 20, "n_thinking_tokens": 100,
            "n_kept_thinking_tokens": 50,
        }) + "\n")

# --- synthetic forced_answer.jsonl (what the GPU stage would have written) --
with open(forced_answer.FORCED_ANSWER_PATH, "w") as f:
    f.write(json.dumps({"record_type": "meta", "stage": "forced_answer"}) + "\n")
    for pid, lab in zip(pids, labels):
        fa = lab if rng.random() < 0.75 else (not lab)
        good = "the reasoning is clean and the answer is clearly option charlie "
        bad = "hmm wait i am confused let me recheck this once again please "
        f.write(json.dumps({
            "row_key": f"{pid}@k{K}", "problem_id": pid, "k_percent": K,
            "label": lab, "forced_correct": fa, "forced_answer": "C" if fa else "D",
            "generated_text": "C}", "prefix_text": (good if lab else bad) * 6,
        }) + "\n")

# --- synthetic results.json (what train_probe + text_floor would write) -----
test_pids = pids[::3]
train_pids = [p for p in pids if p not in set(test_pids)]
results = {
    "metric": "roc_auc", "truncation_k_percent": K,
    "n_train": len(train_pids), "n_test": len(test_pids),
    "split": {"train_problem_ids": sorted(train_pids), "test_problem_ids": sorted(test_pids)},
    "per_layer": {"layer_9": {"auc": 0.66, "auc_ci95": [0.5, 0.82]},
                  "layer_18": {"auc": 0.81, "auc_ci95": [0.66, 0.94]},
                  "layer_27": {"auc": 0.74, "auc_ci95": [0.58, 0.89]}},
    "best_layer": "layer_18",
    "text_floor": {"features": ["prefix_token_count", "prompt_token_count"],
                   "auc": 0.57, "auc_ci95": [0.4, 0.73], "n_test": len(test_pids)},
}
with open(CONFIG.results_path, "w") as f:
    json.dump(results, f, indent=2)

print("=" * 70, "\nSTAGE: forced_answer score\n", "=" * 70)
forced_answer.score()

print("=" * 70, "\nSTAGE: llm_judge (no key => graceful skip)\n", "=" * 70)
llm_judge.main([])

print("=" * 70, "\nSTAGE: llm_judge (dummy key + faked transport)\n", "=" * 70)
os.environ["OPENROUTER_API_KEY"] = "dummy-key-for-dry-run"
os.environ["JUDGE_MODEL"] = "fake/model"
def fake_transport(user_message, model):
    p = 0.8 if "clean" in user_message else 0.3
    return f"Reasoning...\nPROBABILITY: {p}"
llm_judge._call_openrouter = fake_transport
llm_judge.main([])

print("=" * 70, "\nSTAGE: llm_judge again (must be 100% cache hits)\n", "=" * 70)
def must_not_call(user_message, model):
    raise AssertionError("cache miss on rerun")
llm_judge._call_openrouter = must_not_call
llm_judge.main([])

print("=" * 70, "\nSTAGE: text_classifier\n", "=" * 70)
text_classifier.main()

print("=" * 70, "\nSTAGE: analysis\n", "=" * 70)
analysis.main()

print("\n--- analysis.json ---")
print(open(analysis.ANALYSIS_PATH).read()[:2500])
print("\n--- files ---")
for fn in sorted(os.listdir(TMP)):
    print(" ", fn, os.path.getsize(os.path.join(TMP, fn)))
print("TMP:", TMP)
