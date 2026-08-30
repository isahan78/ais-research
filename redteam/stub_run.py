"""Run generate_traces.main() against stub vllm/transformers/datasets. No GPU."""
import sys, types, json, dataclasses, tempfile, os
sys.path.insert(0, "/Users/IsahanKhan/ais-research")

import experiment.config as cfgmod
import experiment.generate_traces as gt

THINK_START, THINK_END = 151667, 151668

# ---- stub datasets ----
class DS:
    def __init__(self, rows): self.rows = rows
    def filter(self, fn): return DS([r for r in self.rows if fn(r)])
    def shuffle(self, seed=0): return self
    def select(self, rng): return DS([self.rows[i] for i in rng])
    def __len__(self): return len(self.rows)
    def __iter__(self): return iter(self.rows)

def load_dataset(name, split=None):
    if "MMLU" in name:
        return DS([{ "question_id": i, "question": f"Q{i}?",
                     "options": ["a","b","c","d"], "answer": "C", "answer_index": 2,
                     "category": "health", "src": "ori"} for i in range(5)])
    return DS([{ "unique_id": f"m/{i}.json", "problem": f"P{i}", "answer": str(i),
                 "level": 5 if i % 2 else 3, "subject": "Algebra"} for i in range(10)])

# ---- stub transformers ----
class Tok:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False
        return "<|im_start|>user\n" + messages[0]["content"] + "<|im_end|>\n<|im_start|>assistant\n"
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [hash(text) % 1000 + i for i in range(7)]}
    def decode(self, ids, skip_special_tokens=True):
        return "boxed: \\boxed{C}" if THINK_END in ids or ids and ids[0] > 2999 else "thinking"
class AutoTokenizer:
    @staticmethod
    def from_pretrained(mid): return Tok()

# ---- stub vllm ----
class Out:
    def __init__(self, ids): self.outputs = [types.SimpleNamespace(token_ids=ids)]
class LLM:
    def __init__(self, **kw): self.kw = kw
    def generate(self, prompts, params):
        outs = []
        for i, p in enumerate(prompts):
            assert set(p) == {"prompt_token_ids"}, "must feed token ids, not text"
            ids = [THINK_START] + [2000+j for j in range(20)] + [THINK_END] + [3000, 3001]
            if i == 0:  # one truncated trace
                ids = [THINK_START] + [2000+j for j in range(20)]
            outs.append(Out(ids))
        return outs
class SamplingParams:
    def __init__(self, **kw): self.__dict__.update(kw)

for name, mod in [("datasets", {"load_dataset": load_dataset}),
                  ("transformers", {"AutoTokenizer": AutoTokenizer}),
                  ("vllm", {"LLM": LLM, "SamplingParams": SamplingParams})]:
    m = types.ModuleType(name); m.__dict__.update(mod); sys.modules[name] = m

def run(kind, dsid, n):
    tmp = tempfile.mkdtemp()
    cfg = dataclasses.replace(cfgmod.CONFIG, dataset_kind=kind, dataset_id=dsid,
                              n_problems=n, output_dir=tmp,
                              traces_path=os.path.join(tmp, "traces.jsonl"))
    gt.CONFIG = cfg
    cfgmod.CONFIG = cfg   # lineage() reads the module-level CONFIG
    gt.main()
    rows = [json.loads(l) for l in open(cfg.traces_path)]
    print(kind, "meta:", {k: rows[0][k] for k in ("dataset_kind","n_problems","n_truncated_incomplete","n_ungradeable","config_hash")})
    r = rows[1]
    print(" keys:", sorted(r))
    print(" pid=%r level=%r subject=%r gold=%r correct=%r trunc=%r meta=%s" % (
        r["problem_id"], r["level"], r["subject"], r["gold_answer"], r["correct"],
        r["truncated_incomplete"], r["meta"]))
    print(" rendered starts:", r["rendered_prompt"][:60].replace("\n","\\n"))
    # stage 2 must consume it
    from experiment.truncate import build_prefix
    ok = sum(1 for x in rows[1:] if build_prefix(x["problem_id"], x["prompt_token_ids"],
                                                 x["trace_token_ids"], x["correct"]).included)
    print(" truncate included:", ok, "/", len(rows)-1)

run("mmlu_pro", "TIGER-Lab/MMLU-Pro", 5)
print()
run("math500", "HuggingFaceH4/MATH-500", 5)
