import json, os, random, glob
from tokenizers import Tokenizer
snap=glob.glob('/Users/IsahanKhan/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/*/tokenizer.json')[0]
tok=Tokenizer.from_file(snap)
BASE='/Users/IsahanKhan/ais-research/experiment/outputs/block2'
print('id 151667 ->', repr(tok.decode([151667], skip_special_tokens=False)))
print('id 151668 ->', repr(tok.decode([151668], skip_special_tokens=False)))
for k in (25,75):
    rows=[]
    with open(f'{BASE}/k{k}/prefixes.jsonl') as f:
        for line in f:
            r=json.loads(line)
            if r.get('record_type')=='meta' or not r['included']: continue
            rows.append(r)
    random.seed(7)
    for r in random.sample(rows,10):
        p=r['prefix_token_ids']; np_=len(r['prompt_token_ids'])
        txt=tok.decode(p[np_:], skip_special_tokens=False)
        print('='*80)
        print(f"k={k} pid={r['problem_id']} label={r['label']} n_think={r['n_thinking_tokens']} keep={r['n_kept_thinking_tokens']} lenprefix={len(p)}")
        print("  HEAD:", repr(txt[:120]))
        print("  TAIL:", repr(txt[-220:]))
        print("  has </think>:", '</think>' in txt, "| has \\boxed:", '\\boxed' in txt)
