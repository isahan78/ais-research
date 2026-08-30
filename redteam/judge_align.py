import json, os, sys, glob
sys.path.insert(0,'/Users/IsahanKhan/ais-research')
from experiment.llm_judge import cache_key, clip_prefix, PROMPT_FINGERPRINT
from tokenizers import Tokenizer
snap=glob.glob('/Users/IsahanKhan/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/*/tokenizer.json')[0]
tok=Tokenizer.from_file(snap)
BASE='/Users/IsahanKhan/ais-research/experiment/outputs/block2'
MODEL='claude-opus-5'
for k in (1,25):
    d=f'{BASE}/k{k}'
    pre={}
    with open(f'{d}/prefixes.jsonl') as f:
        for line in f:
            r=json.loads(line)
            if r.get('record_type')=='meta' or not r['included']: continue
            pre[r['problem_id']]=r
    # prefix texts: reuse forced_answer cache if present else decode
    fa={}
    p=f'{d}/forced_answer.jsonl'
    if os.path.exists(p):
        for line in open(p):
            r=json.loads(line)
            if r.get('record_type')=='meta': continue
            fa[r['problem_id']]=r['prefix_text']
    pay=json.load(open(f'{d}/baseline_llm_judge.json'))['per_k'][str(k)]
    ok=0; bad=[]
    for rk,sc in zip(pay['test_row_keys'], pay['test_scores']):
        pid=rk.split('@')[0]
        txt = fa.get(pid) or tok.decode(pre[pid]['prefix_token_ids'], skip_special_tokens=True)
        key=cache_key(clip_prefix(txt)[0], MODEL)
        f2=f'{d}/judge_cache/{key}.json'
        if not os.path.exists(f2): bad.append((pid,'no cache file')); continue
        cached=json.load(open(f2))
        if abs((cached.get('probability') or -9) - sc) > 1e-9:
            bad.append((pid,'prob mismatch',cached.get('probability'),sc))
        else: ok+=1
    print(f'k={k}: verified {ok}/{len(pay["test_row_keys"])} judge scores trace back to the cache entry for that row\'s own prefix; problems: {bad[:5]} (n={len(bad)})')
    # also: does the cached raw text mention the right question?
