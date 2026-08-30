import json, os, glob
import numpy as np
from tokenizers import Tokenizer
snap=glob.glob('/Users/IsahanKhan/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/*/tokenizer.json')[0]
tok=Tokenizer.from_file(snap)
BASE='/Users/IsahanKhan/ais-research/experiment/outputs/block2'
for k in [10,25,50,75,90]:
    d=f'{BASE}/k{k}'
    fa=[]
    with open(f'{d}/forced_answer.jsonl') as f:
        for line in f:
            r=json.loads(line)
            if r.get('record_type')=='meta': meta=r; continue
            fa.append(r)
    pre={}
    with open(f'{d}/prefixes.jsonl') as f:
        for line in f:
            r=json.loads(line)
            if r.get('record_type')=='meta' or not r['included']: continue
            pre[r['problem_id']]=r
    ks=set(r['k_percent'] for r in fa)
    lab_ok=all(bool(r['label'])==bool(pre[r['problem_id']]['label']) for r in fa)
    # verify prefix_text matches decoding the prefix ids of THIS k
    bad=0; checked=0
    import random; random.seed(1)
    for r in random.sample(fa, 25):
        exp=tok.decode(pre[r['problem_id']]['prefix_token_ids'], skip_special_tokens=True)
        checked+=1
        if exp!=r['prefix_text']: bad+=1
    agree_all=np.mean([1.0 if (r['forced_correct'] is r['label']) else 0.0 for r in fa])
    print(f'k={k}: n_fa={len(fa)} k_set={ks} labels_ok={lab_ok} prefix_text_mismatch={bad}/{checked} agreement_all_rows={agree_all:.4f} meta_ungrade={meta["n_ungradeable"]}')
    # forced answer distribution
    fc=[r['forced_correct'] for r in fa]
    print('   forced_correct counts:', {v:fc.count(v) for v in set(fc)})
