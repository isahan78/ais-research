import json, numpy as np, glob, sys
sys.path.insert(0,'/Users/IsahanKhan/ais-research')
from tokenizers import Tokenizer
snap=glob.glob('/Users/IsahanKhan/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/*/tokenizer.json')[0]
tok=Tokenizer.from_file(snap)
BASE='/Users/IsahanKhan/ais-research/experiment/outputs/block2'
for k in [1,10,25,50,75,90]:
    z=np.load(f'{BASE}/k{k}/acts.npz'); pids=[str(p) for p in z['problem_ids']]
    i=pids.index('3800')
    pre={}
    with open(f'{BASE}/k{k}/prefixes.jsonl') as f:
        for line in f:
            r=json.loads(line)
            if r.get('record_type')=='meta' or not r['included']: continue
            pre[r['problem_id']]=r
    ids=pre['3800']['prefix_token_ids']
    print(f"k={k} pid3800 last3 tokens={ids[-3:]} decoded={tok.decode(ids[-3:],skip_special_tokens=False)!r} norms:",
          {L: round(float(np.linalg.norm(z[f'acts_layer{L}'][i])),1) for L in (9,18,27)})
# how many rows exceed 3x median norm at each k/layer
for k in [1,10,25,50,75,90]:
    z=np.load(f'{BASE}/k{k}/acts.npz'); pids=[str(p) for p in z['problem_ids']]
    for L in (18,):
        A=z[f'acts_layer{L}']; n=np.linalg.norm(A,axis=1); med=np.median(n)
        out=[(pids[j], round(float(n[j]),1)) for j in np.where(n>5*med)[0]]
        print(f'k={k} L={L} rows>5x median:',out)
