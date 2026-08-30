import json, glob, os, sys, re
sys.path.insert(0,'/Users/IsahanKhan/ais-research')
from experiment.llm_judge import parse_probability
BASE='/Users/IsahanKhan/ais-research/experiment/outputs/block2'
for k in (1,25):
    files=glob.glob(f'{BASE}/k{k}/judge_cache/*.json')
    n=0; nprob=0; nnone=0; nofallback=0
    probs=[]
    for f in files:
        d=json.load(open(f))
        raw=d.get('raw'); p=d.get('probability')
        n+=1
        rp=parse_probability(raw)
        if rp!=p: print('  REPARSE MISMATCH', f, p, rp)
        if p is None:
            nnone+=1
            print(f'  UNPARSEABLE k={k}:', repr((raw or '')[-200:]))
        else:
            probs.append(p)
            if not re.search(r'PROBABILITY', raw or '', re.I): nofallback+=1
    print(f'k={k}: cache files={n} unparseable={nnone} used_fallback_number={nofallback} min={min(probs):.3f} max={max(probs):.3f}')
    # show one full example
    d=json.load(open(files[0]))
    print('   sample raw tail:', repr((d.get('raw') or '')[-300:]))
