import json,sys,re
import numpy as np
from sklearn.metrics import roc_auc_score
B='/Users/IsahanKhan/ais-research/experiment/outputs/block2'
traces={}
for l in open(f'{B}/gen/traces.jsonl'):
    r=json.loads(l)
    if r.get('record_type')=='meta': continue
    traces[r['problem_id']]=r
res=json.load(open(f'{B}/k50/results.json')); TEST=set(res['split']['test_problem_ids']); TR=set(res['split']['train_problem_ids'])

def load_pref(K):
    out={}
    for l in open(f'{B}/{K}/prefixes.jsonl'):
        r=json.loads(l)
        if r.get('record_type')=='meta' or not r.get('included'): continue
        out[r['problem_id']]=r
    return out

print('TRIVIAL SURFACE FEATURES -> AUC on the SAME 102-row test split (sign-optimised, i.e. best case)')
print(f'{"k":>4} {"n_think_full":>13} {"n_kept":>9} {"n_prompt":>9} {"prefix_chars":>13} {"n_options":>10} {"n_wait":>8} {"n_alt":>8}')
rows=[]
for K in ['k1','k10','k25','k50','k75','k90']:
    P=load_pref(K)
    pids=[p for p in sorted(P) if p in TEST]
    y=np.array([bool(P[p]['label']) for p in pids])
    feats={}
    feats['n_think_full']=np.array([P[p]['n_thinking_tokens'] for p in pids],float)
    feats['n_kept']=np.array([P[p]['n_kept_thinking_tokens'] for p in pids],float)
    feats['n_prompt']=np.array([len(P[p]['prompt_token_ids']) for p in pids],float)
    try:
        FA={r['problem_id']:r for r in (json.loads(l) for l in open(f'{B}/{K}/forced_answer.jsonl')) if r.get('record_type')!='meta'}
        feats['prefix_chars']=np.array([len(FA[p]['prefix_text']) for p in pids],float)
    except FileNotFoundError:
        feats['prefix_chars']=feats['n_kept']
    feats['n_options']=np.array([traces[p]['meta']['n_options'] for p in pids],float)
    txt={}
    try:
        txt={p:FA[p]['prefix_text'] for p in pids}
    except Exception: pass
    feats['n_wait']=np.array([txt.get(p,'').lower().count('wait') for p in pids],float)
    feats['n_alt']=np.array([txt.get(p,'').lower().count('alternatively') for p in pids],float)
    line=[K]
    best=(0,None)
    for name,v in feats.items():
        a=roc_auc_score(y,-v)   # longer/more hedging => more likely wrong
        a=max(a,1-a)
        line.append(f'{a:.3f}')
        if a>best[0]: best=(a,name)
    print(f'{line[0]:>4} '+' '.join(f'{x:>13}' if i==0 else f'{x:>9}' for i,x in enumerate(line[1:])))
    rows.append((K,best))
print()
for K,b in rows: print(f'  best single trivial feature at {K}: {b[1]} AUC={b[0]:.4f}')

# how much of TF-IDF's AUC is length? compare with reported
print()
print('reported text_classifier AUCs: ', {K: json.load(open(f'{B}/{K}/baseline_text_classifier.json'))['per_k'][K[1:]]['auc'] for K in ['k1','k10','k25','k50','k75','k90']})
print('reported crude floor AUCs:', {K: (json.load(open(f'{B}/{K}/results.json')).get('text_floor') or {}).get('auc') for K in ['k1','k10','k25','k50','k75','k90']})
# is prefix length at k proportional to full trace length?
P=load_pref('k50')
pids=[p for p in sorted(P) if p in TEST]
a=np.array([P[p]['n_kept_thinking_tokens'] for p in pids],float); b=np.array([P[p]['n_thinking_tokens'] for p in pids],float)
print('corr(n_kept@k50, n_thinking_full) =', np.corrcoef(a,b)[0,1])
