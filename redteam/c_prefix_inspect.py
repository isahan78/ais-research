import json,re,sys,random
import numpy as np
sys.path.insert(0,'/Users/IsahanKhan/ais-research')
from experiment.grading import extract_boxed
from sklearn.metrics import roc_auc_score
B='/Users/IsahanKhan/ais-research/experiment/outputs/block2'
traces={}
for l in open(f'{B}/gen/traces.jsonl'):
    r=json.loads(l)
    if r.get('record_type')=='meta': continue
    traces[r['problem_id']]=r
final={p:extract_boxed(r['trace_text']) for p,r in traces.items()}

# patterns for a stated/implied answer inside the THINKING part of the prefix
PATS=[r'answer is\s*\**\s*\(?([A-J])\b', r'answer:\s*\(?([A-J])\b', r'\\boxed\{\s*([A-J])\s*\}',
      r"it'?s\s*\(?([A-J])\)", r'option\s*\(?([A-J])\)?\s*(?:is|seems|looks|must|would)',
      r'go with\s*\(?([A-J])\b', r'choose\s*\(?([A-J])\b', r'pick\s*\(?([A-J])\b',
      r'so,?\s*\(?([A-J])\)?\.', r'therefore,?\s*\(?([A-J])\b']
def thinking_part(pt):
    # prefix_text = rendered prompt (user turn) + thinking. Split at the options block end.
    i=pt.find('\nassistant')
    if i>=0: return pt[i:]
    # fallback: find last occurrence of the option list terminator
    m=list(re.finditer(r'\nJ\..*?\n', pt))
    return pt[m[-1].end():] if m else pt

def stated(pt):
    t=thinking_part(pt); hits=[]
    for p in PATS:
        for m in re.finditer(p,t,re.I): hits.append(m.group(1).upper())
    return hits

res=json.load(open(f'{B}/k50/results.json')); TEST=set(res['split']['test_problem_ids'])
for K in ['k25','k75']:
    rows=[json.loads(l) for l in open(f'{B}/{K}/forced_answer.jsonl')]
    rows=[r for r in rows if r.get('record_type')!='meta']
    te=[r for r in rows if r['problem_id'] in TEST]
    n_stated=0; n_match=0; recs=[]
    for r in te:
        h=stated(r['prefix_text'])
        last=h[-1] if h else None
        n_stated+= bool(h)
        if last and last==r['forced_answer']: n_match+=1
        recs.append((r,last,h))
    y=np.array([bool(r['label']) for r in te]); f=np.array([1.0*r['forced_correct'] for r in te])
    has=np.array([bool(h) for _,_,h in recs])
    print(f'=== {K}: n_test={len(te)}  prefix contains an explicit stated-answer pattern: {n_stated} ({n_stated/len(te):.0%})')
    print(f'    last stated letter == forced answer: {n_match}/{n_stated if n_stated else 1} ({n_match/max(n_stated,1):.0%})')
    for grp,mask in [('STATED',has),('NOT STATED',~has)]:
        if mask.sum()>2 and len(set(y[mask].tolist()))>1:
            print(f'    forced-answer AUC on {grp} subset (n={mask.sum()}, pos={y[mask].sum()}): {roc_auc_score(y[mask],f[mask]):.4f}  acc={np.mean(f[mask]==y[mask]):.3f}')
    # letter presence is trivially 100% (options block lists A-J), report on thinking only
random.seed(7)
print('\n\n############ HAND INSPECTION: 15 random test rows, k=25 and k=75 ############')
rows25={r['problem_id']:r for r in (json.loads(l) for l in open(f'{B}/k25/forced_answer.jsonl')) if r.get('record_type')!='meta'}
rows75={r['problem_id']:r for r in (json.loads(l) for l in open(f'{B}/k75/forced_answer.jsonl')) if r.get('record_type')!='meta'}
pids=sorted(set(rows25)&set(rows75)&TEST); sample=random.sample(pids,15)
for pid in sample:
    for K,RR in [('k25',rows25),('k75',rows75)]:
        r=RR[pid]; t=thinking_part(r['prefix_text'])
        print(f'\n--- pid {pid} [{K}] gold={traces[pid]["gold_answer"]} final={final[pid]} forced={r["forced_answer"]} forced_correct={r["forced_correct"]} label={r["label"]}')
        print('    stated-pattern hits in thinking:',stated(r['prefix_text'])[-6:])
        print('    ...TAIL OF PREFIX:', repr(t[-420:]))
