import json,re,sys,os
import numpy as np
sys.path.insert(0,'/Users/IsahanKhan/ais-research')
from experiment.grading import extract_boxed, grade
from sklearn.metrics import roc_auc_score
B='/Users/IsahanKhan/ais-research/experiment/outputs/block2'

traces={}
for l in open(f'{B}/gen/traces.jsonl'):
    r=json.loads(l)
    if r.get('record_type')=='meta': continue
    traces[r['problem_id']]=r

# final answer of the full trace
final={}
for pid,r in traces.items():
    final[pid]=extract_boxed(r['trace_text'])

res=json.load(open(f'{B}/k50/results.json'))
TEST=set(res['split']['test_problem_ids'])

for K in ['k10','k25','k50','k75','k90']:
    rows=[json.loads(l) for l in open(f'{B}/{K}/forced_answer.jsonl')]
    rows=[r for r in rows if r.get('record_type')!='meta']
    te=[r for r in rows if r['problem_id'] in TEST]
    y=np.array([bool(r['label']) for r in te])
    f=np.array([1.0 if r['forced_correct'] else 0.0 for r in te])
    auc=roc_auc_score(y,f)
    fa=[r['forced_answer'] for r in te]
    fin=[final[r['problem_id']] for r in te]
    agree=np.mean([a==b for a,b in zip(fa,fin)])
    gold=[traces[r['problem_id']]['gold_answer'] for r in te]
    # sanity: label == (final==gold)?
    lab_check=np.mean([ (a==b)==l for a,b,l in zip(fin,gold,y)])
    print(f'{K}: n={len(te)} AUC(forced_correct->label)={auc:.4f} '
          f'forced==final agreement={agree:.3f}  label==(final==gold) {lab_check:.3f}')
    # decomposition: if forced answer == final answer, then forced_correct == label EXACTLY
    same=np.array([a==b for a,b in zip(fa,fin)])
    ident=np.mean(f[same]==y[same].astype(float))
    print(f'      among rows where forced==final ({same.sum()}), forced_correct==label in {ident*100:.1f}%')
    # AUC using ONLY the gold-free signal "forced == final"? not deployable either. 
    # AUC restricted to disagreement rows
    if (~same).sum()>0 and len(set(y[~same].tolist()))>1:
        print(f'      AUC on the {int((~same).sum())} disagreement rows = {roc_auc_score(y[~same],f[~same]):.4f}')
