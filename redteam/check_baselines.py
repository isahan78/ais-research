import json, os, glob
import numpy as np
from sklearn.metrics import roc_auc_score
BASE='/Users/IsahanKhan/ais-research/experiment/outputs/block2'
traces={}
with open(os.path.join(BASE,'gen','traces.jsonl')) as f:
    for line in f:
        r=json.loads(line)
        if r.get('record_type')=='meta': continue
        traces[r['problem_id']]=r

for k in [1,10,25,50,75,90]:
    d=f'{BASE}/k{k}'
    res=json.load(open(f'{d}/results.json'))
    test_pids=set(res['split']['test_problem_ids']); train_pids=set(res['split']['train_problem_ids'])
    print(f'=== k={k}  n_train={res["n_train"]} n_test={res["n_test"]} best={res["best_layer"]} auc={res["per_layer"][res["best_layer"]]["auc"]} floor_p95={res["shuffled_floor"]["p95"]} overlap={len(test_pids&train_pids)}')
    print('    per_layer:', {L:v['auc'] for L,v in res['per_layer'].items()}, 'text_floor:', res.get('text_floor',{}).get('auc'))
    for f_ in sorted(glob.glob(f'{d}/baseline_*.json')):
        name=os.path.basename(f_)[len('baseline_'):-len('.json')]
        pay=json.load(open(f_))
        for kk,p in pay['per_k'].items():
            y=np.array(p['test_labels'],dtype=bool); s=np.array(p['test_scores'],dtype=float)
            rk=p['test_row_keys']
            rec=roc_auc_score(y,s)
            # label check vs traces
            pids=[x.split('@')[0] for x in rk]
            lab_ok=all(bool(traces[p_]['correct'])==bool(yy) for p_,yy in zip(pids,y))
            in_test=all(p_ in test_pids for p_ in pids)
            keysuffix=set(x.split('@')[1] for x in rk)
            print(f'    {name}[k={kk}]: stored_auc={p["auc"]} recomputed={rec:.4f} n={len(y)} npos={int(y.sum())} nneg={int((~y).sum())} '
                  f'mean_s|pos={s[y].mean():.4f} mean_s|neg={s[~y].mean():.4f} corr={np.corrcoef(s,y.astype(float))[0,1]:.4f} '
                  f'labels_match_traces={lab_ok} all_in_test_split={in_test} keysuffix={keysuffix} dup_pids={len(pids)!=len(set(pids))}')
            if 'notes' in p and name=='llm_judge': print('       notes:',p['notes'])
            if 'notes' in p and name=='forced_answer': print('       notes:',p['notes'])
