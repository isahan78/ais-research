import json, numpy as np, sys
sys.path.insert(0,'/Users/IsahanKhan/ais-research')
from experiment.train_probe import shuffled_label_floor_max
BASE='/Users/IsahanKhan/ais-research/experiment/outputs/block2'
for k in (1,10):
    res=json.load(open(f'{BASE}/k{k}/results.json'))
    z=np.load(f'{BASE}/k{k}/acts.npz'); pids=[str(p) for p in z['problem_ids']]; y=z['labels'].astype(bool)
    trp=set(res['split']['train_problem_ids']); tep=set(res['split']['test_problem_ids'])
    tr=np.array([i for i,p in enumerate(pids) if p in trp]); te=np.array([i for i,p in enumerate(pids) if p in tep])
    Xs_tr=[z[f'acts_layer{L}'][tr] for L in (9,18,27)]
    Xs_te=[z[f'acts_layer{L}'][te] for L in (9,18,27)]
    f=shuffled_label_floor_max(Xs_tr,y[tr],Xs_te,y[te],500,0)
    a=np.array(f)
    print(f'k={k}: repro floor n={len(f)} mean={a.mean():.4f} p95={np.percentile(a,95):.4f} '
          f'| stored mean={res["shuffled_floor"]["mean"]} p95={res["shuffled_floor"]["p95"]} n={res["shuffled_floor"]["n_seeds"]}')
