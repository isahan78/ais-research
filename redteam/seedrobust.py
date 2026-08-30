import json, numpy as np, sys
sys.path.insert(0,'/Users/IsahanKhan/ais-research')
from experiment.train_probe import group_split, fit_and_auc
from sklearn.metrics import roc_auc_score
BASE='/Users/IsahanKhan/ais-research/experiment/outputs/block2'
for k in (10,50,90):
    z=np.load(f'{BASE}/k{k}/acts.npz'); pids=np.array([str(p) for p in z['problem_ids']]); y=z['labels'].astype(bool)
    fa={}
    for line in open(f'{BASE}/k{k}/forced_answer.jsonl'):
        r=json.loads(line)
        if r.get('record_type')=='meta': continue
        fa[r['problem_id']]=1.0 if r['forced_correct'] else 0.0
    res=[]
    for seed in range(10):
        tr,te=group_split(pids,y,0.35,seed*100,50)
        best=max(fit_and_auc(z[f'acts_layer{L}'][tr],y[tr],z[f'acts_layer{L}'][te],y[te],seed)[0] for L in (9,18,27))
        s=np.array([fa[p] for p in pids[te]])
        faauc=roc_auc_score(y[te],s)
        res.append((best,faauc,best-faauc))
    a=np.array(res)
    print(f'k={k}: probe best-layer AUC over 10 splits mean={a[:,0].mean():.3f} sd={a[:,0].std():.3f} range=[{a[:,0].min():.3f},{a[:,0].max():.3f}] '
          f'| forced mean={a[:,1].mean():.3f} sd={a[:,1].std():.3f} | delta mean={a[:,2].mean():+.3f} sd={a[:,2].std():.3f} n_pos_delta={(a[:,2]>0).sum()}/10')
