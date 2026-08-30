import json, numpy as np, sys
sys.path.insert(0,'/Users/IsahanKhan/ais-research')
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import roc_auc_score
BASE='/Users/IsahanKhan/ais-research/experiment/outputs/block2'

def fit(Xtr,ytr,Xte,yte,scaler=None):
    clf=make_pipeline(scaler or StandardScaler(), LogisticRegression(max_iter=2000,C=1.0,random_state=0))
    clf.fit(Xtr,ytr); s=clf.predict_proba(Xte)[:,1]
    return roc_auc_score(yte,s), s

for k in [1,10,25,50,75,90]:
    res=json.load(open(f'{BASE}/k{k}/results.json'))
    z=np.load(f'{BASE}/k{k}/acts.npz'); pids=[str(p) for p in z['problem_ids']]; y=z['labels'].astype(bool)
    trp=set(res['split']['train_problem_ids']); tep=set(res['split']['test_problem_ids'])
    tr=np.array([i for i,p in enumerate(pids) if p in trp]); te=np.array([i for i,p in enumerate(pids) if p in tep])
    out={}
    for L in (9,18,27):
        X=z[f'acts_layer{L}']
        a,_=fit(X[tr],y[tr],X[te],y[te])
        out[f'layer_{L}']=round(a,4)
    stored={L:v['auc'] for L,v in res['per_layer'].items()}
    print(f'k={k}: recomputed={out} stored={stored} MATCH={out==stored}')
    if k==25:
        i=pids.index('3800')
        print('   pid 3800 in', 'TRAIN' if i in set(tr.tolist()) else 'TEST')
        keep=np.array([j for j in range(len(pids)) if j!=i])
        tr2=np.array([j for j in tr if j!=i]); te2=np.array([j for j in te if j!=i])
        for L in (9,18,27):
            X=z[f'acts_layer{L}']
            a,_=fit(X[tr2],y[tr2],X[te2],y[te2])
            ar,_=fit(X[tr],y[tr],X[te],y[te],RobustScaler())
            print(f'   layer{L}: drop3800 AUC={a:.4f}  RobustScaler AUC={ar:.4f}  (orig {out[f"layer_{L}"]})')
