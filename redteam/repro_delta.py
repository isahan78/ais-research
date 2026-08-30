import json, os, sys
import numpy as np
sys.path.insert(0,'/Users/IsahanKhan/ais-research')
BASE='/Users/IsahanKhan/ais-research/experiment/outputs/block2'
os.environ['EXPERIMENT_OUTPUT_DIR']=BASE+'/k25'
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

def probe_scores(k):
    res=json.load(open(f'{BASE}/k{k}/results.json'))
    z=np.load(f'{BASE}/k{k}/acts.npz'); pids=[str(p) for p in z['problem_ids']]; y=z['labels'].astype(bool)
    trp=set(res['split']['train_problem_ids']); tep=set(res['split']['test_problem_ids'])
    tr=[i for i,p in enumerate(pids) if p in trp]; te=[i for i,p in enumerate(pids) if p in tep]
    L=res['best_layer'].rsplit('_',1)[-1]
    X=z[f'acts_layer{L}']
    clf=make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000,C=1.0,random_state=0))
    clf.fit(X[tr],y[tr]); s=clf.predict_proba(X[te])[:,1]
    return {f'{pids[i]}@k{k}':(bool(y[i]),float(sc)) for i,sc in zip(te,s)}

def paired(y,ps,ts,n=1000,seed=0):
    y=np.asarray(y,bool); ps=np.asarray(ps,float); ts=np.asarray(ts,float)
    delta=roc_auc_score(y,ps)-roc_auc_score(y,ts)
    rng=np.random.default_rng(seed); out=[]
    while len(out)<n:
        idx=rng.integers(0,len(y),len(y))
        if len(set(y[idx].tolist()))<2: continue
        out.append(roc_auc_score(y[idx],ps[idx])-roc_auc_score(y[idx],ts[idx]))
    a=np.array(out)
    return delta, np.percentile(a,2.5), np.percentile(a,97.5), (a>0).mean()

for k,name in [(1,'text_classifier'),(25,'text_classifier'),(25,'forced_answer'),(1,'llm_judge'),(25,'llm_judge')]:
    P=probe_scores(k)
    pay=json.load(open(f'{BASE}/k{k}/baseline_{name}.json'))
    p=pay['per_k'][str(k)]
    tmap=dict(zip(p['test_row_keys'],p['test_scores']))
    keys=sorted(set(P)&set(tmap))
    y=[P[q][0] for q in keys]; ps=[P[q][1] for q in keys]; ts=[tmap[q] for q in keys]
    d,lo,hi,f=paired(y,ps,ts)
    print(f'k={k} probe vs {name}: n_shared={len(keys)} (probe {len(P)}, base {len(tmap)}) delta={d:+.4f} CI=[{lo:.4f},{hi:.4f}] frac>0={f:.3f}')
