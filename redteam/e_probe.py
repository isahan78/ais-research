import json,warnings,numpy as np
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.svm import LinearSVC
B='/Users/IsahanKhan/ais-research/experiment/outputs/block2'
KS=['k1','k10','k25','k50','k75','k90']
CS=[1e-5,1e-4,1e-3,1e-2,1e-1,1.0,10.0,100.0]

def load(K):
    d=np.load(f'{B}/{K}/acts.npz',allow_pickle=False)
    pids=[str(p) for p in d['problem_ids']]; y=d['labels'].astype(bool)
    r=json.load(open(f'{B}/{K}/results.json'))
    tr=set(r['split']['train_problem_ids']); te=set(r['split']['test_problem_ids'])
    itr=np.array([i for i,p in enumerate(pids) if p in tr]); ite=np.array([i for i,p in enumerate(pids) if p in te])
    X={L:d[f'acts_layer{L}'] for L in (9,18,27)}
    return X,y,itr,ite,r

def cvauc(X,y,C,seed=0,folds=5):
    skf=StratifiedKFold(n_splits=folds,shuffle=True,random_state=seed)
    s=np.zeros(len(y))
    for a,b in skf.split(X,y):
        clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,C=C))
        clf.fit(X[a],y[a]); s[b]=clf.predict_proba(X[b])[:,1]
    return roc_auc_score(y,s)

def fit_test(Xtr,ytr,Xte,yte,C):
    clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,C=C,random_state=0))
    clf.fit(Xtr,ytr); s=clf.predict_proba(Xte)[:,1]
    return roc_auc_score(yte,s), s

print(f'{"k":>4} {"reported":>9} {"testmaxL":>9} | {"trainCV-sel layer+C":>22} {"CVauc":>7} {"TEST":>7} | {"concat3 bestC":>14} {"TEST":>7} | {"oracleC bestlayer TEST":>22}')
out={}
for K in KS:
    X,y,itr,ite,r=load(K)
    ytr,yte=y[itr],y[ite]
    rep=r['per_layer'][r['best_layer']]['auc']
    # honest: pick (layer,C) by CV inside train
    best=(-1,None)
    grid={}
    for L in (9,18,27):
        for C in CS:
            a=cvauc(X[L][itr],ytr,C)
            grid[(L,C)]=a
            if a>best[0]: best=(a,(L,C))
    L,C=best[1]
    honest,_=fit_test(X[L][itr],ytr,X[L][ite],yte,C)
    # concat all 3 layers, C picked by CV in train
    Xc=np.concatenate([X[L] for L in (9,18,27)],axis=1)
    bc=max(CS,key=lambda C: cvauc(Xc[itr],ytr,C))
    conc,_=fit_test(Xc[itr],ytr,Xc[ite],yte,bc)
    # full oracle: best (layer,C) on TEST (upper bound on any tuning)
    orc=max(max(fit_test(X[L][itr],ytr,X[L][ite],yte,C)[0] for C in CS) for L in (9,18,27))
    orc=max(orc, max(fit_test(Xc[itr],ytr,Xc[ite],yte,C)[0] for C in CS))
    out[K]=dict(reported=rep,honest=honest,concat=conc,oracle=orc,sel=(L,C),cv=best[0],concC=bc)
    print(f'{K:>4} {rep:>9.4f} {rep:>9.4f} | {"L%d C=%g"%(L,C):>22} {best[0]:>7.4f} {honest:>7.4f} | {"C=%g"%bc:>14} {conc:>7.4f} | {orc:>22.4f}')
json.dump({k:{a:(b if not isinstance(b,tuple) else list(b)) for a,b in v.items()} for k,v in out.items()},open(f'{__import__("os").path.dirname(__file__)}/probe_tuning.json','w'),indent=1)
