"""Final results table for the expanded dataset (Run 011). CPU-only.
One shared per-cut split; honest CV probe over all 35 layers; paired-bootstrap
Delta. Hard asserts: n_test is ROWS not tokens, split disjoint.
"""
import json
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

BASE = Path("/Users/IsahanKhan/ais-research/experiment/outputs/expansion")
LAYERS = list(range(35))
CGRID = [10.0**e for e in range(-5, 3)]
TAGS = ["k1","k10","k25","k50","k75","k90","abs64","abs128","abs256","abs512","abs1024"]
rng = np.random.default_rng(0)

def ci(y, s, n=1000):
    y=np.asarray(y); s=np.asarray(s); a=[]
    for _ in range(n):
        i=rng.integers(0,len(y),len(y))
        if len(set(y[i]))>1: a.append(roc_auc_score(y[i],s[i]))
    return [round(float(np.percentile(a,2.5)),3), round(float(np.percentile(a,97.5)),3)]

def paired(y, sp, st, n=1000):
    y=np.asarray(y); sp=np.asarray(sp); st=np.asarray(st); d=[]
    for _ in range(n):
        i=rng.integers(0,len(y),len(y))
        if len(set(y[i]))>1: d.append(roc_auc_score(y[i],sp[i])-roc_auc_score(y[i],st[i]))
    return round(float(np.mean(d)),4), [round(float(np.percentile(d,2.5)),3), round(float(np.percentile(d,97.5)),3)]

def load(tag):
    d=BASE/tag
    z=np.load(d/"acts.npz", allow_pickle=True)
    pid=np.array([str(x) for x in z["problem_ids"]]); y=np.array(z["labels"]).astype(bool)
    fc={}
    for l in open(d/"forced_confidence.jsonl"):
        r=json.loads(l)
        if r.get("record_type")!="meta": fc[str(r["problem_id"])]=r
    txt=np.array([fc.get(p,{}).get("prefix_text","") for p in pid])
    def _ptop(r):
        v=r.get("variants") or {}
        return v.get("p_top") if isinstance(v,dict) and v.get("p_top") is not None else np.nan
    ptop=np.array([_ptop(fc.get(p,{})) for p in pid])
    plen_by_pid={}
    for l in open(d/"prefixes.jsonl"):
        r=json.loads(l)
        if r.get("record_type")!="meta" and r.get("included"):
            plen_by_pid[str(r["problem_id"])]=np.log(len(r["prefix_token_ids"]))
    plen=np.array([plen_by_pid.get(p, np.nan) for p in pid])
    return z,pid,y,txt,ptop,plen

def honest_probe(z,y,tr,te,pid):
    best=None
    for L in LAYERS:
        X=z[f"acts_layer{L}"]; Xtr,ytr,gtr=X[tr],y[tr],pid[tr]; bc,bcv=None,-1
        for C in CGRID:
            sc=[]
            try:
                for a,b in StratifiedGroupKFold(5).split(Xtr,ytr,gtr):
                    m=make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=2000)).fit(Xtr[a],ytr[a])
                    sc.append(roc_auc_score(ytr[b],m.decision_function(Xtr[b])))
                if np.mean(sc)>bcv: bcv=float(np.mean(sc));bc=C
            except Exception: pass
        m=make_pipeline(StandardScaler(),LogisticRegression(C=bc,max_iter=2000)).fit(X[tr],y[tr])
        s=m.decision_function(X[te])
        if best is None or bcv>best[0]: best=(bcv,L,bc,s)
    return best[1],best[2],best[3]

out={}
for tag in TAGS:
    z,pid,y,txt,ptop,plen=load(tag)
    tr,te=next(GroupShuffleSplit(1,test_size=0.35,random_state=0).split(y,y,groups=pid))
    assert not(set(pid[tr])&set(pid[te])), f"{tag}: split leak"
    yte=y[te]
    n_test=len(te)
    assert n_test<2000, f"{tag}: n_test={n_test} looks like tokens"
    L,C,ps=honest_probe(z,y,tr,te,pid)
    tv=make_pipeline(TfidfVectorizer(ngram_range=(1,2),min_df=2,sublinear_tf=True),LogisticRegression(C=1.0,max_iter=2000)).fit(txt[tr],y[tr]); ts=tv.decision_function(txt[te])
    lv=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000)).fit(plen[tr].reshape(-1,1),y[tr]); lsc=lv.decision_function(plen[te].reshape(-1,1))
    cs=ptop[te]; cs=np.where(np.isnan(cs), np.nanmedian(ptop[tr]), cs)
    from sklearn.metrics import roc_auc_score as A
    readers={"tfidf":(ts,A(yte,ts)),"forced_conf":(cs,A(yte,cs)),"length_only":(lsc,A(yte,lsc))}
    bestname=max(readers,key=lambda k:readers[k][1]); bst=readers[bestname][0]
    dmean,dci=paired(yte,ps,bst)
    out[tag]=dict(n_test=int(n_test),n_neg=int((~yte).sum()),
        probe=round(A(yte,ps),4),probe_layer=int(L),probe_C=C,probe_ci=ci(yte,ps),
        tfidf=round(readers["tfidf"][1],4),tfidf_ci=ci(yte,ts),
        forced_conf=round(readers["forced_conf"][1],4),length_only=round(readers["length_only"][1],4),
        best_text=bestname,delta=dmean,delta_ci=dci,delta_excludes_0=bool(dci[0]>0 or dci[1]<0))
    print(f"{tag:8} n={out[tag]['n_test']} neg={out[tag]['n_neg']} probe={out[tag]['probe']}(L{L},C={C:g}) "
          f"tfidf={out[tag]['tfidf']} conf={out[tag]['forced_conf']} len={out[tag]['length_only']} "
          f"D={dmean:+.3f}{dci}{'*' if out[tag]['delta_excludes_0'] else ''}", flush=True)
json.dump(out, open(BASE/"final_table.json","w"), indent=1)
print("DONE")
