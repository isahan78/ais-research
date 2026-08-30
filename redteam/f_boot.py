import json,warnings,sys,numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0,'/private/tmp/claude-501/-Users-IsahanKhan-ais-research/5f92af70-0f21-46f3-8230-6aeef45872b7/scratchpad')
from fastauc import auc_fast
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
B='/Users/IsahanKhan/ais-research/experiment/outputs/block2'
KS=['k10','k25','k50','k75','k90']
def probe_scores(K,C=1.0,layer=None):
    d=np.load(f'{B}/{K}/acts.npz',allow_pickle=False)
    pids=[str(p) for p in d['problem_ids']]; y=d['labels'].astype(bool)
    r=json.load(open(f'{B}/{K}/results.json'))
    tr=set(r['split']['train_problem_ids']); te=set(r['split']['test_problem_ids'])
    itr=np.array([i for i,p in enumerate(pids) if p in tr]); ite=np.array([i for i,p in enumerate(pids) if p in te])
    L=layer or int(r['best_layer'].split('_')[1])
    X=d[f'acts_layer{L}']
    clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,C=C,random_state=0))
    clf.fit(X[itr],y[itr]); s=clf.predict_proba(X[ite])[:,1]
    return [pids[i] for i in ite],y[ite],s
def baseline(K,name):
    p=json.load(open(f'{B}/{K}/baseline_{name}.json'))['per_k'][K[1:]]
    return [k.split('@')[0] for k in p['test_row_keys']], np.array(p['test_labels']), np.array(p['test_scores']), p['auc']
def align(pk,py,ps,tk,ty,ts):
    o=np.argsort(pk); pk=[pk[i] for i in o]; py=py[o]; ps=ps[o]
    o=np.argsort(tk); tk=[tk[i] for i in o]; ty=ty[o]; ts=ts[o]
    assert pk==tk and (py==ty).all()
    return py,ps,ts
def paired_boot(y,a,b,n=20000,seed=0):
    rng=np.random.default_rng(seed); N=len(y); out=np.empty(n); c=0
    while c<n:
        idx=rng.integers(0,N,N); yy=y[idx]
        if yy.all() or not yy.any(): continue
        out[c]=auc_fast(yy,a[idx])-auc_fast(yy,b[idx]); c+=1
    return auc_fast(y,a)-auc_fast(y,b), np.percentile(out,2.5),np.percentile(out,97.5),(out>0).mean()

print('=== 5. PAIRED BOOTSTRAP  Delta = probe - forced_answer (20k resamples, n_test=102, 25 neg) ===')
for K in KS:
    py,ps,fs=align(*probe_scores(K),*baseline(K,'forced_answer')[:3])
    d,lo,hi,f=paired_boot(py,ps,fs)
    print(f'{K}: probe={auc_fast(py,ps):.4f} forced={auc_fast(py,fs):.4f} delta={d:+.4f} CI95=[{lo:+.4f},{hi:+.4f}] P(delta>0)={f:.3%}')
print('\n=== 5b. fairly-tuned probe vs forced_answer ===')
tuned={'k10':(18,1e-3),'k25':(18,1e-4),'k50':(27,1e-3),'k75':(27,1e-3),'k90':(27,1.0)}
for K in KS:
    L,C=tuned[K]
    py,ps,fs=align(*probe_scores(K,C=C,layer=L),*baseline(K,'forced_answer')[:3])
    d,lo,hi,f=paired_boot(py,ps,fs)
    print(f'{K}: probe_tuned={auc_fast(py,ps):.4f} forced={auc_fast(py,fs):.4f} delta={d:+.4f} CI95=[{lo:+.4f},{hi:+.4f}] P(delta>0)={f:.3%}')
print('\n=== 5c. Delta vs TF-IDF only (drop the gold-using forced-answer) ===')
for K in ['k1']+KS:
    py,ps,ts=align(*probe_scores(K),*baseline(K,'text_classifier')[:3])
    d,lo,hi,f=paired_boot(py,ps,ts)
    print(f'{K}: probe={auc_fast(py,ps):.4f} tfidf={auc_fast(py,ts):.4f} delta={d:+.4f} CI95=[{lo:+.4f},{hi:+.4f}] P(delta>0)={f:.3%}')
print('\n=== 5d. fairly-tuned probe vs TF-IDF ===')
tuned2={'k1':(18,1.0),'k10':(18,1e-3),'k25':(18,1e-4),'k50':(27,1e-3),'k75':(27,1e-3),'k90':(27,1.0)}
for K in ['k1']+KS:
    L,C=tuned2[K]
    py,ps,ts=align(*probe_scores(K,C=C,layer=L),*baseline(K,'text_classifier')[:3])
    d,lo,hi,f=paired_boot(py,ps,ts)
    print(f'{K}: probe_tuned={auc_fast(py,ps):.4f} tfidf={auc_fast(py,ts):.4f} delta={d:+.4f} CI95=[{lo:+.4f},{hi:+.4f}] P(delta>0)={f:.3%}')
