import json,warnings,sys,numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0,'/private/tmp/claude-501/-Users-IsahanKhan-ais-research/5f92af70-0f21-46f3-8230-6aeef45872b7/scratchpad')
sys.path.insert(0,'/Users/IsahanKhan/ais-research')
from fastauc import auc_fast
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
B='/Users/IsahanKhan/ais-research/experiment/outputs/block2'
KS=['k1','k10','k25','k50','k75','k90']
res=json.load(open(f'{B}/k50/results.json')); TE=set(res['split']['test_problem_ids']); TR=set(res['split']['train_problem_ids'])

FA={}
for K in KS[1:]:
    FA[K]={r['problem_id']:r for r in (json.loads(l) for l in open(f'{B}/{K}/forced_answer.jsonl')) if r.get('record_type')!='meta'}
pids=sorted(TE & set(FA['k10']))
y=np.array([bool(FA['k10'][p]['label']) for p in pids])

print('=== A. GOLD-FREE forced-answer variants (deployable: no ground truth at test time) ===')
print('   predictor "forced answer at k agrees with forced answer at the previous cut":')
prev=None
for K in KS[1:]:
    cur=np.array([FA[K][p]['forced_answer'] or '' for p in pids])
    if prev is not None:
        s=(cur==prev).astype(float)
        print(f'   {K}: AUC={auc_fast(y,s):.4f}  (agree rate {s.mean():.2f})')
    prev=cur
print('   predictor "forced answer identical across ALL cuts up to k" (stability):')
acc=None
for K in KS[1:]:
    cur=np.array([FA[K][p]['forced_answer'] or '' for p in pids])
    acc = np.ones(len(pids),bool) if acc is None else acc
    if K!='k10':
        acc = acc & (cur==first)
    else:
        first=cur
    print(f'   {K}: AUC={auc_fast(y,acc.astype(float)):.4f} (stable frac {acc.mean():.2f})')
print('   [for reference] gold-using forced_correct AUCs:',
      {K: round(float(auc_fast(y,np.array([1.0*FA[K][p]["forced_correct"] for p in pids]))),4) for K in KS[1:]})

print('\n=== B. Does the "widening gap" just track the answer-copy rate? ===')
import re
tr_all={}
for l in open(f'{B}/gen/traces.jsonl'):
    r=json.loads(l)
    if r.get('record_type')=='meta': continue
    tr_all[r['problem_id']]=r
from experiment.grading import extract_boxed
final={p:extract_boxed(r['trace_text']) for p,r in tr_all.items()}
for K in KS[1:]:
    cur=[FA[K][p]['forced_answer'] for p in pids]
    copy=np.mean([a==final[p] for a,p in zip(cur,pids)])
    fc=np.array([1.0*FA[K][p]['forced_correct'] for p in pids])
    # theoretical AUC if the ONLY error source is answer-copy failure
    print(f'   {K}: copy_rate(forced==final)={copy:.3f}  AUC={auc_fast(y,fc):.4f}   '
          f'(as k->100 copy_rate->1 and AUC->1 by construction)')

print('\n=== C. Is the probe reading nothing but trace length? ===')
def prefinfo(K):
    out={}
    for l in open(f'{B}/{K}/prefixes.jsonl'):
        r=json.loads(l)
        if r.get('record_type')=='meta' or not r.get('included'): continue
        out[r['problem_id']]=r
    return out
for K in KS:
    d=np.load(f'{B}/{K}/acts.npz',allow_pickle=False)
    pid=[str(p) for p in d['problem_ids']]; yy=d['labels'].astype(bool)
    r=json.load(open(f'{B}/{K}/results.json'))
    itr=np.array([i for i,p in enumerate(pid) if p in TR]); ite=np.array([i for i,p in enumerate(pid) if p in TE])
    L=int(r['best_layer'].split('_')[1]); X=d[f'acts_layer{L}']
    clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,C=1.0,random_state=0)).fit(X[itr],yy[itr])
    s=clf.predict_proba(X[ite])[:,1]
    P=prefinfo(K)
    ln=np.array([P[pid[i]]['n_kept_thinking_tokens'] for i in ite],float)
    # length-only logistic trained on train (honest)
    ltr=np.array([P[pid[i]]['n_kept_thinking_tokens'] for i in itr],float).reshape(-1,1)
    lc=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)).fit(np.log1p(ltr),yy[itr])
    ls=lc.predict_proba(np.log1p(ln.reshape(-1,1)))[:,1]
    # partial: AUC of probe within length-matched strata (quartiles of length)
    q=np.quantile(ln,[.25,.5,.75]); strat=np.digitize(ln,q)
    parts=[]
    for g in range(4):
        m=strat==g
        if m.sum()>4 and 0<yy[ite][m].sum()<m.sum(): parts.append((m.sum(),auc_fast(yy[ite][m],s[m])))
    wavg=sum(n*a for n,a in parts)/sum(n for n,_ in parts) if parts else float('nan')
    print(f'   {K}: probe={auc_fast(yy[ite],s):.4f}  length-only(train-fit)={auc_fast(yy[ite],ls):.4f}  '
          f'corr(probe_score, log len)={np.corrcoef(s,np.log1p(ln))[0,1]:+.3f}  '
          f'probe AUC within length quartiles (weighted)={wavg:.4f}')
