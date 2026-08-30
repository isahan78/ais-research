import json,warnings,sys,numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0,'/private/tmp/claude-501/-Users-IsahanKhan-ais-research/5f92af70-0f21-46f3-8230-6aeef45872b7/scratchpad')
from fastauc import auc_fast
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
B='/Users/IsahanKhan/ais-research/experiment/outputs/block2'
res=json.load(open(f'{B}/k50/results.json')); TE=set(res['split']['test_problem_ids']); TR=set(res['split']['train_problem_ids'])
print('HONEST (train-fitted, no sign choice on test) 1-2 feature length baselines vs the "tuned TF-IDF"')
print(f'{"k":>4} {"thinkchars":>11} {"promptchars":>12} {"both":>7} {"TFIDF(reported)":>16} {"crude_floor":>12}')
for K in ['k1','k10','k25','k50','k75','k90']:
    FA={r['problem_id']:r for r in (json.loads(l) for l in open(f'{B}/{K}/forced_answer.jsonl')) if r.get('record_type')!='meta'} if K!='k1' else None
    P={}
    for l in open(f'{B}/{K}/prefixes.jsonl'):
        r=json.loads(l)
        if r.get('record_type')=='meta' or not r.get('included'): continue
        P[r['problem_id']]=r
    pids=sorted(P)
    def feats(p):
        if FA:
            t=FA[p]['prefix_text']; n_all=len(t)
        else:
            n_all=P[p]['n_kept_thinking_tokens']*4
        npr=len(P[p]['prompt_token_ids'])
        return np.log1p(n_all), np.log1p(npr)
    X=np.array([feats(p) for p in pids]); y=np.array([bool(P[p]['label']) for p in pids])
    itr=[i for i,p in enumerate(pids) if p in TR]; ite=[i for i,p in enumerate(pids) if p in TE]
    outs=[]
    for cols in [[0],[1],[0,1]]:
        m=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)).fit(X[itr][:,cols],y[itr])
        outs.append(auc_fast(y[ite],m.predict_proba(X[ite][:,cols])[:,1]))
    tf=json.load(open(f'{B}/{K}/baseline_text_classifier.json'))['per_k'][K[1:]]['auc']
    cf=(json.load(open(f'{B}/{K}/results.json')).get('text_floor') or {}).get('auc')
    print(f'{K:>4} {outs[0]:>11.4f} {outs[1]:>12.4f} {outs[2]:>7.4f} {tf:>16.4f} {str(cf):>12}')
