import json, os, numpy as np
B='/Users/IsahanKhan/ais-research/experiment/outputs/block2'
KS=['k1','k10','k25','k50','k75','k90']
allsplits={}
for k in KS:
    r=json.load(open(f'{B}/{k}/results.json'))
    tr=set(r['split']['train_problem_ids']); te=set(r['split']['test_problem_ids'])
    allsplits[k]=(tr,te)
    print(f'{k}: ntr_pid={len(tr)} nte_pid={len(te)} overlap={len(tr&te)} n_train_rows={r["n_train"]} n_test_rows={r["n_test"]}')
# same split across cuts?
base=allsplits['k50']
for k in KS:
    tr,te=allsplits[k]
    print(k,'test==k50 test?',te==base[1],'train==?',tr==base[0], 'sym diff test', len(te^base[1]))
# do baselines use the same test rows?
print('\n--- baseline test row keys vs probe split ---')
for k in KS:
    r=json.load(open(f'{B}/{k}/results.json'))
    te=set(r['split']['test_problem_ids']); tr=set(r['split']['train_problem_ids'])
    for name in ['forced_answer','text_classifier','llm_judge']:
        p=f'{B}/{k}/baseline_{name}.json'
        if not os.path.exists(p): continue
        pay=json.load(open(p))
        for kk,pt in pay['per_k'].items():
            rk=pt['test_row_keys']
            pids=set(x.split('@')[0] for x in rk) if rk else None
            print(f'  {k} {name}: n_test={pt["n_test"]} n_pos={pt["n_pos"]} n_neg={pt["n_neg"]} auc={pt["auc"]}'
                  + (f' pid_subset_of_probe_test={pids<=te} extra={len(pids-te)} missing={len(te-pids)} leak_into_train={len(pids&tr)}' if pids else ' (no row keys)'))
