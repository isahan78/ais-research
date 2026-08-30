import json, os
import numpy as np
BASE='/Users/IsahanKhan/ais-research/experiment/outputs/block2'
traces={}
with open(os.path.join(BASE,'gen','traces.jsonl')) as f:
    for line in f:
        r=json.loads(line)
        if r.get('record_type')=='meta': continue
        traces[r['problem_id']]=r

for k in [1,10,25,50,75,90]:
    d=f'{BASE}/k{k}'
    inc=[]
    with open(f'{d}/prefixes.jsonl') as f:
        for line in f:
            r=json.loads(line)
            if r.get('record_type')=='meta' or not r['included']: continue
            inc.append(r)
    z=np.load(f'{d}/acts.npz', allow_pickle=False)
    pids=[str(p) for p in z['problem_ids']]
    labels=z['labels']
    pre_pids=[r['problem_id'] for r in inc]
    pre_labels=[bool(r['label']) for r in inc]
    same_order = pids==pre_pids
    same_set = set(pids)==set(pre_pids)
    lab_match = list(labels)==pre_labels
    trace_lab_match = all(bool(traces[p]['correct'])==bool(l) for p,l in zip(pids,labels))
    dup = len(pids)!=len(set(pids))
    print(f'k={k}: n_acts={len(pids)} n_pref_inc={len(inc)} order_match={same_order} set_match={same_set} labels_match_prefixes={lab_match} labels_match_traces={trace_lab_match} dup_pids={dup}')
    print('    npz keys:', list(z.keys()), 'shapes:', {kk: z[kk].shape for kk in z if kk.startswith('acts_')}, 'config_hash', str(z['config_hash']))
    # check activations differ across rows and across layers
    for L in (9,18,27):
        A=z[f'acts_layer{L}']
        print(f'    layer{L}: mean|x|={np.abs(A).mean():.4f} std={A.std():.4f} n_unique_rows={len(np.unique(A,axis=0))}')
    # cross-layer identity check
    print('    L9==L18?', np.array_equal(z['acts_layer9'],z['acts_layer18']), 'L18==L27?', np.array_equal(z['acts_layer18'],z['acts_layer27']))
