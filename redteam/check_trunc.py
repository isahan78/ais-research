import json, os, sys, random
BASE='/Users/IsahanKhan/ais-research/experiment/outputs/block2'
THINK_START=151667; THINK_END=151668

traces={}
with open(os.path.join(BASE,'gen','traces.jsonl')) as f:
    for line in f:
        r=json.loads(line)
        if r.get('record_type')=='meta': continue
        traces[r['problem_id']]=r
print('n traces', len(traces))

# where is <think> ?
n_prompt_has_think=0; n_trace_starts_think=0; n_trace_has_think=0
for pid,r in traces.items():
    if THINK_START in r['prompt_token_ids']: n_prompt_has_think+=1
    if r['trace_token_ids'] and r['trace_token_ids'][0]==THINK_START: n_trace_starts_think+=1
    if THINK_START in r['trace_token_ids']: n_trace_has_think+=1
print('prompt contains <think>:',n_prompt_has_think,'trace[0]==<think>:',n_trace_starts_think,'trace contains <think>:',n_trace_has_think)

for k in [1,10,25,50,75,90]:
    d=os.path.join(BASE,f'k{k}')
    rows=[]
    with open(os.path.join(d,'prefixes.jsonl')) as f:
        for line in f:
            r=json.loads(line)
            if r.get('record_type')=='meta': continue
            rows.append(r)
    inc=[r for r in rows if r['included']]
    bad=[]
    fracs=[]
    for r in inc:
        p=r['prefix_token_ids']; pid=r['problem_id']
        t=traces[pid]
        # recompute expected
        tt=t['trace_token_ids']
        end=tt.index(THINK_END)
        if THINK_START in tt[:end]:
            start=tt.index(THINK_START); head=tt[:start+1]
        else:
            start=-1; head=[]
        thinking=tt[start+1:end]
        n_think=len(thinking)
        n_keep=int(n_think*k/100); n_keep=max(1,min(n_keep,n_think-1))
        expected=list(t['prompt_token_ids'])+head+thinking[:n_keep]
        issues=[]
        if p!=expected: issues.append('prefix != recomputed')
        if THINK_END in p: issues.append('</think> present')
        if p.count(THINK_START)!=1: issues.append(f'<think> count={p.count(THINK_START)}')
        if r['n_thinking_tokens']!=n_think: issues.append('n_think mismatch')
        if r['n_kept_thinking_tokens']!=n_keep: issues.append('n_keep mismatch')
        if r['prompt_token_ids']!=t['prompt_token_ids']: issues.append('prompt mismatch')
        if p[:len(r['prompt_token_ids'])]!=r['prompt_token_ids']: issues.append('prompt not prefix')
        if r['label']!=t['correct']: issues.append('label != correct')
        fracs.append(n_keep/n_think)
        if issues: bad.append((pid,issues))
    import statistics
    print(f'k={k}: included={len(inc)} bad={len(bad)} frac_kept mean={statistics.mean(fracs):.5f} min={min(fracs):.5f} max={max(fracs):.5f}')
    for b in bad[:5]: print('   ',b)
