import json, os, sys, glob
sys.path.insert(0,'/Users/IsahanKhan/ais-research')
from experiment.grading import grade, extract_boxed
from experiment.dataset_adapters import OPTION_LABELS
from tokenizers import Tokenizer
snap=glob.glob('/Users/IsahanKhan/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/*/tokenizer.json')[0]
tok=Tokenizer.from_file(snap)
BASE='/Users/IsahanKhan/ais-research/experiment/outputs/block2'
THINK_END=151668
rows=[]
with open(f'{BASE}/gen/traces.jsonl') as f:
    for line in f:
        r=json.loads(line)
        if r.get('record_type')=='meta': continue
        rows.append(r)
mismatch=0; gold_mismatch=0; n_none=0; n_trunc=0
regrade_disagree=[]
for r in rows:
    ai=r['meta'].get('answer_index')
    if isinstance(ai,int) and OPTION_LABELS[ai]!=r['gold_answer']: gold_mismatch+=1
    tt=r['trace_token_ids']
    if THINK_END not in tt:
        n_trunc+=1
        if r['correct'] is not None: regrade_disagree.append((r['problem_id'],'truncated but labeled',r['correct']))
        continue
    end=tt.index(THINK_END)
    resp=tok.decode(tt[end+1:], skip_special_tokens=True)
    g=grade(resp, r['gold_answer'])
    if g is None: n_none+=1
    if g!=r['correct']:
        regrade_disagree.append((r['problem_id'], g, r['correct'], repr(extract_boxed(resp)), r['gold_answer']))
print('n rows',len(rows),'gold_vs_answer_index_mismatch',gold_mismatch,'truncated',n_trunc,'ungradeable_on_regrade',n_none)
print('regrade disagreements:',len(regrade_disagree))
for x in regrade_disagree[:20]: print('  ',x)
from collections import Counter
print(Counter([r['correct'] for r in rows]))
