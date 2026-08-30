import json, random, sys
sys.path.insert(0,'/Users/IsahanKhan/ais-research')
from experiment.forced_answer import grade_forced_answer, forced_answer_text, FORCED_ANSWER_SUFFIX
BASE='/Users/IsahanKhan/ais-research/experiment/outputs/block2'
gold={}
for line in open(f'{BASE}/gen/traces.jsonl'):
    r=json.loads(line)
    if r.get('record_type')=='meta': continue
    gold[r['problem_id']]=r['gold_answer']
for k in (25,90):
    rows=[json.loads(l) for l in open(f'{BASE}/k{k}/forced_answer.jsonl')]
    rows=[r for r in rows if r.get('record_type')!='meta']
    random.seed(3)
    bad=0
    for r in rows:
        g=grade_forced_answer(FORCED_ANSWER_SUFFIX, r['generated_text'], gold[r['problem_id']])
        if g is not r['forced_correct']: bad+=1
    print(f'k={k}: regrade disagreements = {bad}/{len(rows)}')
    for r in random.sample(rows,6):
        print(f"  pid={r['problem_id']} gold={gold[r['problem_id']]} forced_ans={r['forced_answer']!r} forced_correct={r['forced_correct']} label={r['label']} gen={r['generated_text'][:60]!r}")
