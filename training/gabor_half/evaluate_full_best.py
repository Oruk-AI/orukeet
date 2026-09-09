"""Freeze one trained candidate for the larger regression after R4 development."""
import json
import os
from pathlib import Path
import subprocess
import time

root=Path('/home/nathanroll/parakeet-ft');exp=root/'gabor_half_20260906'
deadline=time.monotonic()+7200
while True:
    state=json.loads((exp/'campaign-status.json').read_text())
    if state.get('phase')=='development-complete' and any(r['label'].startswith('r4-') for r in state.get('results',[])):break
    if (exp/'train-r4.log').exists() and 'Traceback (most recent call last)' in (exp/'train-r4.log').read_text(errors='replace'):
        raise RuntimeError('R4 failed; full-run selection is paused')
    if time.monotonic()>deadline:raise RuntimeError('R4 development timeout')
    time.sleep(10)
candidates=[]
for run in ('r2','r3','r4'):
    for comparison in sorted(exp.glob('comparison-'+run+'-*.json')):
        result=json.loads(comparison.read_text());label=comparison.stem.removeprefix('comparison-')
        step=int(label.rsplit('-',1)[1]);model=exp/('orukeet_gabor_half_'+run+'_20260906')/f'step-{step:04d}.nemo'
        candidates.append({'label':label,'model':str(model),'decision':str(comparison),
            'status':result['status'],'primary_wer':result['primary']['candidate_wer'],
            'candidate_sha256':result['candidate_sha256']})
qualified=[r for r in candidates if r['status']=='pass']
selected=min(qualified or candidates,key=lambda r:(r['primary_wer'],r['label']))
selection={'rule':'Choose lowest development primary WER among qualified trained R2-R4 exports; if none qualify, measure the lowest-WER trained export as diagnostic only. No candidate selected using this larger evaluation.',
    'selected':selected,'diagnostic_only':not qualified,'considered':candidates}
(exp/'full-selection.json').write_text(json.dumps(selection,indent=2)+'\n')
args=['python3',str(exp/'full_regression.py'),'--candidate',selected['model'],
      '--development-decision',selected['decision'],'--output',str(exp/'full-best')]
if not qualified:args.append('--diagnostic')
subprocess.run(args,cwd=root,check=True)
