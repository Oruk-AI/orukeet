"""Evaluate a recovery run's fixed checkpoint order on the exposed larger suite."""
import argparse
import json
from pathlib import Path
import subprocess
import time

p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--steps',nargs='+',type=int,required=True);a=p.parse_args()
root=Path('/home/nathanroll/parakeet-ft');exp=root/'gabor_half_20260906';deadline=time.monotonic()+7200
while True:
    state=json.loads((exp/'campaign-status.json').read_text())
    if state.get('phase')=='development-complete' and any(r['label'].startswith(a.label+'-') for r in state.get('results',[])):break
    train=exp/('train-'+a.label+'.log')
    if train.exists() and 'Traceback (most recent call last)' in train.read_text(errors='replace'):raise RuntimeError('Training failed')
    if time.monotonic()>deadline:raise RuntimeError('Development evaluation timeout')
    time.sleep(10)
results=[]
for step in a.steps:
    label=f'{a.label}-{step:04d}';decision=exp/('comparison-'+label+'.json')
    if not decision.exists():continue
    meta=json.loads(decision.read_text())
    model=exp/('orukeet_gabor_half_'+a.label+'_20260906')/f'step-{step:04d}.nemo'
    output=exp/('full-'+label)
    args=['python3',str(exp/'full_regression.py'),'--candidate',str(model),
          '--development-decision',str(decision),'--output',str(output)]
    if meta['status']!='pass':args.append('--diagnostic')
    with (exp/('full-'+label+'.log')).open('x') as log:
        subprocess.run(args,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
    result=json.loads((output/'comparison.json').read_text())
    results.append({'label':label,'status':result['status'],'release_qualified':result['release_qualified'],
        'primary':result['primary'],'english':result['english'],'failures':result['failures']})
    (exp/('full-'+a.label+'-status.json')).write_text(json.dumps({'results':results},indent=2)+'\n')
    if result['status']=='pass':break
