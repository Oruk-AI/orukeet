"""Audit and evaluate completed recovery exports, preserving every result."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--steps',type=int,nargs='+',required=True)
p.add_argument('--label',required=True);a=p.parse_args()
ROOT=Path('/home/nathanroll/parakeet-ft');EXP=ROOT/'gabor_half_20260906';os.chdir(ROOT)
log=EXP/('train-'+a.label+'.log')
deadline=time.monotonic()+7200
while 'RECOVERY_COMPLETE' not in log.read_text(errors='replace'):
    if 'Traceback (most recent call last)' in log.read_text(errors='replace'):raise RuntimeError('Training failed; inspect '+str(log))
    if time.monotonic()>deadline:raise RuntimeError('Training timeout')
    time.sleep(10)
results=[]
for step in a.steps:
    label=f'{a.label}-{step:04d}';model=EXP/a.run/f'step-{step:04d}.nemo'
    (EXP/'campaign-status.json').write_text(json.dumps({'phase':'audit-and-evaluate','candidate':label},indent=2)+'\n')
    with (EXP/('audit-'+label+'.log')).open('x') as out:
        subprocess.run(['./nemo.sh','env','OPENBLAS_NUM_THREADS=2','OMP_NUM_THREADS=2','python',str(EXP/'audit_checkpoint.py'),
            '--original',str(ROOT/'models/ft/stage3_baseblend_a075_20260905.nemo'),'--candidate',str(model),
            '--fits',str(EXP/'fit-full/fits.json'),'--output',str(EXP/(label+'-audit.json'))],
            env=dict(os.environ,NEMO_NAME='orukeet-gabor-audit-'+label),stdout=out,stderr=subprocess.STDOUT,check=True)
    with (EXP/('eval-'+label+'.log')).open('x') as out:
        subprocess.run(['bash',str(EXP/'run_eval.sh'),str(model),label],stdout=out,stderr=subprocess.STDOUT,check=True)
    subprocess.run(['python3',str(EXP/'compare.py'),'--reference',str(EXP/'eval-original'),
        '--candidate',str(EXP/('eval-'+label)),'--registry',str(EXP/'development/registry.json'),
        '--output',str(EXP/('comparison-'+label+'.json'))],check=True)
    result=json.loads((EXP/('comparison-'+label+'.json')).read_text())
    results.append({'label':label,'model':str(model),'status':result['status'],'primary':result['primary'],'failures':result['failures']})
# Evaluate every requested export so the recorded minimum-WER selection rule
# is applied to the complete declared candidate set, including after a pass.
(EXP/'campaign-status.json').write_text(json.dumps({'phase':'development-complete','results':results,
    'selected':next((r for r in results if r['status']=='pass'),None)},indent=2)+'\n')
