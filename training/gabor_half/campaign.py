"""Private, sequential recovery campaign on the existing dedicated GPU host."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path('/home/nathanroll/parakeet-ft');EXP=ROOT/'gabor_half_20260906'
SOURCE=ROOT/'models/ft/stage3_baseblend_a075_20260905.nemo'
os.chdir(ROOT)

def state(phase,**extra):
    (EXP/'campaign-status.json').write_text(json.dumps({'phase':phase,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**extra},indent=2)+'\n')
    print(phase,flush=True)

def run(label,args,container=True,env=None):
    state(label)
    command=['./nemo.sh','env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=2']+args if container else args
    with (EXP/(label+'.log')).open('x') as log:
        subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,
                       env=dict(os.environ,NEMO_NAME='orukeet-gabor-'+label,**(env or {})))

def audit(model,label):
    run('audit-'+label,['python',str(EXP/'audit_checkpoint.py'),'--original',str(SOURCE),'--candidate',str(model),
                       '--fits',str(EXP/'fit-full/fits.json'),'--output',str(EXP/(label+'-audit.json'))])

def evaluate(model,label):
    run('eval-'+label,['bash',str(EXP/'run_eval.sh'),str(model),label],container=False)
    run('compare-'+label,['python',str(EXP/'compare.py'),'--reference',str(EXP/'eval-original'),
        '--candidate',str(EXP/('eval-'+label)),'--registry',str(EXP/'development/registry.json'),
        '--output',str(EXP/('comparison-'+label+'.json'))])

try:
    state('waiting-for-fits-and-original-evaluation')
    deadline=time.monotonic()+7200
    while not ((EXP/'fit-full/summary.json').exists() and 'EVALUATION_COMPLETE' in (EXP/'eval-original.log').read_text(errors='replace')):
        if time.monotonic()>deadline:raise RuntimeError('Timed out waiting for preparation')
        time.sleep(10)
    run('audit-fit',['python',str(EXP/'audit_fit.py'),str(EXP/'fit-full')])
    run('export-zero',['python',str(EXP/'export_initial.py'),'--source',str(SOURCE),'--fits',str(EXP/'fit-full/fits.json'),
        '--output',str(EXP/'zero-shot.nemo')])
    audit(EXP/'zero-shot.nemo','zero')
    evaluate(EXP/'zero-shot.nemo','zero')
    smoke='orukeet_gabor_half_smoke_20260906'
    run('train-smoke',['bash',str(EXP/'run_recovery.sh'),'model.optim.sched.warmup_steps=0','++export_interval=2'],
        container=False,env={'NAME':smoke,'STEPS':'2'})
    audit(EXP/smoke/'step-0002.nemo','smoke')
    name='orukeet_gabor_half_r1_20260906'
    run('train-r1',['bash',str(EXP/'run_recovery.sh')],container=False)
    comparisons=[]
    # Evaluate the final checkpoint first; earlier exports are fallback candidates.
    for step in [400,300,200,100]:
        label=f'r1-{step:04d}';model=EXP/name/f'step-{step:04d}.nemo'
        audit(model,label);evaluate(model,label)
        result=json.loads((EXP/('comparison-'+label+'.json')).read_text())
        comparisons.append({'label':label,'path':str(model),'status':result['status'],'primary':result['primary']})
        if result['status']=='pass':break
        if step==400 and result['primary']['delta_pp']>1.0:
            state('diagnosis-required',comparisons=comparisons,reason='Large recovery regression; inspect normalization state before evaluating more snapshots')
            raise SystemExit(0)
    passing=[r for r in comparisons if r['status']=='pass']
    state('development-complete',comparisons=comparisons,selected=passing[0] if passing else None,
          remaining='Larger matched release regression and inference export validation; no release promotion performed')
except Exception as exc:
    state('failed',error=repr(exc));raise
