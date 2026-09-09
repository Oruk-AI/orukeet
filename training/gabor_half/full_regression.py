"""Run the larger exposed regression; diagnostic runs never imply qualification."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from identity import SOURCE_SHA,sha

p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--development-decision',type=Path,required=True)
p.add_argument('--diagnostic',action='store_true',help='Measure a failed development candidate without qualifying it')
a=p.parse_args()
root=Path('/home/nathanroll/parakeet-ft')
decision=json.loads(a.development_decision.read_text())
if decision['candidate_sha256']!=sha(a.candidate):raise ValueError('Candidate identity mismatch')
if decision['status']!='pass' and not a.diagnostic:raise ValueError('Candidate did not qualify')
a.output.mkdir(parents=True,exist_ok=False)
(a.output/'qualification.json').write_text(json.dumps({'development_status':decision['status'],
    'development_failures':decision['failures'],'diagnostic_only':a.diagnostic,
    'candidate_sha256':decision['candidate_sha256']},indent=2)+'\n')
registry_path=root/'manifests/goal_v2_confirmation/sealed_metric_registry.json'
registry=json.loads(registry_path.read_text())
reference=root/'eval/goal_v2/selected_confirmation'
meta=json.loads((reference/'results.json').read_text())
assert meta['model_sha256']==SOURCE_SHA
manifests=[str(root/'manifests/goal_v2_confirmation'/(name+'.jsonl')) for name in registry['sets']]
for name,path in zip(registry['sets'],manifests):
    if sha(path)!=registry['sets'][name]['manifest_sha256']:raise ValueError('Membership changed')
# Preserve the evaluator and decoding; the wrapper changes only verified-equivalent
# integer edit distance, avoiding Python's quadratic scoring bottleneck.
with (a.output/'evaluation.log').open('x') as log:
    subprocess.run([str(root/'nemo.sh'),'env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=2',
        'PYTHONPATH='+str(root/'gabor_half_20260906/export-deps'),'python',str(root/'gabor_half_20260906/evaluate_fast.py'),
        '--model',str(a.candidate),'--output',str(a.output/'candidate'),'--batch-size','32',*manifests],
        cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True,env=dict(os.environ,NEMO_NAME='orukeet-gabor-full-regression'))
with (a.output/'statistics.log').open('x') as log:
    subprocess.run([str(root/'nemo.sh'),'env','OPENBLAS_NUM_THREADS=4','OMP_NUM_THREADS=4','python',
        str(root/'gabor_half_20260906/full_compare.py'),'--output',str(a.output)],cwd=root,stdout=log,
        stderr=subprocess.STDOUT,check=True,env=dict(os.environ,NEMO_NAME='orukeet-gabor-full-statistics'))
result=json.loads((a.output/'comparison.json').read_text())
print(json.dumps({k:result[k] for k in ['status','primary','english','failures','rows','hours']}),flush=True)
