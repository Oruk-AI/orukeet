"""Matched-runtime statistics in the same NeMo image as the experiment."""
import argparse
import json
from pathlib import Path
import sys
from identity import sha
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
root=Path('/home/nathanroll/parakeet-ft')
registry_path=root/'manifests/goal_v2_confirmation/sealed_metric_registry.json'
registry=json.loads(registry_path.read_text())
reference=root/'eval/goal_v2/selected_confirmation'
meta=json.loads((reference/'results.json').read_text())
new=json.loads((a.output/'candidate/results.json').read_text())
assert new['decoding']==meta['decoding'] and new['normalizer']==meta['normalizer']
# Use the repository comparator; remote staging copies this exact file beside
# the script instead of relying on a VM-only historical module.
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'evaluation'))
from compare_oruk_export import compare
import compare_oruk_export
result=compare(reference,a.output/'candidate',registry)
result['interpretation']='Matched NeMo/bf16 greedy decoding; existing exposed release recordings. Not an unseen-data or universal-accuracy claim.'
failures=[]
if result['primary']['delta_pp']>0:failures.append('Primary WER exceeds original')
if result['english']['delta_pp']>0:failures.append('English macro WER exceeds original')
for name,metric in result['metrics'].items():
    cap=.5 if name.startswith('language:') else .3 if name.startswith('english:') else None
    if cap is not None and metric['delta_pp']>cap:failures.append(name+' exceeds regression guardrail')
result['status']='pass' if not failures else 'fail';result['failures']=failures
qualification=json.loads((a.output/'qualification.json').read_text())
result['development_status']=qualification['development_status']
result['diagnostic_only']=qualification['diagnostic_only']
result['release_qualified']=not failures and qualification['development_status']=='pass' and not qualification['diagnostic_only']
result['registry_sha256']=sha(registry_path)
result['statistics_implementation_sha256']=sha(compare_oruk_export.__file__)
if new.get('metric_implementation'):
    result['candidate_metric_implementation']=new['metric_implementation']
(a.output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['status','primary','english','failures','rows','hours']}),flush=True)
