"""Reweight existing audited training data and verify no evaluation paths enter."""
import copy
import json
from pathlib import Path
from identity import sha

root=Path('/home/nathanroll/parakeet-ft');exp=root/'gabor_half_20260906'
source=root/'manifests/balanced_global_20260905/input_cfg_with_accents.yaml'
original=json.loads(source.read_text())
global_group={'type':'group','weight':.50,'input_cfg':copy.deepcopy(original)}
english=copy.deepcopy(next(r for r in original if r.get('tags')=={'lang':'en'}));english['weight']=.20
accent=next(r for r in original if r.get('tags',{}).get('cohort')=='accent_extension')
extras=[]
for row in accent['input_cfg']:
    row=copy.deepcopy(row);row['weight']=.20 if row['tags']['src']=='speechocean762' else .10;extras.append(row)
result=[global_group,english,*extras]
assert abs(sum(r['weight'] for r in result)-1)<1e-10
def leaves(nodes):
    return [p for n in nodes for p in (leaves(n['input_cfg']) if n['type']=='group' else [Path(n['manifest_filepath'])])]
old=set(leaves(original));new=set(leaves(result));assert old==new
evaluation=set()
for directory in [exp/'development',root/'manifests/goal_v2_confirmation']:
    for path in directory.glob('*.jsonl'):
        for line in path.open():evaluation.add(json.loads(line)['audio_filepath'])
counts={};overlaps=[]
for path in sorted(new):
    count=0
    for line in path.open():
        row=json.loads(line);count+=1
        if row['audio_filepath'] in evaluation:overlaps.append(row['audio_filepath'])
    counts[str(path)]={'sha256':sha(path),'rows':count}
if overlaps:raise RuntimeError(f'{len(overlaps)} evaluation paths occur in training')
dest=exp/'english-recovery-input.json';dest.write_text(json.dumps(result,indent=2)+'\n')
report={'source_config_sha256':sha(source),'config_sha256':sha(dest),'membership_changed':False,
    'evaluation_audio_overlap':0,'evaluation_paths_checked':len(evaluation),
    'top_level_weights':{'original_mixture':.50,'english_cv_fleurs':.20,'speechocean_train':.20,'english_dialects_train':.10},
    'training_manifests':counts,'interpretation':'Adapted after exposed regression results. Only existing audited training membership is used; this is not a new held-out experiment.'}
(exp/'english-recovery-data-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='training_manifests'}),flush=True)
