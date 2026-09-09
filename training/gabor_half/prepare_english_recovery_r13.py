"""Retain all training membership with a modest English sampling increase."""
import copy
import json
from pathlib import Path

from identity import sha
from prepare_slice_recovery import leaves

root = Path('/home/nathanroll/parakeet-ft')
exp = root / 'gabor_half_20260906'
source = root / 'manifests/balanced_global_20260905/input_cfg_with_accents.yaml'
original = json.loads(source.read_text())
english = copy.deepcopy(next(group for group in original if group.get('tags') == {'lang': 'en'}))
english['weight'] = .2
mixture = [{'type': 'group', 'weight': .8, 'input_cfg': copy.deepcopy(original)}, english]
assert leaves(mixture) == leaves(original)
evaluation = set()
for directory in [exp / 'development', root / 'manifests/goal_v2_confirmation']:
    for path in directory.glob('*.jsonl'):
        evaluation.update(json.loads(line)['audio_filepath'] for line in path.open())
assert len(evaluation) == 28138
counts = {}
for path in sorted(leaves(mixture)):
    rows = 0
    for line in path.open():
        assert json.loads(line)['audio_filepath'] not in evaluation
        rows += 1
    counts[str(path)] = {'rows': rows, 'sha256': sha(path)}
assert sum(item['rows'] for item in counts.values()) == 1716650
output = exp / 'english-recovery-input-r13.json'
with output.open('x') as file:
    file.write(json.dumps(mixture, indent=2) + '\n')
audit = {
    'source_config_sha256': sha(source), 'config_sha256': sha(output),
    'membership_changed': False, 'exact_evaluation_audio_path_overlap': 0,
    'evaluation_paths_checked': len(evaluation), 'training_manifests': counts,
    'top_level_weights': {'original_mixture': .8, 'english_cv_fleurs': .2},
    'interpretation': 'English emphasis after exposed regression feedback. All original training '
                      'memberships remain; exact path checks do not establish acoustic or speaker independence.'}
with output.with_suffix('.audit.json').open('x') as file:
    file.write(json.dumps(audit, indent=2) + '\n')
print(json.dumps({key: value for key, value in audit.items() if key != 'training_manifests'}), flush=True)
