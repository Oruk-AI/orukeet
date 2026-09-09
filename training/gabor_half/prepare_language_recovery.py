"""Prepare an audited curriculum for remaining development language regressions."""
import argparse
import copy
import json
from pathlib import Path

from identity import sha

p = argparse.ArgumentParser()
p.add_argument('--decision', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
root = Path('/home/nathanroll/parakeet-ft')
exp = root / 'gabor_half_20260906'
decision = json.loads(a.decision.read_text())
languages = sorted({name.split('_')[1] for name, value in decision['sets'].items()
    if name.startswith(('cv_', 'fleurs_')) and value['delta_pp'] > .5
    and name.split('_')[1] != 'en'})
if decision['status'] != 'fail' or not languages:
    raise ValueError('This curriculum requires a measured non-English language regression')
source = root / 'manifests/balanced_global_20260905/input_cfg_with_accents.yaml'
original = json.loads(source.read_text())
result = [{'type': 'group', 'weight': .4, 'input_cfg': copy.deepcopy(original)}]
for language in languages:
    group = copy.deepcopy(next(g for g in original if g.get('tags') == {'lang': language}))
    group['weight'] = .4 / len(languages)
    result.append(group)
english = copy.deepcopy(next(g for g in original if g.get('tags') == {'lang': 'en'}))
english['weight'] = .12
accent = copy.deepcopy(next(g for g in original if g.get('tags', {}).get('cohort') == 'accent_extension'))
accent['weight'] = .08
result.extend([english, accent])
assert abs(sum(g['weight'] for g in result) - 1.) < 1e-10


def leaves(nodes):
    return [p for n in nodes for p in (leaves(n['input_cfg']) if n['type'] == 'group'
                                      else [Path(n['manifest_filepath'])])]


before, after = set(leaves(original)), set(leaves(result))
assert before == after
evaluation = set()
for directory in (exp / 'development', root / 'manifests/goal_v2_confirmation'):
    for path in directory.glob('*.jsonl'):
        for line in path.open():
            evaluation.add(json.loads(line)['audio_filepath'])
counts = {}
for path in sorted(after):
    rows = 0
    for line in path.open():
        record = json.loads(line)
        if record['audio_filepath'] in evaluation:
            raise ValueError('Evaluation audio path found in training')
        rows += 1
    counts[str(path)] = {'rows': rows, 'sha256': sha(path)}
with a.output.open('x') as f:
    f.write(json.dumps(result, indent=2) + '\n')
audit = {'source_config_sha256': sha(source), 'decision_sha256': sha(a.decision),
    'candidate_sha256': decision['candidate_sha256'], 'config_sha256': sha(a.output),
    'target_languages': languages, 'membership_changed': False, 'exact_evaluation_audio_path_overlap': 0,
    'evaluation_paths_checked': len(evaluation), 'training_manifests': counts,
    'top_level_weights': {'original_mixture': .4, 'target_languages_equal': .4,
                          'english_cv_fleurs': .12, 'english_accent_extension': .08},
    'interpretation': 'Curriculum adapted from exposed development language regressions. Only existing audited training membership is used; exact path checks do not prove acoustic or speaker independence.'}
a.output.with_suffix('.audit.json').write_text(json.dumps(audit, indent=2) + '\n')
print(json.dumps({k: v for k, v in audit.items() if k != 'training_manifests'}), flush=True)
