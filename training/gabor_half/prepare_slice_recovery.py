"""Retain the complete training pool while emphasizing failing source/language slices."""
import argparse
import copy
import json
from pathlib import Path

from identity import sha


def curriculum(original, decision):
    targets = sorted({tuple(name.split('_')[:2]) for name, metric in decision['sets'].items()
                      if name.startswith(('cv_', 'fleurs_')) and metric['delta_pp'] > .5
                      and name.split('_')[1] != 'en'})
    if decision['status'] != 'fail' or not targets:
        raise ValueError('Expected measured non-English development failures')
    result = [{'type': 'group', 'weight': .6, 'input_cfg': copy.deepcopy(original)}]
    for source, language in targets:
        language_group = next(g for g in original if g.get('tags') == {'lang': language})
        source_group = copy.deepcopy(next(g for g in language_group['input_cfg']
                                          if g.get('tags') == {'src': source}))
        source_group.update(weight=.2 / len(targets), tags={'lang': language, 'src': source})
        result.append(source_group)
    english = copy.deepcopy(next(g for g in original if g.get('tags') == {'lang': 'en'}))
    accent = copy.deepcopy(next(g for g in original
                                if g.get('tags', {}).get('cohort') == 'accent_extension'))
    english['weight'] = accent['weight'] = .1
    result.extend([english, accent])
    assert abs(sum(g['weight'] for g in result) - 1.) < 1e-10
    return result, targets


def leaves(nodes):
    return {p for n in nodes for p in (leaves(n['input_cfg']) if n['type'] == 'group'
                                       else {Path(n['manifest_filepath'])})}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--decision', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path('/home/nathanroll/parakeet-ft')
    exp = root / 'gabor_half_20260906'
    source = root / 'manifests/balanced_global_20260905/input_cfg_with_accents.yaml'
    original = json.loads(source.read_text())
    decision = json.loads(args.decision.read_text())
    result, targets = curriculum(original, decision)
    before, after = leaves(original), leaves(result)
    assert before == after
    evaluation = set()
    for directory in [exp / 'development', root / 'manifests/goal_v2_confirmation']:
        for path in directory.glob('*.jsonl'):
            evaluation.update(json.loads(line)['audio_filepath'] for line in path.open())
    assert len(evaluation) == 28138
    counts = {}
    for path in sorted(after):
        rows = 0
        for line in path.open():
            if json.loads(line)['audio_filepath'] in evaluation:
                raise ValueError('Evaluation path found in training')
            rows += 1
        counts[str(path)] = {'rows': rows, 'sha256': sha(path)}
    with args.output.open('x') as file:
        file.write(json.dumps(result, indent=2) + '\n')
    audit = {'source_config_sha256': sha(source), 'decision_sha256': sha(args.decision),
             'candidate_sha256': decision['candidate_sha256'], 'config_sha256': sha(args.output),
             'target_source_languages': [list(t) for t in targets], 'membership_changed': False,
             'exact_evaluation_audio_path_overlap': 0, 'evaluation_paths_checked': len(evaluation),
             'training_manifests': counts,
             'top_level_weights': {'original_mixture': .6, 'target_slices_equal': .2,
                                  'english_cv_fleurs': .1, 'english_accent_extension': .1},
             'interpretation': 'Outcome-informed curriculum on existing training membership. '
                               'Source-specific emphasis and a larger original-mixture weight '
                               'are a recovery hypothesis, not evidence of independent generalization.'}
    args.output.with_suffix('.audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps({k: v for k, v in audit.items() if k != 'training_manifests'}), flush=True)


if __name__ == '__main__':
    main()
