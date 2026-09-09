"""Mix the audited R14 and R7 recovery states under unchanged accuracy gates."""
import json
import os
from pathlib import Path
import shutil
import subprocess

from identity import sha, SOURCE_SHA

root = Path('/home/nathanroll/parakeet-ft')
exp = root / 'gabor_half_20260906'
assert 'SHARED_TEACHER_EVALUATION_COMPLETE' in (exp / 'shared-teacher-r14-orchestrator.log').read_text()
assert json.loads((exp / 'r14-full-selection.json').read_text())['source_qualified_selection'] is None
assert json.loads((exp / 'full-r14-0200/comparison.json').read_text())['failures'] == ['English macro WER exceeds original']
assert json.loads((exp / 'full-r7-0300/comparison.json').read_text())['failures'] == ['Primary WER exceeds original']
assert sha(exp / 'average_recovered.py') == '6e2216e9926eae383e10d99301a8e6f3dd5708247ba2a44a41578aadf3454e5d'
assert shutil.disk_usage(exp).free >= 18_000_000_000
parents = [
    ('r14-0200', exp / 'orukeet_gabor_half_r14_20260906/step-0200.nemo',
     '63d462cf943d07785b382db97f179aaf8d9d4b7fe48672bcccdf8cf4e2617a47'),
    ('r7-0300', exp / 'orukeet_gabor_half_r7_20260906/step-0300.nemo',
     '992a35cdfcea8d4d6ca57e8bef4d2fa2ae82a8e1ab02a872598e338faea40ef3')]
for label, path, digest in parents:
    audit = json.loads((exp / (label + '-audit.json')).read_text())
    assert sha(path) == digest == audit['candidate_sha256']
    assert audit['status'] == 'pass' and audit['frozen_gabor_rows_exact'] == 12288
out = exp / 'orukeet_gabor_half_a3_20260906'
out.mkdir(exist_ok=False)
candidates = [
    {'label': 'a3-0150', 'weights': [.85, .15]},
    {'label': 'a3-0250', 'weights': [.75, .25]},
    {'label': 'a3-0350', 'weights': [.65, .35]}]
decision = {
    'label': 'a3', 'method': 'Single float64 weighted sum of trained parameters, rounded once',
    'source_sha256': SOURCE_SHA,
    'parents': [{'label': label, 'path': str(path), 'sha256': digest} for label, path, digest in parents],
    'parent_comparisons_sha256': {
        label: {'development': sha(exp / ('comparison-' + label + '.json')),
                'larger': sha(exp / ('full-' + label) / 'comparison.json')}
        for label, _, _ in parents},
    'candidates': candidates, 'fixed_rows': 12288, 'parameter_tensors_averaged': 651,
    'rationale': 'R14-200 passes development and improves the larger multilingual mean by '
                 '0.022459 points, but misses English by 0.006812 points. R7-300 improves '
                 'English by 0.060395 points while missing the multilingual mean by 0.020564 '
                 'points. Test conservative R7 shares of 15%, 25% and 35% in the R14 state. '
                 'Both endpoints descend from the same fitted, frozen rows; copy these rows '
                 'and all non-parameter buffers exactly. Model-weight interpolation does not '
                 'imply metric interpolation. This is adaptive selection on exposed '
                 'regressions, with no independent generalization claim.',
    'selection_rule': 'Audit and evaluate all three development candidates. Test development-passing '
                      'candidates on the larger regression in increasing development-WER order, '
                      'stopping at the first full pass. If none passes development, run the '
                      'lowest-development-WER candidate once for diagnosis only. Preserve every '
                      'result and explicitly name any larger comparisons not run.',
    'new_optimizer_steps': 0, 'inference_ensemble': False, 'no_gate_changes': True,
    'native_qualification_required_separately': True,
    'code_sha256': {name: sha(exp / name) for name in
                    ['average_recovered.py', 'audit_checkpoint.py', 'compare.py', 'full_regression.py',
                     'run_average_recovery_a3.py']}}
with (exp / 'a3-start-decision.json').open('x') as file:
    file.write(json.dumps(decision, indent=2) + '\n')


def run(arguments, logfile, container=None):
    with (exp / logfile).open('x') as log:
        subprocess.run(arguments, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True,
                       env=dict(os.environ, NEMO_NAME=container) if container else None)


results = []
for item in candidates:
    label = item['label']
    model = out / (label + '.nemo')
    (exp / 'campaign-status.json').write_text(json.dumps({'phase': 'average-and-evaluate', 'candidate': label}) + '\n')
    run(['./nemo.sh', 'env', 'CUDA_VISIBLE_DEVICES=', 'OPENBLAS_NUM_THREADS=2', 'OMP_NUM_THREADS=2',
         'python', str(exp / 'average_recovered.py'), '--parents', *[str(p[1]) for p in parents],
         '--audits', *[str(exp / (p[0] + '-audit.json')) for p in parents],
         '--weights', *[str(w) for w in item['weights']], '--fits', str(exp / 'fit-full/fits.json'),
         '--coverage', str(exp / 'orukeet_gabor_half_r14_20260906/gradient-coverage.json'),
         '--output', str(model)], 'average-' + label + '.log', 'orukeet-average-' + label)
    run(['./nemo.sh', 'env', 'CUDA_VISIBLE_DEVICES=', 'OPENBLAS_NUM_THREADS=2', 'OMP_NUM_THREADS=2',
         'python', str(exp / 'audit_checkpoint.py'), '--original',
         str(root / 'models/ft/stage3_baseblend_a075_20260905.nemo'), '--candidate', str(model),
         '--fits', str(exp / 'fit-full/fits.json'), '--output', str(exp / (label + '-audit.json'))],
        'audit-' + label + '.log', 'orukeet-audit-' + label)
    run(['bash', str(exp / 'run_eval.sh'), str(model), label], 'eval-' + label + '.log')
    subprocess.run(['python3', str(exp / 'compare.py'), '--reference', str(exp / 'eval-original'),
                    '--candidate', str(exp / ('eval-' + label)), '--registry',
                    str(exp / 'development/registry.json'), '--output',
                    str(exp / ('comparison-' + label + '.json'))], cwd=root, check=True)
    result = json.loads((exp / ('comparison-' + label + '.json')).read_text())
    results.append({'label': label, 'model': str(model), 'weights': item['weights'],
                    'status': result['status'], 'primary': result['primary'], 'failures': result['failures']})
(exp / 'campaign-status.json').write_text(json.dumps({'phase': 'development-complete', 'results': results}) + '\n')
passing = sorted((row for row in results if row['status'] == 'pass'), key=lambda row: row['primary']['candidate_wer'])
order = passing or [min(results, key=lambda row: row['primary']['candidate_wer'])]
selection = {'rule': decision['selection_rule'], 'development_candidates': results,
             'larger_order': [r['label'] for r in order], 'diagnostic_only': not passing,
             'larger_results': [], 'source_qualified_selection': None}
selection_path = exp / 'a3-selection.json'
with selection_path.open('x') as file:
    file.write(json.dumps(selection, indent=2) + '\n')
for row in order:
    label = row['label']
    args = ['python3', str(exp / 'full_regression.py'), '--candidate', row['model'],
            '--development-decision', str(exp / ('comparison-' + label + '.json')),
            '--output', str(exp / ('full-' + label))]
    if not passing:
        args.append('--diagnostic')
    run(args, 'full-' + label + '.log')
    result = json.loads((exp / ('full-' + label) / 'comparison.json').read_text())
    selection['larger_results'].append({'label': label, 'source_qualified': result['release_qualified'],
                                        'comparison_sha256': sha(exp / ('full-' + label) / 'comparison.json')})
    if result['release_qualified']:
        selection['source_qualified_selection'] = row
    selection_path.write_text(json.dumps(selection, indent=2) + '\n')
    if result['release_qualified']:
        break
tested = {r['label'] for r in selection['larger_results']}
selection['larger_comparisons_not_run'] = [r['label'] for r in results if r['label'] not in tested]
selection['native_qualified'] = False
selection_path.write_text(json.dumps(selection, indent=2) + '\n')
print('AVERAGE_RECOVERY_EVALUATION_COMPLETE', flush=True)
