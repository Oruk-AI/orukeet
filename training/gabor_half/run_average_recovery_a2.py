"""Test nearby teacher-recovery checkpoint averages under the original gates."""
import json
import os
from pathlib import Path
import shutil
import subprocess

from identity import sha, SOURCE_SHA

root = Path('/home/nathanroll/parakeet-ft')
exp = root / 'gabor_half_20260906'
assert 'TEACHER_CONTINUATION_EVALUATION_COMPLETE' in (exp / 'teacher-continuation-r12-orchestrator.log').read_text()
assert json.loads((exp / 'full-r12-0800/comparison.json').read_text())['release_qualified'] is False
assert shutil.disk_usage(exp).free >= 18_000_000_000
parents = [
    ('r11-0400', exp / 'orukeet_gabor_half_r11_20260906/step-0400.nemo',
     'ac0b8f43d67580bafd79614207bf012037147fb792b6a117f02bf0cfb85e92be'),
    ('r12-0800', exp / 'orukeet_gabor_half_r12_20260906/step-0800.nemo',
     'd1610da004fa008fa9475895ee509998fee1f8986de6aa002107ef78eab70f3d')]
for label, path, digest in parents:
    audit = json.loads((exp / (label + '-audit.json')).read_text())
    assert sha(path) == digest == audit['candidate_sha256']
    assert audit['status'] == 'pass' and audit['frozen_gabor_rows_exact'] == 12288
out = exp / 'orukeet_gabor_half_a2_20260906'
out.mkdir(exist_ok=False)
candidates = [
    {'label': 'a2-0350', 'weights': [.65, .35]},
    {'label': 'a2-0500', 'weights': [.5, .5]},
    {'label': 'a2-0650', 'weights': [.35, .65]}]
decision = {
    'label': 'a2', 'method': 'Single float64 weighted sum of trained parameters, rounded once',
    'source_sha256': SOURCE_SHA,
    'parents': [{'label': label, 'path': str(path), 'sha256': digest} for label, path, digest in parents],
    'parent_comparisons_sha256': {
        label: {'development': sha(exp / ('comparison-' + label + '.json')),
                'larger': sha(exp / ('full-' + label) / 'comparison.json')}
        for label, _, _ in parents},
    'candidates': candidates, 'fixed_rows': 12288, 'parameter_tensors_averaged': 651,
    'rationale': 'R11 and its R12 continuation use the same strong teacher objective and original '
                 'training mixture. They each miss the larger multilingual mean by about 0.008 '
                 'points while passing English and every larger individual guard. Their sole '
                 'development failures differ: Finnish for R11, Latvian for R12. Test direct '
                 'averages with R12 shares of 35%, 50%, 65%, retaining the fixed fits and '
                 'unchanged buffers. This is adaptive recovery on exposed regressions; neither '
                 'WER nor qualification is assumed to interpolate between the parents.',
    'selection_rule': 'Audit and evaluate all three development candidates. Test development-passing '
                      'candidates on the larger regression in increasing development-WER order, '
                      'stopping at the first full pass. If none passes development, run the '
                      'lowest-development-WER candidate once for diagnosis only. Preserve every '
                      'result and explicitly name any larger comparisons not run.',
    'new_optimizer_steps': 0, 'inference_ensemble': False, 'no_gate_changes': True,
    'native_qualification_required_separately': True,
    'code_sha256': {name: sha(exp / name) for name in
                    ['average_recovered.py', 'audit_checkpoint.py', 'compare.py', 'full_regression.py']}}
with (exp / 'a2-start-decision.json').open('x') as file:
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
         '--coverage', str(exp / 'orukeet_gabor_half_r12_20260906/gradient-coverage.json'),
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
selection_path = exp / 'a2-selection.json'
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
