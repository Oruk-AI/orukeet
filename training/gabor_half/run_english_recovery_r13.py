"""Short English-emphasized continuation of the development-passing A2 average."""
import json
import os
from pathlib import Path
import shutil
import subprocess

from identity import sha, SOURCE_SHA

root = Path('/home/nathanroll/parakeet-ft')
exp = root / 'gabor_half_20260906'
name = 'orukeet_gabor_half_r13_20260906'
assert 'AVERAGE_RECOVERY_EVALUATION_COMPLETE' in (exp / 'average-recovery-a2-orchestrator.log').read_text()
previous = json.loads((exp / 'a2-selection.json').read_text())
assert previous['source_qualified_selection'] is None
start = exp / 'orukeet_gabor_half_a2_20260906/a2-0650.nemo'
start_sha = '44059de88f9797dcd9a3386e1b267733937326c518b2d570468911b6f7da18db'
audit = json.loads((exp / 'a2-0650-audit.json').read_text())
development = json.loads((exp / 'comparison-a2-0650.json').read_text())
comparison = json.loads((exp / 'full-a2-0650/comparison.json').read_text())
assert sha(start) == start_sha == audit['candidate_sha256'] == development['candidate_sha256']
assert audit['status'] == development['status'] == 'pass' and audit['frozen_gabor_rows_exact'] == 12288
assert comparison['candidate_model_sha256'] == start_sha and comparison['release_qualified'] is False
assert comparison['failures'] == ['English macro WER exceeds original']
data = exp / 'english-recovery-input-r13.json'
data_audit = json.loads(data.with_suffix('.audit.json').read_text())
assert sha(data) == data_audit['config_sha256']
assert data_audit['top_level_weights'] == {'original_mixture': .8, 'english_cv_fleurs': .2}
assert not data_audit['membership_changed'] and data_audit['exact_evaluation_audio_path_overlap'] == 0
assert data_audit['evaluation_paths_checked'] == 28138
for path, record in data_audit['training_manifests'].items():
    assert sha(path) == record['sha256']
assert shutil.disk_usage(exp).free >= 25_000_000_000
launcher = exp / (name + '-launch.sh')
with launcher.open('xb') as file:
    file.write((exp / 'run_recovery.sh').read_bytes())
args = ['bash', str(launcher), '++export_interval=100', '++seed=20260910',
        '++model.train_ds.seed=20260910', '++model.train_ds.shard_seed=20260910',
        '++recovery_start_model=' + str(start),
        '++recovery_start_audit=' + str(exp / 'a2-0650-audit.json'),
        '++model.train_ds.input_cfg=' + str(data),
        'model.optim.lr=1e-6', 'model.optim.sched.min_lr=0.0',
        'model.optim.sched.warmup_steps=10', '++preserve_bn_statistics=true',
        '++disable_dropout=true', '++resume_keep=1',
        '++layer_anchor_scale=50.0', '++conv_anchor_scale=25.0', '++asr_loss_scale=0.1',
        '++recovery_layer_lrs.lower_encoder=5e-7', '++recovery_layer_lrs.upper_encoder=1e-6',
        '++recovery_layer_lrs.conv_neighbors=1e-5',
        '++recovery_layer_lrs.other=5e-7', '++recovery_layer_lrs.decoder=5e-8',
        '++recovery_layer_lrs.joint=5e-8',
        'model.spec_augment.freq_masks=0', 'model.spec_augment.time_masks=0']
decision = {
    'label': 'r13', 'source_sha256': SOURCE_SHA, 'initial_student_sha256': start_sha,
    'teacher_sha256': SOURCE_SHA, 'data_config_sha256': sha(data),
    'data_audit_sha256': sha(data.with_suffix('.audit.json')),
    'preceding_selection_sha256': sha(exp / 'a2-selection.json'),
    'parent_development_comparison_sha256': sha(exp / 'comparison-a2-0650.json'),
    'parent_full_comparison_sha256': sha(exp / 'full-a2-0650/comparison.json'),
    'launcher_sha256': sha(launcher), 'command': args,
    'environment': {'NAME': name, 'STEPS': '300'}, 'declared_export_steps': [100, 200, 300],
    'rationale': 'A2-0650 passes every development guard, the larger multilingual mean and '
                 'every larger individual guard. English macro WER alone remains 0.012449 '
                 'points above the original. Test a short English-emphasized continuation: '
                 '80% original mixture plus 20% original English CV/FLEURS group, preserving '
                 'all training membership. Retain R11/R12 teacher coefficients, halve every '
                 'peak learning rate, and use a fresh 300-step cosine schedule with 10 warmup '
                 'steps and seed 20260910. All 651 remaining parameter tensors train; exact '
                 'Gabor functions and BN running statistics stay fixed. This is adaptive '
                 'recovery on exposed regressions, not independent validation.',
    'selection_rule': 'Audit and evaluate all three declared exports. Run development-passing '
                      'exports on the larger regression in increasing development-WER order, '
                      'stopping at the first full pass. If none passes development, evaluate '
                      'the lowest-development-WER export once for diagnosis. Record all results '
                      'and explicitly name larger comparisons not run.',
    'no_gate_changes': True, 'native_qualification_required_separately': True}
with (exp / 'r13-start-decision.json').open('x') as file:
    file.write(json.dumps(decision, indent=2) + '\n')


def run(arguments, logfile, environment=None):
    with (exp / logfile).open('x') as log:
        subprocess.run(arguments, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True,
                       env=dict(os.environ, **environment) if environment else None)


run(args, 'train-r13.log', {'NAME': name, 'STEPS': '300'})
run(['python3', str(exp / 'evaluate_run.py'), '--run', name, '--label', 'r13',
     '--steps', '100', '200', '300'], 'evaluate-r13.log')
state = json.loads((exp / 'campaign-status.json').read_text())
rows = state['results']
assert state['phase'] == 'development-complete'
assert {row['label'] for row in rows} == {'r13-0100', 'r13-0200', 'r13-0300'}
passing = sorted((row for row in rows if row['status'] == 'pass'), key=lambda row: row['primary']['candidate_wer'])
order = passing or [min(rows, key=lambda row: row['primary']['candidate_wer'])]
selection = {'rule': decision['selection_rule'], 'candidates': rows,
             'larger_order': [row['label'] for row in order], 'diagnostic_only': not passing,
             'larger_results': [], 'source_qualified_selection': None}
selection_path = exp / 'r13-full-selection.json'
with selection_path.open('x') as file:
    file.write(json.dumps(selection, indent=2) + '\n')
for row in order:
    label = row['label']
    arguments = ['python3', str(exp / 'full_regression.py'), '--candidate', row['model'],
                 '--development-decision', str(exp / ('comparison-' + label + '.json')),
                 '--output', str(exp / ('full-' + label))]
    if not passing:
        arguments.append('--diagnostic')
    run(arguments, 'full-' + label + '.log')
    result = json.loads((exp / ('full-' + label) / 'comparison.json').read_text())
    selection['larger_results'].append({'label': label, 'source_qualified': result['release_qualified'],
                                        'comparison_sha256': sha(exp / ('full-' + label) / 'comparison.json')})
    if result['release_qualified']:
        selection['source_qualified_selection'] = row
    selection_path.write_text(json.dumps(selection, indent=2) + '\n')
    if result['release_qualified']:
        break
tested = {row['label'] for row in selection['larger_results']}
selection['larger_comparisons_not_run'] = [row['label'] for row in rows if row['label'] not in tested]
selection['native_qualified'] = False
selection_path.write_text(json.dumps(selection, indent=2) + '\n')
print('ENGLISH_RECOVERY_EVALUATION_COMPLETE', flush=True)
