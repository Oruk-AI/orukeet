"""Continue shared-teacher recovery with modest English sampling emphasis."""
import json
import os
from pathlib import Path
import shutil
import subprocess

from identity import sha, SOURCE_SHA

root = Path('/home/nathanroll/parakeet-ft')
exp = root / 'gabor_half_20260906'
name = 'orukeet_gabor_half_r15_20260906'
assert 'AVERAGE_RECOVERY_EVALUATION_COMPLETE' in (exp / 'average-recovery-a3-orchestrator.log').read_text()
assert json.loads((exp / 'a3-selection.json').read_text())['source_qualified_selection'] is None
assert json.loads((exp / 'full-a3-0150/comparison.json').read_text())['failures'] == ['English macro WER exceeds original']
tests = json.loads((exp / 'shared-teacher-tests.json').read_text())
assert tests['status'] == 'pass' and tests['tests'] == 4
for filename, digest in tests['source_sha256'].items():
    assert sha(exp / filename) == digest
start = exp / 'orukeet_gabor_half_r14_20260906/step-0200.nemo'
start_sha = '63d462cf943d07785b382db97f179aaf8d9d4b7fe48672bcccdf8cf4e2617a47'
audit = json.loads((exp / 'r14-0200-audit.json').read_text())
assert sha(start) == start_sha == audit['candidate_sha256']
assert audit['status'] == 'pass' and audit['frozen_gabor_rows_exact'] == 12288
assert json.loads((exp / 'comparison-r14-0200.json').read_text())['status'] == 'pass'
assert json.loads((exp / 'full-r14-0200/comparison.json').read_text())['failures'] == ['English macro WER exceeds original']
data = exp / 'english-recovery-input-r13.json'
previous = json.loads((exp / 'english-recovery-input-r13.audit.json').read_text())
assert sha(data) == previous['config_sha256']
assert previous['top_level_weights'] == {'original_mixture': .8, 'english_cv_fleurs': .2}
assert previous['exact_evaluation_audio_path_overlap'] == 0 and previous['evaluation_paths_checked'] == 28138
for path, record in previous['training_manifests'].items():
    assert sha(path) == record['sha256']
data_audit = {
    'config_sha256': sha(data), 'membership_changed': False,
    'exact_evaluation_audio_path_overlap': 0, 'evaluation_paths_checked': 28138,
    'training_manifests': previous['training_manifests'],
    'inherited_path_check_sha256': sha(exp / 'english-recovery-input-r13.audit.json'),
    'mixture': '80% original balanced multilingual/accent mixture + 20% original English CV/FLEURS',
    'interpretation': 'All original training manifest hashes reverified against the prior path audit. '
                      'Exact path exclusion does not establish acoustic or speaker independence.'}
with (exp / 'r15-data-audit.json').open('x') as file:
    file.write(json.dumps(data_audit, indent=2) + '\n')
assert shutil.disk_usage(exp).free >= 22_000_000_000
launcher = exp / (name + '-launch.sh')
with launcher.open('xb') as file:
    file.write((exp / 'run_recovery.sh').read_bytes())
args = ['bash', str(launcher), '++export_interval=100', '++seed=20260912',
        '++model.train_ds.seed=20260912', '++model.train_ds.shard_seed=20260912',
        '++recovery_start_model=' + str(start), '++recovery_start_audit=' + str(exp / 'r14-0200-audit.json'),
        '++model.train_ds.input_cfg=' + str(data), 'model.optim.lr=5e-7',
        'model.optim.sched.min_lr=0.0', 'model.optim.sched.warmup_steps=10',
        '++preserve_bn_statistics=true', '++disable_dropout=true', '++resume_keep=1',
        '++layer_anchor_scale=50.0', '++conv_anchor_scale=25.0', '++asr_loss_scale=0.1',
        '++layer_posterior_scale=1.0', '++posterior_points=16',
        '++recovery_layer_lrs.lower_encoder=2.5e-7', '++recovery_layer_lrs.upper_encoder=5e-7',
        '++recovery_layer_lrs.conv_neighbors=5e-6', '++recovery_layer_lrs.other=2.5e-7',
        '++recovery_layer_lrs.decoder=1e-7', '++recovery_layer_lrs.joint=1e-7',
        'model.spec_augment.freq_masks=0', 'model.spec_augment.time_masks=0']
decision = {
    'label': 'r15', 'source_sha256': SOURCE_SHA, 'initial_student_sha256': start_sha,
    'teacher_sha256': SOURCE_SHA, 'data_config_sha256': sha(data),
    'data_audit_sha256': sha(exp / 'r15-data-audit.json'),
    'preceding_selection_sha256': sha(exp / 'a3-selection.json'),
    'preceding_full_comparison_sha256': sha(exp / 'full-a3-0150/comparison.json'),
    'parent_full_comparison_sha256': sha(exp / 'full-r14-0200/comparison.json'),
    'unit_tests_sha256': sha(exp / 'shared-teacher-tests.json'),
    'launcher_sha256': sha(launcher), 'driver_sha256': sha(exp / 'run_shared_english_r15.py'), 'command': args,
    'environment': {'NAME': name, 'STEPS': '200'}, 'declared_export_steps': [100, 200],
    'rationale': 'A3 failed development and its larger diagnostic did not improve the R14 English '
                 'mean. Return to R14-200, which passes development and the larger multilingual '
                 'mean. Common Voice English is its largest positive English-corpus delta. '
                 'Reuse the already audited 80% original plus 20% English CV/FLEURS mixture, '
                 'retaining every training member and the original shared-teacher objective. '
                 'Run 200 steps with encoder/conv rates half R14, unchanged decoder/joint rates, '
                 'a fresh cosine schedule, ten warmup steps and seed 20260912. The loss remains '
                 '0.1 ASR + 50 block + 25 branch + 1 token KL + 1 duration KL, sampled at 16 '
                 'valid time/prefix positions. Every remaining student parameter trains. All '
                 '12,288 Gabor rows, teacher parameters and BN statistics stay fixed. This is '
                 'adaptive recovery on exposed regressions, not independent generalization.',
    'selection_rule': 'Audit both declared exports. Run development-passing exports on the '
                      'larger regression in increasing development-WER order, stopping at the '
                      'first full pass. If neither passes development, evaluate the lowest mean '
                      'once for diagnosis. Preserve every result and name unrun larger comparisons.',
    'no_gate_changes': True, 'native_qualification_required_separately': True}
with (exp / 'r15-start-decision.json').open('x') as file:
    file.write(json.dumps(decision, indent=2) + '\n')


def run(arguments, logfile, environment=None):
    with (exp / logfile).open('x') as log:
        subprocess.run(arguments, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True,
                       env=dict(os.environ, **environment) if environment else None)


run(args, 'train-r15.log', {'NAME': name, 'STEPS': '200'})
run(['python3', str(exp / 'evaluate_run.py'), '--run', name, '--label', 'r15', '--steps', '100', '200'],
    'evaluate-r15.log')
state = json.loads((exp / 'campaign-status.json').read_text())
rows = state['results']
assert state['phase'] == 'development-complete'
assert {row['label'] for row in rows} == {'r15-0100', 'r15-0200'}
passing = sorted((row for row in rows if row['status'] == 'pass'), key=lambda row: row['primary']['candidate_wer'])
order = passing or [min(rows, key=lambda row: row['primary']['candidate_wer'])]
selection = {'rule': decision['selection_rule'], 'candidates': rows,
             'larger_order': [row['label'] for row in order], 'diagnostic_only': not passing,
             'larger_results': [], 'source_qualified_selection': None}
selection_path = exp / 'r15-full-selection.json'
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
print('SHARED_TEACHER_EVALUATION_COMPLETE', flush=True)
