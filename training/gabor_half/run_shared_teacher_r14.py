"""Match original hidden states and TDT posteriors with one shared teacher pass."""
import json
import os
from pathlib import Path
import shutil
import subprocess

from identity import sha, SOURCE_SHA

root = Path('/home/nathanroll/parakeet-ft')
exp = root / 'gabor_half_20260906'
name = 'orukeet_gabor_half_r14_20260906'
assert 'ENGLISH_RECOVERY_EVALUATION_COMPLETE' in (exp / 'english-recovery-r13-orchestrator.log').read_text()
assert json.loads((exp / 'r13-full-selection.json').read_text())['source_qualified_selection'] is None
tests = json.loads((exp / 'shared-teacher-tests.json').read_text())
assert tests['status'] == 'pass' and tests['tests'] == 4
for filename, digest in tests['source_sha256'].items():
    assert sha(exp / filename) == digest
start = exp / 'orukeet_gabor_half_a2_20260906/a2-0650.nemo'
start_sha = '44059de88f9797dcd9a3386e1b267733937326c518b2d570468911b6f7da18db'
audit = json.loads((exp / 'a2-0650-audit.json').read_text())
assert sha(start) == start_sha == audit['candidate_sha256']
assert audit['status'] == 'pass' and audit['frozen_gabor_rows_exact'] == 12288
assert json.loads((exp / 'comparison-a2-0650.json').read_text())['status'] == 'pass'
assert json.loads((exp / 'full-a2-0650/comparison.json').read_text())['failures'] == ['English macro WER exceeds original']
data = root / 'manifests/balanced_global_20260905/input_cfg_with_accents.yaml'
previous = json.loads((exp / 'english-recovery-input-r13.audit.json').read_text())
assert sha(data) == previous['source_config_sha256']
assert previous['exact_evaluation_audio_path_overlap'] == 0 and previous['evaluation_paths_checked'] == 28138
for path, record in previous['training_manifests'].items():
    assert sha(path) == record['sha256']
data_audit = {
    'config_sha256': sha(data), 'membership_changed': False,
    'exact_evaluation_audio_path_overlap': 0, 'evaluation_paths_checked': 28138,
    'training_manifests': previous['training_manifests'],
    'inherited_path_check_sha256': sha(exp / 'english-recovery-input-r13.audit.json'),
    'mixture': '100% original balanced multilingual and accent mixture',
    'interpretation': 'All original training manifest hashes reverified against the prior path audit. '
                      'Exact path exclusion does not establish acoustic or speaker independence.'}
with (exp / 'r14-data-audit.json').open('x') as file:
    file.write(json.dumps(data_audit, indent=2) + '\n')
assert shutil.disk_usage(exp).free >= 22_000_000_000
launcher = exp / (name + '-launch.sh')
with launcher.open('xb') as file:
    file.write((exp / 'run_recovery.sh').read_bytes())
args = ['bash', str(launcher), '++export_interval=200', '++seed=20260911',
        '++model.train_ds.seed=20260911', '++model.train_ds.shard_seed=20260911',
        '++recovery_start_model=' + str(start), '++recovery_start_audit=' + str(exp / 'a2-0650-audit.json'),
        '++model.train_ds.input_cfg=' + str(data), 'model.optim.lr=1e-6',
        'model.optim.sched.min_lr=0.0', 'model.optim.sched.warmup_steps=10',
        '++preserve_bn_statistics=true', '++disable_dropout=true', '++resume_keep=1',
        '++layer_anchor_scale=50.0', '++conv_anchor_scale=25.0', '++asr_loss_scale=0.1',
        '++layer_posterior_scale=1.0', '++posterior_points=16',
        '++recovery_layer_lrs.lower_encoder=5e-7', '++recovery_layer_lrs.upper_encoder=1e-6',
        '++recovery_layer_lrs.conv_neighbors=1e-5', '++recovery_layer_lrs.other=5e-7',
        '++recovery_layer_lrs.decoder=1e-7', '++recovery_layer_lrs.joint=1e-7',
        'model.spec_augment.freq_masks=0', 'model.spec_augment.time_masks=0']
decision = {
    'label': 'r14', 'source_sha256': SOURCE_SHA, 'initial_student_sha256': start_sha,
    'teacher_sha256': SOURCE_SHA, 'data_config_sha256': sha(data),
    'data_audit_sha256': sha(exp / 'r14-data-audit.json'),
    'preceding_selection_sha256': sha(exp / 'r13-full-selection.json'),
    'preceding_full_comparison_sha256': sha(exp / 'full-r13-0300/comparison.json'),
    'parent_full_comparison_sha256': sha(exp / 'full-a2-0650/comparison.json'),
    'unit_tests_sha256': sha(exp / 'shared-teacher-tests.json'),
    'launcher_sha256': sha(launcher), 'command': args,
    'environment': {'NAME': name, 'STEPS': '400'}, 'declared_export_steps': [200, 400],
    'rationale': 'R13 passed development but missed both larger means. Return to A2-0650, which '
                 'passed development and the larger multilingual mean. Add token/duration '
                 'posterior matching to its strong block/branch teacher objective on the original '
                 'training mixture. Reuse one teacher encoder pass, keep one ASR term, sample '
                 '16 valid time and prefix positions, and normalize 8,193 token and five duration '
                 'outputs separately. Use R13 encoder rates and R12 decoder/joint rates, a fresh '
                 '400-step cosine schedule, 10 warmup steps and seed 20260911. The loss weights '
                 'are 0.1 ASR + 50 block + 25 branch + 1 token KL + 1 duration KL. The original '
                 'teacher is outside the student optimizer and exports. Every remaining student '
                 'parameter trains, and all 12,288 Gabor rows and BN statistics stay fixed. '
                 'This is adaptive recovery, not an independent generalization experiment.',
    'selection_rule': 'Audit both declared exports. Run development-passing exports on the '
                      'larger regression in increasing development-WER order, stopping at the '
                      'first full pass. If neither passes development, evaluate the lowest mean '
                      'once for diagnosis. Preserve every result and name unrun larger comparisons.',
    'no_gate_changes': True, 'native_qualification_required_separately': True}
with (exp / 'r14-start-decision.json').open('x') as file:
    file.write(json.dumps(decision, indent=2) + '\n')


def run(arguments, logfile, environment=None):
    with (exp / logfile).open('x') as log:
        subprocess.run(arguments, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True,
                       env=dict(os.environ, **environment) if environment else None)


run(args, 'train-r14.log', {'NAME': name, 'STEPS': '400'})
run(['python3', str(exp / 'evaluate_run.py'), '--run', name, '--label', 'r14', '--steps', '200', '400'],
    'evaluate-r14.log')
state = json.loads((exp / 'campaign-status.json').read_text())
rows = state['results']
assert state['phase'] == 'development-complete'
assert {row['label'] for row in rows} == {'r14-0200', 'r14-0400'}
passing = sorted((row for row in rows if row['status'] == 'pass'), key=lambda row: row['primary']['candidate_wer'])
order = passing or [min(rows, key=lambda row: row['primary']['candidate_wer'])]
selection = {'rule': decision['selection_rule'], 'candidates': rows,
             'larger_order': [row['label'] for row in order], 'diagnostic_only': not passing,
             'larger_results': [], 'source_qualified_selection': None}
selection_path = exp / 'r14-full-selection.json'
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
