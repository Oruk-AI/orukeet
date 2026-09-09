"""Bounded recovery with larger updates around the immutable convolution rows."""
import json
import os
from pathlib import Path
import shutil
import subprocess

from identity import sha, SOURCE_SHA

root = Path('/home/nathanroll/parakeet-ft')
exp = root / 'gabor_half_20260906'
name = 'orukeet_gabor_half_r10_20260906'
start = exp / 'orukeet_gabor_half_r9_20260906/step-0600.nemo'
start_sha = 'ae59dcdaa87c9cf476ae5ecd9986287c49f38ead8a92e53da7bb9fa286514f7e'
assert 'SLICE_RECOVERY_EVALUATION_COMPLETE' in (exp / 'slice-recovery-r9-orchestrator.log').read_text()
assert sha(start) == start_sha
comparison = json.loads((exp / 'full-r9-0600/comparison.json').read_text())
assert comparison['release_qualified'] is False
assert comparison['candidate_model_sha256'] == start_sha
selection = json.loads((exp / 'r9-full-selection.json').read_text())
assert selection['selected']['label'] == 'r9-0600'
data = root / 'manifests/balanced_global_20260905/input_cfg_with_accents.yaml'
previous = json.loads((exp / 'slice-recovery-input-r9.audit.json').read_text())
assert sha(data) == previous['source_config_sha256']
assert previous['exact_evaluation_audio_path_overlap'] == 0
assert previous['evaluation_paths_checked'] == 28138
for path, record in previous['training_manifests'].items():
    assert sha(path) == record['sha256']
data_audit = {
    'config_sha256': sha(data), 'membership_changed': False,
    'exact_evaluation_audio_path_overlap': 0, 'evaluation_paths_checked': 28138,
    'training_manifests': previous['training_manifests'],
    'inherited_path_check_sha256': sha(exp / 'slice-recovery-input-r9.audit.json'),
    'mixture': '100% original balanced multilingual and accent mixture',
    'interpretation': 'Exact original configuration and all manifests rehashed against the prior '
                      'complete path-overlap audit. This does not establish acoustic or speaker independence.'}
data_audit_path = exp / 'r10-data-audit.json'
if data_audit_path.exists():
    assert json.loads(data_audit_path.read_text()) == data_audit
else:
    with data_audit_path.open('x') as file:
        file.write(json.dumps(data_audit, indent=2) + '\n')
assert shutil.disk_usage(exp).free >= 25_000_000_000
launcher = exp / (name + '-launch.sh')
with launcher.open('xb') as file:
    file.write((exp / 'run_recovery.sh').read_bytes())
args = ['bash', str(launcher), '++export_interval=200', '++seed=20260908',
        '++model.train_ds.seed=20260908', '++model.train_ds.shard_seed=20260908',
        '++recovery_start_model=' + str(start),
        '++recovery_start_audit=' + str(exp / 'r9-0600-audit.json'),
        '++model.train_ds.input_cfg=' + str(data),
        'model.optim.lr=2e-6', 'model.optim.sched.min_lr=0.0',
        'model.optim.sched.warmup_steps=25', '++preserve_bn_statistics=true',
        '++disable_dropout=true', '++resume_keep=1',
        '++layer_anchor_scale=5.0', '++conv_anchor_scale=5.0', '++asr_loss_scale=1.0',
        '++recovery_layer_lrs.lower_encoder=1e-6', '++recovery_layer_lrs.upper_encoder=2e-6',
        '++recovery_layer_lrs.conv_neighbors=2e-5',
        '++recovery_layer_lrs.other=1e-6', '++recovery_layer_lrs.decoder=1e-7',
        '++recovery_layer_lrs.joint=1e-7',
        'model.spec_augment.freq_masks=0', 'model.spec_augment.time_masks=0']
decision = {
    'label': 'r10', 'source_sha256': SOURCE_SHA, 'initial_student_sha256': start_sha,
    'teacher_sha256': SOURCE_SHA, 'data_config_sha256': sha(data),
    'data_audit_sha256': sha(exp / 'r10-data-audit.json'),
    'preceding_full_comparison_sha256': sha(exp / 'full-r9-0600/comparison.json'),
    'launcher_sha256': sha(launcher), 'command': args,
    'environment': {'NAME': name, 'STEPS': '400'}, 'declared_export_steps': [200, 400],
    'rationale': 'Continue the lowest-development-WER R9 export. R9 passes larger per-language '
                 'and English guards but misses the multilingual mean and one development slice. '
                 'Return to the complete original sampling mixture. Give the 120 trainable '
                 'convolution tensors (pointwise projections, free depthwise rows, BatchNorm '
                 'affine terms) a 2e-5 learning rate and increase branch reconstruction scale '
                 'from 1 to 5. All other 531 parameter tensors retain positive learning rates '
                 'and ASR gradients. This tests compensation near the fixed kernels, without '
                 'editing those kernels or the qualification criteria.',
    'selection_rule': 'Evaluate both declared exports; lowest development WER among passing '
                      'exports, otherwise lowest for diagnosis only, then unchanged larger regression.',
    'no_gate_changes': True}
with (exp / 'r10-start-decision.json').open('x') as file:
    file.write(json.dumps(decision, indent=2) + '\n')
with (exp / 'train-r10.log').open('x') as log:
    subprocess.run(args, cwd=root, env=dict(os.environ, NAME=name, STEPS='400'),
                   stdout=log, stderr=subprocess.STDOUT, check=True)
with (exp / 'evaluate-r10.log').open('x') as log:
    subprocess.run(['python3', str(exp / 'evaluate_run.py'), '--run', name,
                    '--label', 'r10', '--steps', '200', '400'], cwd=root,
                   stdout=log, stderr=subprocess.STDOUT, check=True)
with (exp / 'full-r10.log').open('x') as log:
    subprocess.run(['python3', str(exp / 'evaluate_selected_recovery.py'), '--label', 'r10'],
                   cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
print('CONV_RECOVERY_EVALUATION_COMPLETE', flush=True)
