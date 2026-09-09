"""Continue the audited R6 candidate on the recorded R7 language curriculum."""
import json
import os
from pathlib import Path
import shutil
import subprocess

from identity import sha, SOURCE_SHA

root = Path('/home/nathanroll/parakeet-ft')
exp = root / 'gabor_half_20260906'
name = 'orukeet_gabor_half_r7_20260906'
start = exp / 'orukeet_gabor_half_r6_20260906/step-0400.nemo'
start_sha = '8ba6d6ba01fa90798e5e4e4b4605d89d1b84f98f20e824a621999184a1f0cd3f'
data = exp / 'language-recovery-input-r7.json'
data_audit = json.loads(data.with_suffix('.audit.json').read_text())
assert sha(start) == start_sha
assert data_audit['candidate_sha256'] == start_sha
assert data_audit['config_sha256'] == sha(data)
assert data_audit['target_languages'] == ['bg', 'et', 'hu', 'uk']
assert data_audit['exact_evaluation_audio_path_overlap'] == 0
assert not data_audit['membership_changed']
if shutil.disk_usage(exp).free < 30_000_000_000:
    raise RuntimeError('Need 30 GB free for exports and optimizer snapshots')

launcher = exp / (name + '-launch.sh')
with launcher.open('xb') as f:
    f.write((exp / 'run_recovery.sh').read_bytes())
args = ['bash', str(launcher), '++export_interval=100',
        '++recovery_start_model=' + str(start),
        '++recovery_start_audit=' + str(exp / 'r6-0400-audit.json'),
        '++model.train_ds.input_cfg=' + str(data),
        'model.optim.lr=2e-6', 'model.optim.sched.min_lr=0.0',
        'model.optim.sched.warmup_steps=10', '++preserve_bn_statistics=true',
        '++disable_dropout=true', '++resume_keep=1',
        '++layer_anchor_scale=5.0', '++conv_anchor_scale=1.0',
        '++asr_loss_scale=0.5', '++recovery_layer_lrs.lower_encoder=2e-7',
        '++recovery_layer_lrs.upper_encoder=2e-6',
        '++recovery_layer_lrs.other=2e-7',
        '++recovery_layer_lrs.decoder=5e-8', '++recovery_layer_lrs.joint=5e-8',
        'model.spec_augment.freq_masks=0', 'model.spec_augment.time_masks=0']
decision = {'label': 'r7', 'source_sha256': SOURCE_SHA,
    'initial_student_sha256': start_sha, 'teacher_sha256': SOURCE_SHA,
    'data_config_sha256': sha(data), 'data_audit_sha256': sha(data.with_suffix('.audit.json')),
    'launcher_sha256': sha(launcher), 'command': args,
    'environment': {'NAME': name, 'STEPS': '300'},
    'rationale': 'R6-0400 recovered the larger-suite English mean and all individual larger guards, but missed the primary mean by 0.029956 WER points and four development guards. Continue the same fixed Gabor model with more supervised weight, targeted existing training languages, and small head updates.',
    'selection_rule': 'Evaluate all three exports on unchanged development gates. Lowest development WER among passing exports, otherwise lowest for diagnosis only; then the unchanged larger regression. No promotion on a diagnostic result.',
    'no_gate_changes': True}
with (exp / 'r7-start-decision.json').open('x') as f:
    f.write(json.dumps(decision, indent=2) + '\n')
with (exp / 'train-r7.log').open('x') as log:
    subprocess.run(args, cwd=root, env=dict(os.environ, NAME=name, STEPS='300'),
                   stdout=log, stderr=subprocess.STDOUT, check=True)
with (exp / 'evaluate-r7.log').open('x') as log:
    subprocess.run(['python3', str(exp / 'evaluate_run.py'), '--run', name,
                    '--label', 'r7', '--steps', '100', '200', '300'],
                   cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
with (exp / 'full-r7.log').open('x') as log:
    subprocess.run(['python3', str(exp / 'evaluate_selected_recovery.py'), '--label', 'r7'],
                   cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
print('LANGUAGE_RECOVERY_EVALUATION_COMPLETE', flush=True)
