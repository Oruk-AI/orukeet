"""Continue the audited R7 candidate on the recorded R8 language curriculum."""
import json
import os
from pathlib import Path
import shutil
import subprocess

from identity import sha, SOURCE_SHA

root = Path('/home/nathanroll/parakeet-ft')
exp = root / 'gabor_half_20260906'
name = 'orukeet_gabor_half_r8_20260906'
start = exp / 'orukeet_gabor_half_r7_20260906/step-0300.nemo'
start_sha = '992a35cdfcea8d4d6ca57e8bef4d2fa2ae82a8e1ab02a872598e338faea40ef3'
data = exp / 'language-recovery-input-r8.json'
data_audit = json.loads(data.with_suffix('.audit.json').read_text())
assert sha(start) == start_sha
assert data_audit['candidate_sha256'] == start_sha
assert data_audit['config_sha256'] == sha(data)
assert data_audit['target_languages'] == ['bg', 'lv']
assert data_audit['exact_evaluation_audio_path_overlap'] == 0
assert not data_audit['membership_changed']
if shutil.disk_usage(exp).free < 30_000_000_000:
    raise RuntimeError('Need 30 GB free for exports and optimizer snapshots')

launcher = exp / (name + '-launch.sh')
with launcher.open('xb') as f:
    f.write((exp / 'run_recovery.sh').read_bytes())
args = ['bash', str(launcher), '++export_interval=100',
        '++recovery_start_model=' + str(start),
        '++recovery_start_audit=' + str(exp / 'r7-0300-audit.json'),
        '++model.train_ds.input_cfg=' + str(data),
        'model.optim.lr=2e-6', 'model.optim.sched.min_lr=0.0',
        'model.optim.sched.warmup_steps=10', '++preserve_bn_statistics=true',
        '++disable_dropout=true', '++resume_keep=1',
        '++layer_anchor_scale=5.0', '++conv_anchor_scale=1.0',
        '++asr_loss_scale=1.0', '++recovery_layer_lrs.lower_encoder=1e-6',
        '++recovery_layer_lrs.upper_encoder=2e-6',
        '++recovery_layer_lrs.other=1e-6',
        '++recovery_layer_lrs.decoder=1e-7', '++recovery_layer_lrs.joint=1e-7',
        'model.spec_augment.freq_masks=0', 'model.spec_augment.time_masks=0']
decision = {'label': 'r8', 'source_sha256': SOURCE_SHA,
    'initial_student_sha256': start_sha, 'teacher_sha256': SOURCE_SHA,
    'data_config_sha256': sha(data), 'data_audit_sha256': sha(data.with_suffix('.audit.json')),
    'launcher_sha256': sha(launcher), 'command': args,
    'environment': {'NAME': name, 'STEPS': '400'},
    'rationale': 'R7-0300 improves development mean by 0.112674 WER points and larger English mean by 0.060395, but misses larger primary mean by 0.020564 and development Bulgarian/Latvian guards. Continue from its audited export with equal emphasis on the two remaining training languages. Increase supervised weight and lower-layer/head learning rates to allow compensation outside the fixed Gabor rows. Keep all evaluation criteria unchanged.',
    'selection_rule': 'Evaluate all four exports on unchanged development gates. Lowest development WER among passing exports, otherwise lowest for diagnosis only; then the unchanged larger regression. No promotion on a diagnostic result.',
    'no_gate_changes': True}
with (exp / 'r8-start-decision.json').open('x') as f:
    f.write(json.dumps(decision, indent=2) + '\n')
with (exp / 'train-r8.log').open('x') as log:
    subprocess.run(args, cwd=root, env=dict(os.environ, NAME=name, STEPS='400'),
                   stdout=log, stderr=subprocess.STDOUT, check=True)
with (exp / 'evaluate-r8.log').open('x') as log:
    subprocess.run(['python3', str(exp / 'evaluate_run.py'), '--run', name,
                    '--label', 'r8', '--steps', '100', '200', '300', '400'],
                   cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
with (exp / 'full-r8.log').open('x') as log:
    subprocess.run(['python3', str(exp / 'evaluate_selected_recovery.py'), '--label', 'r8'],
                   cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
print('LANGUAGE_RECOVERY_EVALUATION_COMPLETE', flush=True)
