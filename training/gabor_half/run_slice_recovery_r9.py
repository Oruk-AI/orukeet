"""Run a bounded source-specific continuation after R8's larger comparison."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from identity import sha, SOURCE_SHA

root = Path('/home/nathanroll/parakeet-ft')
exp = root / 'gabor_half_20260906'
name = 'orukeet_gabor_half_r9_20260906'
start = exp / 'orukeet_gabor_half_r8_20260906/step-0100.nemo'
start_sha = 'b22dea04beecf1eba46cef835501082d97c44a4e28e1eed356a650246abc3b4d'
data = exp / 'slice-recovery-input-r9.json'
audit = json.loads(data.with_suffix('.audit.json').read_text())
assert sha(start) == audit['candidate_sha256'] == start_sha
assert sha(data) == audit['config_sha256']
assert audit['target_source_languages'] == [['cv', 'bg'], ['cv', 'lv'], ['fleurs', 'fi']]
assert audit['exact_evaluation_audio_path_overlap'] == 0 and not audit['membership_changed']
deadline = time.monotonic() + 7200
while not ('LANGUAGE_RECOVERY_EVALUATION_COMPLETE' in
           (exp / 'language-recovery-r8-orchestrator.log').read_text(errors='replace')
           and (exp / 'r4-r5-private-archive.json').exists()):
    for log in [exp / 'full-r8.log', exp / 'language-recovery-r8-orchestrator.log']:
        if log.exists() and 'Traceback (most recent call last)' in log.read_text(errors='replace'):
            raise RuntimeError('R8 evaluation failed: ' + str(log))
    if time.monotonic() > deadline:
        raise RuntimeError('R8 full comparison timeout')
    time.sleep(15)
comparison = json.loads((exp / 'full-r8-0100/comparison.json').read_text())
assert comparison['release_qualified'] is False
with (exp / 'cleanup-r9.log').open('x') as log:
    subprocess.run([str(root / 'nemo.sh'), 'env', 'OPENBLAS_NUM_THREADS=1', 'OMP_NUM_THREADS=1',
                    'python', str(exp / 'retire_archived_r4_r5.py')], cwd=root, stdout=log,
                   stderr=subprocess.STDOUT, check=True,
                   env=dict(os.environ, NEMO_NAME='orukeet-retire-archived-r4-r5'))
assert shutil.disk_usage(exp).free >= 30_000_000_000
launcher = exp / (name + '-launch.sh')
with launcher.open('xb') as file:
    file.write((exp / 'run_recovery.sh').read_bytes())
args = ['bash', str(launcher), '++export_interval=200', '++seed=20260907',
        '++model.train_ds.seed=20260907', '++model.train_ds.shard_seed=20260907',
        '++recovery_start_model=' + str(start),
        '++recovery_start_audit=' + str(exp / 'r8-0100-audit.json'),
        '++model.train_ds.input_cfg=' + str(data),
        'model.optim.lr=2e-6', 'model.optim.sched.min_lr=0.0',
        'model.optim.sched.warmup_steps=25', '++preserve_bn_statistics=true',
        '++disable_dropout=true', '++resume_keep=1',
        '++layer_anchor_scale=5.0', '++conv_anchor_scale=1.0', '++asr_loss_scale=1.0',
        '++recovery_layer_lrs.lower_encoder=1e-6', '++recovery_layer_lrs.upper_encoder=2e-6',
        '++recovery_layer_lrs.other=1e-6', '++recovery_layer_lrs.decoder=1e-7',
        '++recovery_layer_lrs.joint=1e-7',
        'model.spec_augment.freq_masks=0', 'model.spec_augment.time_masks=0']
decision = {'label': 'r9', 'source_sha256': SOURCE_SHA, 'initial_student_sha256': start_sha,
            'teacher_sha256': SOURCE_SHA, 'data_config_sha256': sha(data),
            'data_audit_sha256': sha(data.with_suffix('.audit.json')),
            'preceding_full_comparison_sha256': sha(exp / 'full-r8-0100/comparison.json'),
            'launcher_sha256': sha(launcher), 'command': args,
            'environment': {'NAME': name, 'STEPS': '600'},
            'rationale': 'Continue the lowest-development-WER R8 export. Increase the original-mixture '
                         'share from 40% to 60%, reduce targeted emphasis from 40% to 20%, and target '
                         'the failing source/language pairs rather than whole languages. Change the '
                         'sampling seed for additional training coverage. Retain the R8 objective, '
                         'learning-rate groups and exact fixed rows. This is an adaptive recovery '
                         'hypothesis; all failures and unchanged qualification criteria remain recorded.',
            'selection_rule': 'Evaluate all three declared exports; lowest development WER among passing '
                              'exports, otherwise lowest for diagnosis only, followed by unchanged larger regression.',
            'no_gate_changes': True}
with (exp / 'r9-start-decision.json').open('x') as file:
    file.write(json.dumps(decision, indent=2) + '\n')
with (exp / 'train-r9.log').open('x') as log:
    subprocess.run(args, cwd=root, env=dict(os.environ, NAME=name, STEPS='600'),
                   stdout=log, stderr=subprocess.STDOUT, check=True)
with (exp / 'evaluate-r9.log').open('x') as log:
    subprocess.run(['python3', str(exp / 'evaluate_run.py'), '--run', name,
                    '--label', 'r9', '--steps', '200', '400', '600'], cwd=root,
                   stdout=log, stderr=subprocess.STDOUT, check=True)
with (exp / 'full-r9.log').open('x') as log:
    subprocess.run(['python3', str(exp / 'evaluate_selected_recovery.py'), '--label', 'r9'],
                   cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
print('SLICE_RECOVERY_EVALUATION_COMPLETE', flush=True)
