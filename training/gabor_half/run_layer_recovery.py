"""Resolve the recorded starting rule, then run bounded layerwise recovery."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from identity import sha

root = Path('/home/nathanroll/parakeet-ft')
exp = root / 'gabor_half_20260906'
deadline = time.monotonic() + 7200
while True:
    status = exp / 'full-r5-status.json'
    results = json.loads(status.read_text())['results'] if status.exists() else []
    passing = next((r for r in results if r['status'] == 'pass'), None)
    if passing or {r['label'] for r in results} == {'r5-0300', 'r5-0200', 'r5-0100'}:
        break
    log = exp / 'evaluate-full-r5.log'
    if log.exists() and 'Traceback (most recent call last)' in log.read_text(errors='replace'):
        raise RuntimeError('R5 larger evaluation failed to complete')
    if time.monotonic() > deadline:
        raise RuntimeError('R5 evaluation timeout')
    time.sleep(10)

if shutil.disk_usage(exp).free < 30_000_000_000:
    raise RuntimeError('Need 30 GB free for materialized exports and optimizer snapshots')
name = 'orukeet_gabor_half_r6_20260906'
args = ['bash', str(exp / 'run_recovery.sh'),
        '++export_interval=200', 'model.optim.lr=3e-6',
        'model.optim.sched.min_lr=0.0', 'model.optim.sched.warmup_steps=10',
        '++preserve_bn_statistics=true', '++disable_dropout=true', '++resume_keep=1',
        '++layer_anchor_scale=5.0', '++conv_anchor_scale=1.0', '++asr_loss_scale=0.1',
        '++recovery_layer_lrs.lower_encoder=3e-6', '++recovery_layer_lrs.upper_encoder=3e-6',
        '++recovery_layer_lrs.other=3e-6', '++recovery_layer_lrs.decoder=1e-7',
        '++recovery_layer_lrs.joint=1e-7',
        'model.spec_augment.freq_masks=0', 'model.spec_augment.time_masks=0']
decision = {'plan_sha256': sha(exp / 'layer-recovery-plan.json'),
            'r5_larger_results': results, 'starting_checkpoint': None,
            'note': 'A larger-suite pass does not erase a development failure.'}
if passing:
    label = passing['label']
    step = label.split('-')[1]
    model = exp / 'orukeet_gabor_half_r5_20260906' / f'step-{step}.nemo'
    args += ['++recovery_start_model=' + str(model),
             '++recovery_start_audit=' + str(exp / (label + '-audit.json')),
             '++model.train_ds.input_cfg=' + str(exp / 'english-recovery-input.json')]
    decision['starting_checkpoint'] = {'label': label, 'sha256': sha(model)}
else:
    decision['starting_checkpoint'] = {'label': 'original',
        'sha256': sha(root / 'models/ft/stage3_baseblend_a075_20260905.nemo')}
decision['command'] = args
decision['environment'] = {'NAME': name, 'STEPS': '600'}
# Bash can read more of its script after the training child exits. A stable
# per-run copy prevents edits to the shared launcher from changing that tail.
launcher = exp / (name + '-launch.sh')
with launcher.open('xb') as f:
    f.write((exp / 'run_recovery.sh').read_bytes())
args[1] = str(launcher)
decision['launcher_sha256'] = sha(launcher)
with (exp / 'r6-start-decision.json').open('x') as f:
    f.write(json.dumps(decision, indent=2) + '\n')
with (exp / 'train-r6.log').open('x') as log:
    subprocess.run(args, cwd=root, env=dict(os.environ, NAME=name, STEPS='600'),
                   stdout=log, stderr=subprocess.STDOUT, check=True)
with (exp / 'evaluate-r6.log').open('x') as log:
    subprocess.run(['python3', str(exp / 'evaluate_run.py'), '--run', name,
                    '--label', 'r6', '--steps', '600', '400', '200'],
                   cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
print('LAYER_RECOVERY_DEVELOPMENT_COMPLETE', flush=True)
