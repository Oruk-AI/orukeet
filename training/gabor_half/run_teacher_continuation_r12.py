"""Continue R11 only if all declared A1 averages fail source qualification."""
import json
import os
from pathlib import Path
import shutil
import subprocess

from identity import sha, SOURCE_SHA

root = Path('/home/nathanroll/parakeet-ft')
exp = root / 'gabor_half_20260906'
name = 'orukeet_gabor_half_r12_20260906'
assert 'AVERAGE_RECOVERY_EVALUATION_COMPLETE' in (exp / 'average-recovery-a1-orchestrator.log').read_text()
average_selection = json.loads((exp / 'a1-selection.json').read_text())
assert average_selection['source_qualified_selection'] is None
assert len(average_selection['development_candidates']) == 3
assert average_selection['larger_results']
start = exp / 'orukeet_gabor_half_r11_20260906/step-0400.nemo'
start_sha = 'ac0b8f43d67580bafd79614207bf012037147fb792b6a117f02bf0cfb85e92be'
assert sha(start) == start_sha
audit = json.loads((exp / 'r11-0400-audit.json').read_text())
assert audit['status'] == 'pass' and audit['candidate_sha256'] == start_sha
assert audit['frozen_gabor_rows_exact'] == 12288
comparison = json.loads((exp / 'full-r11-0400/comparison.json').read_text())
assert comparison['release_qualified'] is False
assert comparison['candidate_model_sha256'] == start_sha
assert comparison['failures'] == ['Primary WER exceeds original']
data = root / 'manifests/balanced_global_20260905/input_cfg_with_accents.yaml'
previous = json.loads((exp / 'r11-data-audit.json').read_text())
assert sha(data) == previous['config_sha256']
assert previous['exact_evaluation_audio_path_overlap'] == 0
assert previous['evaluation_paths_checked'] == 28138
for path, record in previous['training_manifests'].items():
    assert sha(path) == record['sha256']
data_audit = {
    'config_sha256': sha(data), 'membership_changed': False,
    'exact_evaluation_audio_path_overlap': 0, 'evaluation_paths_checked': 28138,
    'training_manifests': previous['training_manifests'],
    'inherited_path_check_sha256': sha(exp / 'r11-data-audit.json'),
    'mixture': '100% original balanced multilingual and accent mixture',
    'interpretation': 'Original configuration and all training manifests rehashed against the '
                      'previous path-overlap audit. Exact paths alone do not establish acoustic '
                      'or speaker independence.'}
assert shutil.disk_usage(exp).free >= 25_000_000_000
with (exp / 'r12-data-audit.json').open('x') as file:
    file.write(json.dumps(data_audit, indent=2) + '\n')
launcher = exp / (name + '-launch.sh')
with launcher.open('xb') as file:
    file.write((exp / 'run_recovery.sh').read_bytes())
args = ['bash', str(launcher), '++export_interval=400', '++seed=20260909',
        '++model.train_ds.seed=20260909', '++model.train_ds.shard_seed=20260909',
        '++recovery_start_model=' + str(start),
        '++recovery_start_audit=' + str(exp / 'r11-0400-audit.json'),
        '++model.train_ds.input_cfg=' + str(data),
        'model.optim.lr=2e-6', 'model.optim.sched.min_lr=0.0',
        'model.optim.sched.warmup_steps=25', '++preserve_bn_statistics=true',
        '++disable_dropout=true', '++resume_keep=1',
        '++layer_anchor_scale=50.0', '++conv_anchor_scale=25.0', '++asr_loss_scale=0.1',
        '++recovery_layer_lrs.lower_encoder=1e-6', '++recovery_layer_lrs.upper_encoder=2e-6',
        '++recovery_layer_lrs.conv_neighbors=2e-5',
        '++recovery_layer_lrs.other=1e-6', '++recovery_layer_lrs.decoder=1e-7',
        '++recovery_layer_lrs.joint=1e-7',
        'model.spec_augment.freq_masks=0', 'model.spec_augment.time_masks=0']
decision = {
    'label': 'r12', 'source_sha256': SOURCE_SHA, 'initial_student_sha256': start_sha,
    'teacher_sha256': SOURCE_SHA, 'data_config_sha256': sha(data),
    'data_audit_sha256': sha(exp / 'r12-data-audit.json'),
    'preceding_average_selection_sha256': sha(exp / 'a1-selection.json'),
    'parent_full_comparison_sha256': sha(exp / 'full-r11-0400/comparison.json'),
    'launcher_sha256': sha(launcher), 'command': args,
    'environment': {'NAME': name, 'STEPS': '800'}, 'declared_export_steps': [400, 800],
    'rationale': 'Continue the strongest teacher-matching run after A1 fails qualification. '
                 'R11 passed every larger individual guard and the English mean, but missed '
                 'the multilingual mean by 0.007812 points and one Finnish development error '
                 'beyond its guard. Test more optimization with the same objective, original '
                 'training mixture and per-group peak rates. Use a fresh optimizer, an 800-step '
                 'cosine schedule and seed 20260909. All remaining parameters train; the '
                 'same 12,288 fitted rows and BatchNorm statistics remain fixed. This is '
                 'adaptive recovery on exposed regressions, not independent validation.',
    'selection_rule': 'Evaluate both declared exports; lowest development WER among passing '
                      'exports, otherwise lowest for diagnosis only, then unchanged larger regression.',
    'no_gate_changes': True, 'native_qualification_required_separately': True}
with (exp / 'r12-start-decision.json').open('x') as file:
    file.write(json.dumps(decision, indent=2) + '\n')
with (exp / 'train-r12.log').open('x') as log:
    subprocess.run(args, cwd=root, env=dict(os.environ, NAME=name, STEPS='800'),
                   stdout=log, stderr=subprocess.STDOUT, check=True)
with (exp / 'evaluate-r12.log').open('x') as log:
    subprocess.run(['python3', str(exp / 'evaluate_run.py'), '--run', name,
                    '--label', 'r12', '--steps', '400', '800'], cwd=root,
                   stdout=log, stderr=subprocess.STDOUT, check=True)
with (exp / 'full-r12.log').open('x') as log:
    subprocess.run(['python3', str(exp / 'evaluate_selected_recovery.py'), '--label', 'r12'],
                   cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
print('TEACHER_CONTINUATION_EVALUATION_COMPLETE', flush=True)
