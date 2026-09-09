"""One source qualification of an existing export after R15-100 native failures."""
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess

from identity import sha, SOURCE_SHA

root = Path('/home/nathanroll/parakeet-ft')
exp = root / 'gabor_half_20260906'
label = 'r15-0200'
model = exp / 'orukeet_gabor_half_r15_20260906/step-0200.nemo'
audit_path = exp / (label + '-audit.json')
development_path = exp / ('comparison-' + label + '.json')
audit = json.loads(audit_path.read_text())
development = json.loads(development_path.read_text())
assert audit['status'] == 'pass' and audit['frozen_gabor_rows_exact'] == 12288
assert audit['original_sha256'] == SOURCE_SHA
assert sha(model) == audit['candidate_sha256'] == development['candidate_sha256'] == (
    'f5111f6132144b5993edbd4e228df9f731cac64d54777416c77dc85e96b58b61')
assert development['status'] == 'pass'
prior_source = exp / 'full-r15-0100/comparison.json'
prior_q8 = exp / 'native-r15-0100/development/comparison.json'
prior_f16 = exp / 'native-r15-0100-f16/larger/comparison.json'
assert json.loads(prior_source.read_text())['release_qualified'] is True
assert json.loads(prior_q8.read_text())['status'] == 'fail'
assert json.loads(prior_f16.read_text())['native_accuracy_qualified'] is False
assert 'NATIVE_LARGER_FAILED' in (exp / 'native-r15-0100-f16-orchestrator.log').read_text()
output = exp / ('full-' + label)
assert not output.exists()
decision = {
    'created_at': datetime.now(timezone.utc).isoformat(), 'label': label,
    'model_sha256': sha(model), 'audit_sha256': sha(audit_path),
    'development_sha256': sha(development_path), 'new_training': False,
    'prior_source_result_sha256': sha(prior_source),
    'prior_q8_failure_sha256': sha(prior_q8), 'prior_f16_failure_sha256': sha(prior_f16),
    'original_r15_selection_sha256': sha(exp / 'r15-full-selection.json'),
    'driver_sha256': sha(__file__), 'full_regression_sha256': sha(exp / 'full_regression.py'),
    'rationale': 'R15-100 passed source accuracy but its Q8 and F16 exports failed separate native gates. '
                 'Evaluate the only other already-trained R15 export, step 200, once on the unchanged '
                 'larger source regression. This is a new adaptive fallback after native results, '
                 'not part of the original stop-at-first-source-pass selection. Preserve that original record.',
    'stopping_rule': 'If source accuracy fails, stop this fallback. If it passes, native Q8 qualification '
                     'may proceed under the same accuracy gates. No training, fitted rows, '
                     'evaluation membership, decoder, thresholds or defaults change.',
}
with (exp / 'r15-0200-native-fallback-decision.json').open('x') as file:
    file.write(json.dumps(decision, indent=2) + '\n')
with (exp / 'full-r15-0200.log').open('x') as log:
    subprocess.run(['python3', str(exp / 'full_regression.py'), '--candidate', str(model),
                    '--development-decision', str(development_path), '--output', str(output)],
                   cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
result = json.loads((output / 'comparison.json').read_text())
print('EXISTING_R15_0200_SOURCE_' + ('PASSED' if result['release_qualified'] else 'FAILED'), flush=True)
