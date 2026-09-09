"""Select by development WER, then run one larger matched regression."""
import argparse
import json
from pathlib import Path
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('--label', required=True)
a = p.parse_args()
root = Path('/home/nathanroll/parakeet-ft')
exp = root / 'gabor_half_20260906'
deadline = time.monotonic() + 7200
while True:
    state = json.loads((exp / 'campaign-status.json').read_text())
    rows = state.get('results', [])
    if state['phase'] == 'development-complete' and rows and all(r['label'].startswith(a.label + '-') for r in rows):
        break
    for name in ('train-' + a.label + '.log', 'evaluate-' + a.label + '.log'):
        log = exp / name
        if log.exists() and 'Traceback (most recent call last)' in log.read_text(errors='replace'):
            raise RuntimeError('Recovery or evaluation failed: ' + name)
    if time.monotonic() > deadline:
        raise RuntimeError('Recovery evaluation timeout')
    time.sleep(10)
qualified = [r for r in rows if r['status'] == 'pass']
selected = min(qualified or rows, key=lambda r: r['primary']['candidate_wer'])
selection = {'rule': 'Lowest development primary WER among development-qualified exports; if none qualify, lowest primary WER for diagnosis only.',
             'candidates': rows, 'selected': selected, 'diagnostic_only': not qualified}
with (exp / (a.label + '-full-selection.json')).open('x') as f:
    f.write(json.dumps(selection, indent=2) + '\n')
args = ['python3', str(exp / 'full_regression.py'), '--candidate', selected['model'],
        '--development-decision', str(exp / ('comparison-' + selected['label'] + '.json')),
        '--output', str(exp / ('full-' + selected['label']))]
if not qualified:
    args.append('--diagnostic')
subprocess.run(args, cwd=root, check=True)
print('SELECTED_RECOVERY_FULL_COMPLETE', flush=True)
