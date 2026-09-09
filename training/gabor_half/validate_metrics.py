"""Verify accelerated edit distances against every saved legacy metric."""
import argparse
import importlib.util
import json
from pathlib import Path
import random
import time

import rapidfuzz
from rapidfuzz.distance import Levenshtein
from identity import sha

p = argparse.ArgumentParser()
p.add_argument('--evaluator', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('directories', nargs='+', type=Path)
a = p.parse_args()
spec = importlib.util.spec_from_file_location('legacy', a.evaluator)
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
rng = random.Random(20260906)
alphabet = ['a', 'é', 'λ', 'я', '我', '🙂', '\u0301', ' ']
for _ in range(1000):
    left, right = (''.join(rng.choices(alphabet, k=rng.randrange(40))) for _ in range(2))
    assert Levenshtein.distance(left, right) == legacy.distance(left, right)
    assert Levenshtein.distance(left.split(), right.split()) == legacy.distance(left.split(), right.split())
started = time.monotonic()
checks = []
for directory in a.directories:
    metadata = json.loads((directory / 'results.json').read_text())
    assert metadata['normalizer'] == 'legacy-compatible-nfc-v1'
    expected = sum(v['slices']['all']['rows'] for v in metadata['sets'].values())
    rows = 0
    files = {}
    for path in sorted(directory.glob('*_hypotheses.jsonl')):
        files[path.name] = sha(path)
        for line in path.open():
            row = json.loads(line)
            reference, hypothesis = legacy.normalize(row['text']), legacy.normalize(row['pred_text'])
            assert Levenshtein.distance(reference.split(), hypothesis.split()) == row['errors']
            assert Levenshtein.distance(reference, hypothesis) == row['char_errors']
            assert len(reference.split()) == row['words'] and len(reference) == row['chars']
            rows += 1
    assert rows == expected
    checks.append({'directory': str(directory), 'model_sha256': metadata['model_sha256'],
                   'rows': rows, 'prediction_files_sha256': files})
receipt = {'status': 'pass', 'legacy_evaluator_sha256': sha(a.evaluator),
           'rapidfuzz_version': rapidfuzz.__version__, 'random_unicode_cases': 1000,
           'verified_saved_predictions': sum(c['rows'] for c in checks),
           'saved_prediction_verification_seconds': time.monotonic() - started,
           'normalization': 'Unmodified legacy-compatible-nfc-v1',
           'change': 'Only integer edit-distance computation; unit costs, no cutoff or preprocessing.',
           'checks': checks}
a.output.write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({k: v for k, v in receipt.items() if k != 'checks'}), flush=True)
