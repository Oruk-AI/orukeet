"""Original NeMo evaluator with independently verified faster edit distance."""
import importlib.util
import json
from pathlib import Path
import sys

import rapidfuzz
from rapidfuzz.distance import Levenshtein
from identity import sha

exp = Path(__file__).resolve().parent
source = exp.parent / 'evaluate_20260905.py'
receipt_path = exp / 'metric-equivalence.json'
receipt = json.loads(receipt_path.read_text())
if (receipt['status'] != 'pass' or receipt['legacy_evaluator_sha256'] != sha(source)
        or receipt['rapidfuzz_version'] != rapidfuzz.__version__
        or receipt['verified_saved_predictions'] < 50_000):
    raise RuntimeError('A matching full metric-equivalence audit is required')
spec = importlib.util.spec_from_file_location('legacy', source)
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
legacy.distance = Levenshtein.distance
legacy.main()
output = Path(sys.argv[sys.argv.index('--output') + 1]) / 'results.json'
metadata = json.loads(output.read_text())
metadata['metric_implementation'] = {
    'legacy_evaluator_sha256': sha(source), 'wrapper_sha256': sha(Path(__file__)),
    'equivalence_receipt_sha256': sha(receipt_path), 'rapidfuzz': rapidfuzz.__version__,
    'difference': 'Verified equivalent integer word and character edit distance only',
}
output.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n')
