"""Score all three checkpoints without changing the existing benchmark evidence."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'evaluation/standard_asr'))
from scoring import counts


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    source_path = a.predictions / 'inference-comparison.json'
    source = json.loads(source_path.read_text())
    assert source['status'] == 'complete' and sha(a.manifest) == source['manifest_sha256']
    rows = [json.loads(line) for line in a.manifest.open()]
    assert len(rows) == len({r['uid'] for r in rows}) == source['rows'] == 4265
    predictions = {}
    for model, model_sha in source['models'].items():
        path = a.predictions / (model + '.jsonl')
        assert sha(path) == source['prediction_sha256'][model]
        part = [json.loads(line) for line in path.open()]
        assert len(part) == len(rows) == len({r['uid'] for r in part})
        assert {r['uid'] for r in rows} == {r['uid'] for r in part}
        for record in part:
            assert record['model_sha256'] == model_sha
            for key in ['manifest_sha256', 'script_sha256', 'decoding_sha256']:
                assert record[key] == source[key]
        predictions[model] = {r['uid']: r['prediction'] for r in part}
    totals = defaultdict(Counter)
    with gzip.open(a.output / 'numeric-evidence.jsonl.gz', 'wt') as stream:
        for row in rows:
            assert hashlib.sha256(row['text'].encode()).hexdigest() == row['reference_sha256']
            values = {m: counts(row['text'], pred[row['uid']], row['language']) for m,pred in predictions.items()}
            for model, value in values.items():
                totals[row['split'], model].update(value)
            stream.write(json.dumps(dict(record_sha256=hashlib.sha256(row['uid'].encode()).hexdigest(),
                                         split=row['split'], language=row['language'], duration=row['duration'],
                                         counts=values)) + '\n')
    for split, spec in source['sets'].items():
        spec['models'] = {model: dict(totals[split, model],
                                     wer=100 * totals[split, model]['errors'] / totals[split, model]['words'],
                                     cer=100 * totals[split, model]['char_errors'] / totals[split, model]['chars'])
                          for model in predictions}
    code = [Path(__file__), ROOT / 'evaluation/standard_asr/scoring.py']
    code += [p for p in (ROOT / 'evaluation/standard_asr/vendor').iterdir() if p.is_file()]
    source['scoring'] = dict(protocol='Identical pinned normalization and compound-aware WER as the preceding full evaluation.',
                             dependencies={n: version(n) for n in ['kaldialign','num2words','regex','rapidfuzz']},
                             code_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in code},
                             reference_word_counts='Non-English compound alignment can change reference token counts separately for each model.',
                             cer='Character distance before compound alignment, including normalized spaces.',
                             inference_comparison_sha256=sha(source_path))
    source['numeric_evidence_sha256'] = sha(a.output / 'numeric-evidence.jsonl.gz')
    (a.output / 'comparison.json').write_text(json.dumps(source, ensure_ascii=False, indent=2) + '\n')
    for split, spec in sorted(source['sets'].items()):
        print(split, {model: round(value['wer'], 4) for model,value in spec['models'].items()})


if __name__ == '__main__':
    main()
