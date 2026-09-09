"""Check published score rows against transcript-free per-record edit counts."""
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'evidence/benchmark-release-20260907'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    scores = json.loads((OUT / 'scores.json').read_text())
    checked = []
    for name, models, bundles in [
        ('complete', {'parakeet': 'parakeet', 'r15': 'orukeet'}, [
            'evidence/unseen-20260907/numeric-evidence.jsonl.gz',
            'evidence/unseen-20260907/coverage/numeric-evidence.jsonl.gz',
            'evidence/unseen-20260907/alignment-followup/numeric-evidence.jsonl.gz']),
        ('sampled', {'parakeet': 'parakeet', 'r15': 'parent', 'ft4035': 'candidate'}, [
            'evidence/regression-ft-20260907/numeric-evidence.jsonl.gz'])]:
        counts, sizes = defaultdict(Counter), Counter()
        rows = {row['split']: row for row in scores[name]}
        for bundle in bundles:
            with gzip.open(ROOT / bundle, 'rt') as stream:
                for line in stream:
                    record = json.loads(line)
                    split = record['bucket' if name == 'complete' else 'split']
                    if name == 'complete':
                        values = record['english_standard'] if rows[split]['english'] else record
                    else:
                        values = record['counts']['standard_english' if rows[split]['english'] else 'legacy']
                    sizes[split] += 1
                    for model, source in models.items():
                        counts[(split, model)].update({k: values[source][k] for k in ['errors', 'words', 'char_errors', 'chars']})
            checked.append(bundle)
        assert set(sizes) == set(rows) and len(rows) == 47
        for split, row in rows.items():
            assert row['rows'] == sizes[split]
            for model in models:
                totals = counts[(split, model)]
                for metric, value in totals.items():
                    assert row['counts'][model][metric] == value, (name, split, model, metric)
                for metric, numerator, denominator in [('wer', 'errors', 'words'), ('cer', 'char_errors', 'chars')]:
                    assert math.isclose(row['counts'][model][metric], 100 * totals[numerator] / totals[denominator], abs_tol=1e-10)
        with (OUT / (name + '.csv')).open() as stream:
            csv_rows = list(csv.DictReader(stream))
        assert len(csv_rows) == 47
        for entry in csv_rows:
            row = rows[entry['split']]
            assert int(entry['clips']) == row['rows']
            for model in models:
                for metric in ['wer', 'cer']:
                    assert float(entry[model + '_' + metric]) == row['counts'][model][metric]
        checked.append(str((OUT / (name + '.csv')).relative_to(ROOT)))
    pdf = 'output/pdf/orukeet-technical-report.pdf'
    text = subprocess.check_output(['pdftotext', '-layout', str(ROOT / pdf), '-'], text=True)
    pdf_lines = {' '.join(line.split()) for line in text.splitlines()}
    cards = ['README.md', 'MODEL_CARD.md', 'hub/README.md',
             'launch/publication/github-README.md', 'launch/publication/github-MODEL_CARD.md',
             'launch/publication/huggingface-README.md', 'docs/benchmark-scores.md']
    # The curated source export omits private preparation copies.
    required_cards = {'README.md', 'MODEL_CARD.md', 'docs/benchmark-scores.md'}
    cards = [name for name in cards if name in required_cards or (ROOT / name).exists()]
    contents = {name: (ROOT / name).read_text() for name in cards}
    for name in ['complete', 'sampled']:
        models = ['parakeet', 'r15'] + (['ft4035'] if name == 'sampled' else [])
        for row in scores[name]:
            numbers = [f"{row['counts'][model][metric]:.2f}" for model in models for metric in ['wer', 'cer']]
            assert ' '.join([row['split'], f"{row['rows']:,}"] + numbers) in pdf_lines
            markdown = '| ' + row['split'] + f" | {row['rows']:,} | " + ' | '.join(' / '.join(numbers[i:i + 2]) for i in range(0, len(numbers), 2)) + ' |'
            for card, content in contents.items():
                assert markdown in content, (card, row['split'])
    for card, content in contents.items():
        assert '6,118 comparison clips were included' in content, card
        assert 'Orukeet FT-4035 is the source of the current NeMo, Q8 and F16 downloads' in content, card
    checked += cards + [pdf, 'evidence/benchmark-release-20260907/scores.json']
    subprocess.run([sys.executable, str(ROOT/'evaluation/standard_asr/build_materials.py'), '--verify-pdf'], check=True)
    standard = json.loads((ROOT/'evidence/standard-asr-20260908/materials-validation.json').read_text())
    receipt = dict(status='passed', publication_authorized=False,
                   complete_clips=327888, sampled_clips=12006, pdf_score_rows=94+standard['splits'],
                   standard_test_clips=standard['rows'], standard_test_splits=standard['splits'],
                   markdown_score_rows=94 * len(cards), csv_score_rows=94,
                   checks=['Per-record integer edit counts match every score', 'Every split is present',
                           'Full-precision CSV values match', 'Rendered PDF values match',
                           'All available Markdown score surfaces match', 'Model and sample conditions retained'],
                   inputs_sha256={name: sha(ROOT / name) for name in checked}, script_sha256=sha(Path(__file__)))
    (OUT / 'materials-validation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'inputs_sha256'}))


if __name__ == '__main__':
    main()
