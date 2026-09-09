"""Extract all English splits from the audited, matched three-model diagnostic."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(root):
    audit = json.loads((root / 'numeric-audit.json').read_text())
    assert audit['status'] == 'pass'
    assert all(sha(root / name) == digest for name, digest in audit['inputs'].items())
    comparison = json.loads((root / 'comparison.json').read_text())
    models = ['parakeet', 'parent', 'candidate']
    splits = {name: spec for name, spec in comparison['sets'].items() if 'standard_english' in spec}
    totals, sizes, seen = defaultdict(Counter), Counter(), set()
    with gzip.open(root / 'numeric-evidence.jsonl.gz', 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            if 'standard_english' not in row['counts']:
                continue
            assert row['split'] in splits and row['record_sha256'] not in seen
            seen.add(row['record_sha256'])
            groups = ['all_english', row['evaluation_role']]
            for group in groups:
                sizes[group] += 1
                for variant, values in row['counts'].items():
                    for model in models:
                        totals[(group, variant, model)].update(values[model])
    assert len(splits) == 20 and len(seen) == sum(v['rows'] for v in splits.values()) == 5120
    groups = {}
    for group, count in sizes.items():
        entry = dict(rows=count, splits=len(splits) if group == 'all_english' else
                     sum(s['evaluation_role'] == group for s in splits.values()))
        for variant in ['standard_english', 'legacy']:
            rates = {}
            for model in models:
                counts = totals[(group, variant, model)]
                rates[model] = dict(counts, wer=100 * counts['errors'] / counts['words'],
                                    cer=100 * counts['char_errors'] / counts['chars'])
            for comparator in ['parakeet', 'parent']:
                rates['candidate_vs_' + comparator] = dict(
                    wer_delta_pp=rates['candidate']['wer'] - rates[comparator]['wer'],
                    wer_relative_reduction_percent=100 * (1 - rates['candidate']['wer'] / rates[comparator]['wer']))
            entry[variant] = rates
        groups[group] = entry
    wins = {}
    for comparator in ['parakeet', 'parent']:
        deltas = [s['standard_english']['candidate']['wer'] - s['standard_english'][comparator]['wer'] for s in splits.values()]
        wins[comparator] = dict(better=sum(d < 0 for d in deltas), tied=sum(d == 0 for d in deltas), worse=sum(d > 0 for d in deltas))
    report = dict(status='complete', scope='All 20 English splits in the latest matched diagnostic; 256 fixed recordings per split. Existing verified error counts aggregated; no new inference or full-corpus rerun.',
                  normalization='Standard English Whisper normalization; pooled word/character edit counts. Legacy normalization retained separately.',
                  exposure='4,352 training-exposed recordings and 768 controls excluded from this fine-tune. This is not a newly unseen benchmark.',
                  models=comparison['models'], manifest_sha256=comparison['manifest_sha256'],
                  script_sha256=sha(__file__), numeric_evidence_sha256=sha(root / 'numeric-evidence.jsonl.gz'),
                  source_comparison_sha256=sha(root / 'comparison.json'), groups=groups,
                  split_wer_outcomes=wins, splits=splits)
    (root / 'english-quick.json').write_text(json.dumps(report, indent=2) + '\n')
    lines = ['# Quick English comparison', '', report['scope'], '',
             'Word error rate (WER), with standard English normalization and identical audio/decoding for all models.', '',
             '| Sample | Clips | Base Parakeet | Previous Orukeet | Fine-tuned Orukeet |',
             '|:--|--:|--:|--:|--:|']
    labels = {'all_english': 'All English', 'training_exposed_fit_check': 'Training-exposed checks', 'nontraining_control': 'Controls excluded from this fine-tune'}
    for group in ['all_english', 'training_exposed_fit_check', 'nontraining_control']:
        entry = groups[group]
        values = ' | '.join(f"{entry['standard_english'][m]['wer']:.2f}%" for m in models)
        lines.append(f"| {labels[group]} | {entry['rows']:,} | {values} |")
    reduction = groups['all_english']['standard_english']
    lines += ['', f"Pooled WER falls by {reduction['candidate_vs_parakeet']['wer_relative_reduction_percent']:.2f}% relative to Parakeet and {reduction['candidate_vs_parent']['wer_relative_reduction_percent']:.2f}% relative to previous Orukeet.", '',
              f"The fine-tuned checkpoint beats Parakeet on {wins['parakeet']['better']}/20 sampled splits and previous Orukeet on {wins['parent']['better']}/20. Against previous Orukeet, humanities rises from 7.86% to 7.88% WER and EuroSpeech English rises from 24.93% to 25.23%.", '',
              report['exposure'], '', '| English split | Exposure | Parakeet WER | Previous WER | Fine-tuned WER |', '|:--|:--|--:|--:|--:|']
    for name, spec in sorted(splits.items()):
        exposure = 'trained' if spec['evaluation_role'] == 'training_exposed_fit_check' else 'control'
        values = ' | '.join(f"{spec['standard_english'][m]['wer']:.2f}%" for m in models)
        lines.append(f'| {name} | {exposure} | {values} |')
    lines += ['', 'Exact model hashes, integer counts, CER, and legacy-normalized results are in [english-quick.json](english-quick.json). The full diagnostic and provenance are in [the run report](../../training/regression_ft/README.md).', '']
    (root / 'ENGLISH.md').write_text('\n'.join(lines))
    print(json.dumps({k: {m: v['standard_english'][m]['wer'] for m in models} for k, v in groups.items()}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('evidence', type=Path)
    main(parser.parse_args().evidence)
