"""Independently sum per-record integer counts and check sealed diagnostic IDs."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(root):
    root = Path(root)
    read = lambda name: json.loads((root / name).read_text())
    protocol = read('diagnostic-protocol.json')
    prepared = read('diagnostic-prepared.json')
    comparison, summary = read('comparison.json'), read('summary.json')
    audit, repair = read('export-audit.json'), read('export-metadata-repair.json')
    complete = read('training-complete.json')
    assert prepared['protocol_sha256'] == sha(root / 'diagnostic-protocol.json')
    assert prepared['manifest_sha256'] == comparison['manifest_sha256'] == summary['manifest_sha256']
    assert summary['comparison_sha256'] == sha(root / 'comparison.json')
    assert comparison['models']['candidate'] == audit['candidate_sha256'] == repair['candidate_sha256']
    assert repair['source_sha256'] == complete['audit']['checkpoint_sha256']
    assert comparison['models']['parent'] == audit['parent_sha256'] == complete['run']['parent_sha256']
    assert audit['status'] == repair['status'] == 'pass'
    assert comparison['status'] == summary['status'] == 'complete' and comparison['failures'] == 0
    ids, totals, sizes = set(), defaultdict(Counter), Counter()
    with gzip.open(root / 'numeric-evidence.jsonl.gz', 'rt') as stream:
        for line in stream:
            record = json.loads(line)
            uid, split, role = record['record_sha256'], record['split'], record['evaluation_role']
            assert uid not in ids
            ids.add(uid)
            assert uid in protocol['splits'][split]['uid_sha256']
            assert role == protocol['splits'][split]['role']
            sizes[split] += 1
            for variant, models in record['counts'].items():
                assert len({c['words'] for c in models.values()}) == 1
                assert len({c['chars'] for c in models.values()}) == 1
                for label, counts in models.items():
                    assert all(isinstance(v, int) and v >= 0 for v in counts.values())
                    totals[(split, variant, label)].update(counts)
    assert len(ids) == prepared['rows'] == comparison['rows'] == 12006
    assert ids == {uid for s in protocol['splits'].values() for uid in s['uid_sha256']}
    roles = defaultdict(Counter)
    verified_rates = 0
    for split, spec in comparison['sets'].items():
        role = spec['evaluation_role']
        assert sizes[split] == spec['rows'] == protocol['splits'][split]['rows']
        variants = [v for v in ['legacy', 'standard_english'] if v in spec]
        selected = 'standard_english' if 'standard_english' in variants else 'legacy'
        for variant in variants:
            for label in ['parakeet', 'parent', 'candidate']:
                counts = totals[(split, variant, label)]
                for name, value in counts.items():
                    assert spec[variant][label][name] == value
                for metric, numerator, denominator in [('wer', 'errors', 'words'), ('cer', 'char_errors', 'chars')]:
                    value = 100 * counts[numerator] / counts[denominator]
                    assert math.isclose(value, spec[variant][label][metric], abs_tol=1e-10)
                    verified_rates += 1
                    if variant == selected:
                        assert math.isclose(value, summary['splits'][split][label + '_' + metric], abs_tol=1e-10)
                if variant == selected:
                    roles[(role, label)].update(counts)
    for role, spec in summary['roles'].items():
        assert spec['rows'] == prepared['roles'][role]
        for label in ['parakeet', 'parent', 'candidate']:
            counts = roles[(role, label)]
            assert spec['reference_words'] == counts['words'] and spec['reference_chars'] == counts['chars']
            for metric, numerator, denominator in [('wer', 'errors', 'words'), ('cer', 'char_errors', 'chars')]:
                assert math.isclose(spec[label + '_' + metric], 100 * counts[numerator] / counts[denominator], abs_tol=1e-10)
    result = dict(status='pass', rows=len(ids), splits=len(sizes), split_rates_recomputed=verified_rates,
                  all_sealed_record_ids_present_exactly_once=True,
                  integer_counts_and_pooled_rates_match=True, model_identity_chain_matches=True,
                  script_sha256=sha(__file__),
                  inputs={name: sha(root / name) for name in ['comparison.json', 'summary.json', 'numeric-evidence.jsonl.gz',
                          'diagnostic-protocol.json', 'diagnostic-prepared.json', 'export-audit.json',
                          'export-metadata-repair.json', 'training-complete.json']})
    (root / 'numeric-audit.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('evidence', type=Path)
    print(json.dumps(verify(parser.parse_args().evidence)), flush=True)
