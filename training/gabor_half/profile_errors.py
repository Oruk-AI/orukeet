"""Aggregate substitution/deletion/insertion diagnostics without emitting speech."""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path


def error_types(reference, hypothesis):
    # Equal-cost paths prefer diagonal, deletion, then insertion. Counts depend
    # on this tie convention; total edit distance does not.
    previous = [(j, 0, 0, j) for j in range(len(hypothesis) + 1)]
    for i, left in enumerate(reference, 1):
        current = [(i, 0, i, 0)]
        for j, right in enumerate(hypothesis, 1):
            mismatch = int(left != right)
            candidates = [
                (previous[j - 1][0] + mismatch, 0, previous[j - 1][1] + mismatch,
                 previous[j - 1][2], previous[j - 1][3]),
                (previous[j][0] + 1, 1, previous[j][1], previous[j][2] + 1, previous[j][3]),
                (current[j - 1][0] + 1, 2, current[j - 1][1], current[j - 1][2], current[j - 1][3] + 1),
            ]
            distance, _, substitutions, deletions, insertions = min(candidates)
            current.append((distance, substitutions, deletions, insertions))
        previous = current
    distance, substitutions, deletions, insertions = previous[-1]
    assert distance == substitutions + deletions + insertions
    return Counter(errors=distance, substitutions=substitutions, deletions=deletions, insertions=insertions)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--evaluator', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    spec = importlib.util.spec_from_file_location('evaluation_normalizer', a.evaluator)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    metadata = [json.loads((path / 'results.json').read_text()) for path in (a.reference, a.candidate)]
    assert metadata[0]['normalizer'] == metadata[1]['normalizer'] == 'legacy-compatible-nfc-v1'
    result = {'reference_sha256': metadata[0]['model_sha256'], 'candidate_sha256': metadata[1]['model_sha256'],
              'interpretation': 'Paired diagnostic on existing exposed evaluation. Edit-type counts depend on alignment tie-breaking and do not identify a causal mechanism.', 'groups': {}}
    for group, pattern, expected_rows in [
        ('learner', 'learner_en_speechocean_official_test_hypotheses.jsonl', 2500),
        ('voxpopuli_accented', 'parliamentaccent_en_voxpopuli_confirmation_hypotheses.jsonl', 1192),
    ]:
        totals = [Counter(), Counter()]
        paired = Counter()
        names = sorted(x.name for x in a.reference.glob(pattern))
        assert names and names == sorted(x.name for x in a.candidate.glob(pattern))
        for name in names:
            old = [json.loads(x) for x in (a.reference / name).read_text().splitlines()]
            new = [json.loads(x) for x in (a.candidate / name).read_text().splitlines()]
            assert len(old) == len(new)
            for before, after in zip(old, new):
                assert (before['audio_filepath'], before['text']) == (after['audio_filepath'], after['text'])
                for total, row in zip(totals, (before, after)):
                    reference = evaluator.normalize(row['text']).split()
                    hypothesis = evaluator.normalize(row['pred_text']).split()
                    counts = error_types(reference, hypothesis)
                    assert counts['errors'] == row['errors'] and len(reference) == row['words']
                    total.update(counts)
                    total.update(rows=1, words=len(reference))
                delta = after['errors'] - before['errors']
                paired['improved' if delta < 0 else 'worsened' if delta > 0 else 'same_error_count'] += 1
        assert totals[0]['rows'] == totals[1]['rows'] == expected_rows
        result['groups'][group] = {'manifest_outputs': names, 'reference': dict(totals[0]),
                                   'candidate': dict(totals[1]), 'paired': dict(paired)}
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
