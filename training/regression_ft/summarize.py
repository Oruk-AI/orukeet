"""Summarize matched diagnostics with paired, split-stratified cluster bootstrap."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rates(a):
    # columns: reference words/chars; baseline/parent/candidate word/char errors
    return dict(parakeet_wer=100 * a[..., 2] / a[..., 0], parent_wer=100 * a[..., 3] / a[..., 0],
                candidate_wer=100 * a[..., 4] / a[..., 0], parakeet_cer=100 * a[..., 5] / a[..., 1],
                parent_cer=100 * a[..., 6] / a[..., 1], candidate_cer=100 * a[..., 7] / a[..., 1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--evaluation', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--bootstrap', type=int, default=2000)
    a = p.parse_args()
    report = json.loads((a.evaluation / 'comparison.json').read_text())
    assert report['status'] == 'complete' and sha(a.manifest) == report['manifest_sha256']
    raw = [json.loads(line) for line in a.manifest.open()]
    metadata = {hashlib.sha256(r['uid'].encode()).hexdigest(): r for r in raw}
    with gzip.open(a.evaluation / 'numeric-evidence.jsonl.gz', 'rt') as f:
        numeric = [json.loads(line) for line in f]
    assert len(numeric) == len(metadata) == report['rows']
    assert {r['record_sha256'] for r in numeric} == metadata.keys()
    strata = defaultdict(lambda: defaultdict(lambda: np.zeros(8, dtype=np.int64)))
    sizes, cluster_metadata = defaultdict(int), defaultdict(int)
    for r in numeric:
        row = metadata[r['record_sha256']]
        assert r['split'] == row['split'] and r['evaluation_role'] == row['evaluation_role']
        variant = 'standard_english' if row['lang'] == 'en' else 'legacy'
        c = r['counts'][variant]
        assert len({c[label]['words'] for label in c}) == len({c[label]['chars'] for label in c}) == 1
        vector = [c['parakeet']['words'], c['parakeet']['chars']]
        vector += [c[label]['errors'] for label in ['parakeet', 'parent', 'candidate']]
        vector += [c[label]['char_errors'] for label in ['parakeet', 'parent', 'candidate']]
        key = (row['evaluation_role'], row['split'])
        available = row.get('cluster_metadata_available', True) and bool(row.get('cluster'))
        cluster = row['cluster'] if available else r['record_sha256']
        strata[key][cluster] += np.array(vector)
        sizes[key] += 1
        cluster_metadata[key] += int(available)
    rng = np.random.default_rng(20260919)
    aggregates, samples, details = {}, {}, {}
    for (role, split), groups in sorted(strata.items()):
        x = np.stack(list(groups.values()))
        observed = x.sum(0)
        weights = rng.multinomial(len(x), np.full(len(x), 1 / len(x)), size=a.bootstrap)
        replicas = weights @ x
        aggregates.setdefault(role, np.zeros(8, dtype=np.int64))
        samples.setdefault(role, np.zeros((a.bootstrap, 8), dtype=np.int64))
        aggregates[role] += observed
        samples[role] += replicas
        details[split] = dict(role=role, rows=sizes[(role, split)], bootstrap_clusters=len(x),
                             rows_with_cluster_metadata=cluster_metadata[(role, split)],
                             **{k: float(v) for k, v in rates(observed).items()})
    results = {}
    for role in aggregates:
        point = rates(aggregates[role])
        boot = rates(samples[role])
        item = dict(rows=sum(v['rows'] for v in details.values() if v['role'] == role),
                    splits=sum(v['role'] == role for v in details.values()),
                    reference_words=int(aggregates[role][0]), reference_chars=int(aggregates[role][1]),
                    **{k: float(v) for k, v in point.items()})
        for metric in ['wer', 'cer']:
            for comparator in ['parakeet', 'parent']:
                delta = boot['candidate_' + metric] - boot[comparator + '_' + metric]
                item['candidate_minus_' + comparator + '_' + metric] = dict(
                    delta_pp=float(point['candidate_' + metric] - point[comparator + '_' + metric]),
                    paired_cluster_bootstrap_95ci_pp=np.quantile(delta, [.025, .975]).tolist(),
                    relative_reduction_percent=float(100 * (1 - point['candidate_' + metric] / point[comparator + '_' + metric])))
        results[role] = item
    output = dict(status='complete', comparison_sha256=sha(a.evaluation / 'comparison.json'),
                  manifest_sha256=sha(a.manifest), script_sha256=sha(__file__),
                  normalization='Standard English for English; legacy multilingual normalization elsewhere.',
                  bootstrap=dict(replicates=a.bootstrap, seed=20260919,
                                 unit='Recording/speaker cluster where available, otherwise utterance; paired across models and stratified by split.',
                                 limitation='Intervals are conditional on these selected datasets and available cluster metadata. Fit checks are training-exposed.'),
                  roles=results, splits=details)
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / 'summary.json').write_text(json.dumps(output, indent=2) + '\n')
    lines = ['# Regression fine-tuning diagnostics', '',
             'Matched fixed samples, scored with standard English normalization for English and the existing multilingual normalizer elsewhere.', '',
             '| Sample | Recordings | Base Parakeet WER | Previous Orukeet WER | Candidate WER |',
             '|:--|--:|--:|--:|--:|']
    for role, item in results.items():
        lines.append(f"| {role.replace('_', ' ')} | {item['rows']:,} | {item['parakeet_wer']:.2f}% | {item['parent_wer']:.2f}% | {item['candidate_wer']:.2f}% |")
    lines += ['', 'The training-exposed sample measures fit. The control sample contains recordings excluded from this training run; it is not a newly collected unseen benchmark.', '',
              '| Split | Role | Base WER | Previous WER | Candidate WER |', '|:--|:--|--:|--:|--:|']
    for split, item in details.items():
        lines.append(f"| {split} | {'fit' if item['role'] == 'training_exposed_fit_check' else 'control'} | {item['parakeet_wer']:.2f}% | {item['parent_wer']:.2f}% | {item['candidate_wer']:.2f}% |")
    lines += ['', 'Paired uncertainty intervals, character error rates, cluster counts, exact model identities, and numeric evidence are retained in the accompanying JSON files.', '']
    (a.output / 'RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
