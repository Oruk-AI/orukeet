"""Plot measured paired accuracy deltas, including unsuccessful recovery runs."""
import argparse
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--results', type=Path, required=True)
a = p.parse_args()
records = []
for path in a.results.glob('full-*-*.json'):
    match = re.fullmatch(r'full-([ra])(\d+)-(\d+)\.json', path.name)
    if match:
        data = json.loads(path.read_text())
        assert data['rows'] == 23038
        assert data['reference_model_sha256'] == '313d615ca34c8ac3a183384e1f87e8274d748e5445af110013a335f02ae42e32'
        family, run, variant = match[1], int(match[2]), int(match[3])
        if family == 'r':
            label = f'R{run}, step {variant}'
        else:
            # Average labels encode the final parent's share, not a training step.
            declaration = json.loads((a.results / f'a{run}-start-decision.json').read_text())
            candidate = next(item for item in declaration['candidates']
                             if item['label'] == f'a{run}-{variant:04d}')
            parent = declaration['parents'][-1]['label']
            assert re.fullmatch(r'r\d+-\d+', parent)
            label = f'A{run}, {candidate["weights"][-1]:.0%} {parent.split("-")[0].upper()}'
        records.append(((0 if family == 'r' else 1, run, variant), label, data))
records.sort(key=lambda row: row[0])
assert records
for _, _, data in records:
    assert data['registry_sha256'] == records[0][2]['registry_sha256']
    assert data['bootstrap'] == records[0][2]['bootstrap']
    assert data['bootstrap']['replicates'] == 5000
    assert data['hours'] == records[0][2]['hours']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10,
    'axes.labelsize': 9, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .7})
fig, axes = plt.subplots(1, 2, figsize=(7, 1.55 + .42 * len(records)), sharey=True, layout='constrained')
labels = [label for _, label, _ in records]
intervals = [d[metric]['paired_delta_95ci'] for _, _, d in records for metric in ('primary', 'english')]
low = min(v[0] for v in intervals) - .03
high = max(v[1] for v in intervals) + .03
for ax, metric, title, color in zip(axes, ('primary', 'english'),
    ('Multilingual: 20-language macro', 'English: seven-corpus macro'), ('#0072B2', '#D55E00')):
    for y, (_, _, data) in enumerate(records):
        item = data[metric]
        delta = item['delta_pp']
        lower, upper = item['paired_delta_95ci']
        ax.errorbar(delta, y, xerr=np.array([[delta - lower], [upper - delta]]),
            fmt='o', color=color, markersize=4.5, capsize=3, elinewidth=1.2)
    ax.axvline(0, color='#666666', linestyle='--', linewidth=.8, zorder=0)
    ax.set(title=title, xlabel='WER change vs original (percentage points)', xlim=(low, high),
           yticks=range(len(records)), yticklabels=labels, ylim=(len(records) - .5, -.5))
    ax.grid(axis='x', color='#dddddd', linewidth=.4)
fig.suptitle('Recovery on an exposed regression suite', fontsize=10)
for ext in ('pdf', 'svg', 'png'):
    fig.savefig(a.results / ('recovery-accuracy.' + ext), dpi=300)
caption = ('Paired WER change from the original Orukeet source; lower is better. '
    'Points are the predeclared macro endpoints and bars are their 95% paired cluster-bootstrap intervals '
    '(5,000 resamples). Each run evaluates the same 23,038 recordings (53.67 hours); the multilingual endpoint uses 13,246 primary recordings. '
    'The dashed line marks equal WER. These recordings were already exposed during development; '
    'the intervals do not account for adaptive model selection. Acceptance also requires the separate '
    'language/corpus guards and development checks, which are retained in the linked JSON decisions. '
    'Only checkpoints with a completed larger regression appear.\n')
(a.results / 'recovery-accuracy-caption.md').write_text(caption)
(a.results / 'recovery-accuracy-sources.json').write_text(json.dumps([
    {'label': label, 'candidate_sha256': data['candidate_model_sha256'], 'status': data['status'],
     'release_qualified': data['release_qualified']} for label, (_, _, data) in zip(labels, records)
], indent=2) + '\n')
