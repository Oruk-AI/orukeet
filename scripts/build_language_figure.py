"""Rebuild the release's per-language evidence figure from committed estimates.

Needs matplotlib. SVG is the editable launch master; PNG is its 300-DPI preview.
No evaluation is run and no confidence interval is recomputed here.
"""
from pathlib import Path
import csv
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'evidence/release-sensitivity.json'
OUT = ROOT / 'launch/assets'
NAMES = dict(cs='Czech', da='Danish', de='German', en='English', es='Spanish', fi='Finnish',
             fr='French', hr='Croatian', hu='Hungarian', it='Italian', lt='Lithuanian', lv='Latvian',
             nl='Dutch', pl='Polish', pt='Portuguese', ro='Romanian', ru='Russian', sk='Slovak',
             sl='Slovenian', sv='Swedish')

def main():
    data = json.loads(SOURCE.read_text())
    rows = [{'language': NAMES[k.split(':')[1]], 'code': k.split(':')[1], **v}
            for k, v in data['metrics'].items() if k.startswith('language:')]
    assert len(rows) == 20
    rows.sort(key=lambda r: r['language'])
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.spines.left': False, 'svg.fonttype': 'path'})
    fig = plt.figure(figsize=(7, 8.4), facecolor='white')
    ax = fig.add_axes([.20, .25, .52, .58])
    y = np.arange(len(rows))
    for i, row in enumerate(rows):
        delta = row['delta_pp']; low, high = row['paired_delta_95ci']
        color = '#0F4D92' if delta < 0 else '#B64342'
        ax.errorbar(delta, i, xerr=[[delta-low], [high-delta]], fmt='o' if delta < 0 else 'D',
                    color=color, markersize=4, elinewidth=1.1, capsize=2)
    ax.set_yticks(y, [r['language'] for r in rows]); ax.invert_yaxis()
    ax.tick_params(axis='y', length=0, pad=8)
    ax.axvline(0, color='#767676', lw=.8, linestyle='--')
    ax.set_xlim(-16, 3); ax.set_xticks([-15, -10, -5, 0])
    ax.grid(axis='x', color='#eeeeee', lw=.65); ax.set_axisbelow(True)
    ax.set_xlabel('Change in word error rate (percentage points)', labelpad=9)
    # Values in an aligned table avoid a compressed secondary axis near zero.
    trans = ax.get_yaxis_transform()
    for i, row in enumerate(rows):
        ax.text(1.04, i, f"{row['reference_wer']:.2f} → {row['candidate_wer']:.2f}",
                transform=trans, va='center', fontsize=8.2)
    ax.text(1.04, 1.055, 'Stock → pre-surgery Q8', transform=ax.transAxes, fontsize=8.3)
    ax.text(0, 1.055, 'Lower is better', transform=ax.transAxes, color='#0F4D92', fontsize=9)
    fig.text(.07, .955, 'Orukeet before Gabor surgery', fontsize=16, weight='bold', color='#142333')
    fig.text(.07, .919, '20-language macro WER: 16.58% → 15.30%', fontsize=12)
    fig.text(.07, .89, '7.70% relative reduction on reused evaluation data', fontsize=10, color='#505050')
    fig.text(.07, .166, 'Latvian accounts for about half the average reduction.', fontsize=10, weight='bold')
    fig.text(.07, .138, 'Excluding Latvian after evaluation leaves 4.24% relative improvement.', fontsize=9)
    fig.text(.07, .108, 'Points: language estimates. Lines: paired 95% cluster-bootstrap intervals.', fontsize=8.3)
    fig.text(.07, .085, '5,000 resamples; shared speaker / sentence clusters. No multiplicity adjustment.', fontsize=8.3)
    fig.text(.07, .062, '13,246 primary / 23,038 total reused recordings; runtime and precision differ.', fontsize=8.3)
    fig.text(.07, .039, 'The original 10% promotion target was missed. General accent gains are unproven.', fontsize=8.3)
    fig.text(.07, .017, 'Oruk-AI/orukeet · release candidate · evidence/release-sensitivity.json', fontsize=7.7, color='#505050')
    OUT.mkdir(exist_ok=True)
    files = []
    for ext in ['svg', 'png']:
        dest = OUT / f'orukeet-language-evidence.{ext}'
        fig.savefig(dest, dpi=300)
        files.append({'path': str(dest.relative_to(ROOT)), 'sha256': hashlib.sha256(dest.read_bytes()).hexdigest()})
    plt.close(fig)
    csv_path = OUT / 'orukeet-language-evidence.csv'
    with csv_path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['language', 'code', 'stock_wer', 'orukeet_q8_wer', 'delta_pp', 'ci_low_pp', 'ci_high_pp', 'rows'])
        writer.writeheader()
        for r in rows:
            writer.writerow(dict(language=r['language'], code=r['code'], stock_wer=r['reference_wer'],
                                 orukeet_q8_wer=r['candidate_wer'], delta_pp=r['delta_pp'],
                                 ci_low_pp=r['paired_delta_95ci'][0], ci_high_pp=r['paired_delta_95ci'][1], rows=r['rows']))
    alt = ('Pre-surgery Orukeet Q8 versus stock Parakeet: 20 language estimates and paired 95% cluster-bootstrap intervals. '
           'Nineteen point estimates improve; German is slightly worse. Latvian has the largest drop, '
           '33.20% to 20.33% word error rate, and accounts for about half the macro reduction. '
           'The overall macro drops from 16.58% to 15.30%, or 7.70% relatively. '
           'The post-hoc average without Latvian improves 4.24% relatively. Several language intervals cross zero. '
           'Evaluation data were reused; runtime and precision differ; the original 10% target was missed. '
           'The CSV carries every plotted value.')
    receipt = {'source': str(SOURCE.relative_to(ROOT)), 'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
               'figure_inches': [7, 8.4], 'png_dpi': 300, 'rows': len(rows), 'files': files,
               'csv': str(csv_path.relative_to(ROOT)), 'alt_text': alt,
               'uncertainty': data['bootstrap'], 'visual_review': 'pending'}
    (OUT / 'orukeet-language-evidence.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({'files': files, 'languages': len(rows)}))

if __name__ == '__main__': main()
