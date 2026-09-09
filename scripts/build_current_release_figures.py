"""Render r3 launch figures directly from the complete benchmark counts."""
from pathlib import Path
import hashlib
import json

import matplotlib.pyplot as plt
from academic_style import publication_style, save_figure

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'launch/assets'


def main():
    source = ROOT / 'evidence/r3-promotion-20260908/scores.json'
    scores = json.loads(source.read_text())
    catalog = json.loads((ROOT / 'src/orukeet/artifacts.json').read_text())
    assert scores['models']['orukeet'] == catalog['files']['source']['sha256']
    panels = [('fleurs', 'FLEURS: 25 languages', '20,146 clips · 25 complete splits'),
              ('domain_english', 'English accents/domains', '5,120 clips · 20 splits')]
    with publication_style(font_size=10):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.0), sharey=True)
        fig.subplots_adjust(left=.10, right=.98, bottom=.24, top=.70, wspace=.30)
        fig.text(.10, .92, 'Orukeet improves on Parakeet', fontsize=16, weight='bold')
        fig.text(.10, .85, 'Pooled word error rate (%) · lower is better', fontsize=10, color='#4D5D65')
        for ax, (key, title, membership) in zip(axes, panels):
            summary = scores['summaries'][key]
            values = [summary['models'][m]['wer'] for m in ['parakeet', 'orukeet']]
            bars = ax.bar([0, 1], values, width=.55, color=['#A8AFB4', '#12617A'])
            ax.set_ylim(0, 14)
            ax.set_yticks([0, 5, 10])
            ax.set_xticks([0, 1], ['Parakeet', 'Orukeet'])
            ax.set_title(title, loc='left', weight='bold', pad=19)
            ax.text(0, 1.025, membership, transform=ax.transAxes, fontsize=8.5, color='#4D5D65')
            ax.grid(axis='y', color='#E5E8EA', linewidth=.6)
            for bar, value in zip(bars, values):
                ax.text(bar.get_x()+bar.get_width()/2, value+.45, f'{value:.2f}',
                        ha='center', va='bottom', fontsize=11, weight='bold')
            reduction = 100*(1-values[1]/values[0])
            ax.text(.5, -.31, f'{reduction:.1f}% relative WER reduction',
                    transform=ax.transAxes, ha='center', fontsize=10, color='#12617A', weight='bold')
        fig.text(.10, .055, 'r3 vs NVIDIA Parakeet TDT 0.6B v3 · matched NeMo greedy decoding', fontsize=7.5)
        fig.text(.10, .020, 'Summed edit counts; FP32 weights / BF16 CUDA. Full split scores and protocol: docs/current-checkpoint-benchmarks.md', fontsize=7)
        paths = save_figure(fig, OUT/'current-benchmark-summary', formats=['pdf', 'svg', 'png'], dpi=300)
    receipt = dict(status='rendered-awaiting-visual-review', model='Orukeet r3',
                   source=str(source.relative_to(ROOT)), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   model_sha256=scores['models']['orukeet'], figsize_inches=[7.2, 4.0],
                   panels={key:scores['summaries'][key] for key, _, _ in panels},
                   improvements=scores['improvements'],
                   alt_text='Orukeet r3 versus NVIDIA Parakeet TDT 0.6B v3. Across 20,146 FLEURS clips and 25 languages, pooled WER is 9.85% versus 11.01%, a 10.6% relative reduction. Across 5,120 English accent/domain clips, WER is 8.84% versus 9.51%, a 7.0% reduction. Both comparisons use matched NeMo decoding; all split scores and the evaluation protocol accompany the release.',
                   files=[dict(path=str(p.relative_to(ROOT)), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths])
    (OUT/'current-benchmark-summary.json').write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__':
    main()
