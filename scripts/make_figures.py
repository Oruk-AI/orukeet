"""Render report figures from committed measurements; no invented error bars."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from academic_style import publication_style, save_figure

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/figures'
BLUE, GREY, TEAL = '#0F4D92', '#767676', '#287C80'


def main():
    result = json.loads((ROOT / 'evidence/release-sensitivity.json').read_text())
    languages = {'cs':'Czech','da':'Danish','de':'German','en':'English','es':'Spanish',
                 'fi':'Finnish','fr':'French','hr':'Croatian','hu':'Hungarian','it':'Italian',
                 'lt':'Lithuanian','lv':'Latvian','nl':'Dutch','pl':'Polish','pt':'Portuguese',
                 'ro':'Romanian','ru':'Russian','sk':'Slovak','sl':'Slovenian','sv':'Swedish'}
    with publication_style(font_size=9, font_family='DejaVu Sans'):
        fig, (ax, lower) = plt.subplots(2, 1, figsize=(7, 6.9),
                                        gridspec_kw={'height_ratios':[4.7, 1.3]}, layout='constrained')
        for y, (key, label) in enumerate(languages.items()):
            value = result['metrics']['language:' + key]
            lo, hi = value['paired_delta_95ci']
            delta = value['delta_pp']
            ax.plot([lo, hi], [y, y], color=BLUE, lw=1.2)
            ax.plot(delta, y, 'o', color=BLUE, ms=4)
        ax.set_yticks(range(len(languages)), list(languages.values()))
        ax.invert_yaxis()
        ax.axvline(0, color=GREY, ls='--', lw=.8)
        ax.set(xlim=(-16.3, 8), xlabel='Q8 minus stock WER (percentage points; lower is better)')
        ax.set_title('A  Language estimates and paired 95% intervals', loc='left', fontsize=11)
        ax.grid(axis='x', color='#dddddd', lw=.5)
        values = [result['primary'], result['posthoc_sensitivity']['leave_one_language_out']['lv'], result['english']]
        labels = ['20-language macro', 'Without Latvian*', 'Equal-corpus English']
        for y, value in enumerate(values):
            lo, hi = value['paired_delta_95ci']
            lower.plot([lo, hi], [y, y], color=TEAL, lw=1.6)
            lower.plot(value['delta_pp'], y, 'D', color=TEAL, ms=4)
        lower.set_yticks(range(3), labels)
        lower.invert_yaxis()
        lower.set_ylim(2.5, -.5)
        lower.set(xlim=(-1.7, .1), xlabel='WER change (percentage points)')
        lower.axvline(0, color=GREY, ls='--', lw=.8)
        lower.set_title('B  Aggregate improvement persists without Latvian', loc='left', fontsize=11)
        lower.grid(axis='x', color='#dddddd', lw=.5)
        fig.get_layout_engine().set(rect=(0, .055, 1, .945))
        fig.text(.03, .015, '5,000 shared speaker/sentence bootstrap draws; fixed languages. *Post-hoc sensitivity.\nIndividual intervals are unadjusted. Panels use different x-axis scales.', fontsize=8, color='#444444')
        save_figure(fig, OUT / 'accuracy', formats=('pdf', 'svg', 'png'))

    rows = [('M5 Max · Q8 Metal','mac-metal-q8.json',BLUE),
            ('M5 Max · FP16 Metal','mac-metal-fp16-full.json',GREY),
            ('M5 Max · Q8 CPU','mac-cpu-q8.json',BLUE),
            ('M5 Max · Whisper Base CPU','mac-whisper-base.json',GREY),
            ('A100 · Q8 CUDA','a100-cuda-q8.json',TEAL)]
    with publication_style(font_size=9, font_family='DejaVu Sans'):
        fig, ax = plt.subplots(figsize=(7, 3.3), layout='constrained')
        for y, (_, file, color) in enumerate(rows):
            data = json.loads((ROOT / 'evidence' / file).read_text())['results'][0]
            samples = np.asarray(data['warm_s'])
            assert len(samples) == 10 and np.isclose(np.median(samples), data['warm_median_s'])
            ax.scatter(samples, y + np.linspace(-.12, .12, len(samples)), color=color, alpha=.45, s=14)
            median = np.median(samples)
            ax.plot(median, y, '|', ms=20, mew=2, color=color)
            ax.text(3.28, y, f'{median:.3f} s', va='center', ha='right', fontsize=9, color='#272727')
        ax.set_yticks(range(len(rows)), [r[0] for r in rows])
        ax.invert_yaxis()
        ax.set(xlim=(0, 3.35), xlabel='Warm transcription time (seconds; 11 seconds of audio)')
        ax.set_title('One clip, ten runs, different hardware', loc='left', fontsize=11)
        ax.grid(axis='x', color='#dddddd', lw=.5)
        fig.get_layout_engine().set(rect=(0, .1, 1, .9))
        fig.text(.03, .025, 'Dots: individual calls. Vertical marks: medians. Includes file decode and worker IPC;\nexcludes model loading, hotkey post-roll, pasting and cleanup.', fontsize=8, color='#444444')
        save_figure(fig, OUT / 'latency', formats=('pdf', 'svg', 'png'))


if __name__ == '__main__':
    main()
