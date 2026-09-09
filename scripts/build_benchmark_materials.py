"""Build complete score tables from audited evidence, preserving model identity."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    return json.loads((ROOT / name).read_text())


def rates(rows, models):
    result = {}
    for model in models:
        counts = Counter()
        for row in rows:
            counts.update({k: row['counts'][model][k] for k in ['words', 'errors', 'chars', 'char_errors']})
        result[model] = dict(counts, wer=100 * counts['errors'] / counts['words'],
                             cer=100 * counts['char_errors'] / counts['chars'])
    return result


def markdown_table(rows, models):
    names = {'parakeet': 'Parakeet', 'r15': 'Orukeet R15', 'ft4035': 'Orukeet FT-4035'}
    lines = ['| Split | Clips | ' + ' | '.join(names[m] + ' WER / CER' for m in models) + ' |',
             '|:--|--:|' + '--:|' * len(models)]
    for row in rows:
        values = ' | '.join(f"{row['counts'][m]['wer']:.2f} / {row['counts'][m]['cer']:.2f}" for m in models)
        lines.append(f"| {row['split']} | {row['rows']:,} | {values} |")
    return '\n'.join(lines)


def tex_table(rows, models, caption, label):
    names = {'parakeet': 'Parakeet', 'r15': 'R15-0100', 'ft4035': 'FT-4035'}
    ncols = 2 + 2 * len(models)
    lines = [r'\begin{table}[!ht]', r'\centering', r'\caption{' + caption + '}',
             r'\label{' + label + '}', r'\footnotesize', r'\setlength{\tabcolsep}{3.4pt}',
             r'\renewcommand{\arraystretch}{1.08}', r'\begin{tabular}{l' + 'r' * (ncols - 1) + '}',
             r'\toprule', 'Split & Clips & ' + ' & '.join(r'\multicolumn{2}{c}{' + names[m] + '}' for m in models) + r'\\',
             ' & & ' + ' & '.join(['WER & CER'] * len(models)) + r'\\', r'\midrule']
    for row in rows:
        name = row['split'].replace('_', r'\_')
        values = ' & '.join(f"{row['counts'][m]['wer']:.2f} & {row['counts'][m]['cer']:.2f}" for m in models)
        lines.append(r'\texttt{' + name + '} & ' + f"{row['rows']:,}" + ' & ' + values + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


def main():
    original = ROOT / 'evidence/unseen-20260907'
    recent = ROOT / 'evidence/regression-ft-20260907'
    numeric_audit = read('evidence/regression-ft-20260907/numeric-audit.json')
    assert numeric_audit['status'] == 'pass'
    assert all(sha(recent / n) == digest for n, digest in numeric_audit['inputs'].items())
    reproduction = read('evidence/unseen-20260907/count-reproduction.json')
    assert reproduction['status'] == 'passed'
    inputs = ['evidence/regression-ft-20260907/numeric-audit.json',
              'evidence/regression-ft-20260907/comparison.json',
              'evidence/regression-ft-20260907/english-quick.json',
              'evidence/regression-ft-20260907/export-audit.json',
              'evidence/unseen-20260907/count-reproduction.json', 'release/model-stages.json']
    complete = []
    model_ids = None
    for phase, folder in [('primary', ''), ('coverage', 'coverage'), ('followup', 'alignment-followup')]:
        for filename, digest in reproduction['inputs'][phase].items():
            assert sha(original / folder / filename) == digest
        filename = str(Path('evidence/unseen-20260907') / folder / 'comparison.json')
        inputs.append(filename)
        source = read(filename)
        assert source['status'] == 'complete' and not source['missing_splits']
        if model_ids is None:
            model_ids = source['models']
        assert source['models'] == model_ids
        for name, spec in source['sets'].items():
            english = name in source['english_standard_sets']
            scored = source['english_standard_sets'].get(name, spec)
            complete.append(dict(split=name, rows=spec['rows'], hours=spec['hours'], english=english,
                                 normalization='standard_english' if english else 'legacy_multilingual',
                                 counts={'parakeet': scored['parakeet'], 'r15': scored['orukeet']}))
    stages = read('release/model-stages.json')
    historical_r15 = stages['report_snapshot']['gabor']['candidate_source_sha256']
    assert model_ids['orukeet'] == historical_r15
    comparison = read('evidence/regression-ft-20260907/comparison.json')
    assert comparison['models']['parakeet'] == model_ids['parakeet']
    assert comparison['models']['parent'] == historical_r15
    assert comparison['models']['candidate'] == stages['canonical']['source_sha256']
    assert comparison['models']['candidate'] == read('evidence/regression-ft-20260907/export-audit.json')['candidate_sha256']
    sampled = []
    for name, spec in comparison['sets'].items():
        english = 'standard_english' in spec
        scored = spec['standard_english' if english else 'legacy']
        sampled.append(dict(split=name, rows=spec['rows'], hours=spec['hours'], english=english,
                            normalization='standard_english' if english else 'legacy_multilingual',
                            counts={m: scored[k] for m, k in [('parakeet', 'parakeet'), ('r15', 'parent'), ('ft4035', 'candidate')]}))
    complete.sort(key=lambda r: r['split'])
    sampled.sort(key=lambda r: r['split'])
    assert len(complete) == len(sampled) == 47
    assert sum(r['rows'] for r in complete) == 327888
    assert sum(r['rows'] for r in sampled) == 12006
    assert {r['split'] for r in complete} == {r['split'] for r in sampled}
    summary = {}
    for name, rows, models in [('complete', complete, ['parakeet', 'r15']),
                               ('sampled', sampled, ['parakeet', 'r15', 'ft4035'])]:
        summary[name] = {}
        for group, members in [('all', rows), ('english', [r for r in rows if r['english']])]:
            summary[name][group] = dict(rows=sum(r['rows'] for r in members), splits=len(members),
                                        hours=sum(r['hours'] for r in members), models=rates(members, models))
    quick = read('evidence/regression-ft-20260907/english-quick.json')['groups']['all_english']['standard_english']
    for model, key in [('parakeet', 'parakeet'), ('r15', 'parent'), ('ft4035', 'candidate')]:
        assert summary['sampled']['english']['models'][model]['wer'] == quick[key]['wer']
    method = ('Scores use matched NeMo greedy decoding with FP32 weights and BF16 CUDA autocast. '
              'English uses standard Whisper text normalization; other languages use the recorded multilingual normalizer. '
              'WER and CER are percentages, computed from summed edit counts and reference lengths.'
              '\n\nThe current comparison covers 47 splits and 25 languages: 256 fixed clips per split and all 230 Lesbos clips, totaling 12,006 clips. '
              'FT-4035 was fine-tuned on 24 of these splits; 6,118 comparison clips were included in that run. '
              'Greek and Italian EuroSpeech use audited human transcript spans. '
              'The earlier 327,888-clip comparison measures R15-0100 against Parakeet, using its recorded clip membership and original EuroSpeech references. '
              'Full tables retain every split from both comparisons.')
    improvements = {}
    for group, members in [('all', sampled), ('english', [r for r in sampled if r['english']])]:
        models = summary['sampled'][group]['models']
        improvements[group] = dict(
            relative_wer_reduction_percent=100 * (1 - models['ft4035']['wer'] / models['parakeet']['wer']),
            improved_splits=sum(r['counts']['ft4035']['wer'] < r['counts']['parakeet']['wer'] for r in members),
            splits=len(members))
    report = dict(status='complete', publication_authorized=False,
                  models={'parakeet': model_ids['parakeet'], 'r15': historical_r15,
                          'ft4035': comparison['models']['candidate']},
                  method=method, summary=summary, improvements=improvements, complete=complete, sampled=sampled,
                  inputs_sha256={n: sha(ROOT / n) for n in inputs}, script_sha256=sha(Path(__file__)))
    out = ROOT / 'evidence/benchmark-release-20260907'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'scores.json').write_text(json.dumps(report, indent=2) + '\n')
    for name, rows, models in [('complete', complete, ['parakeet', 'r15']), ('sampled', sampled, ['parakeet', 'r15', 'ft4035'])]:
        with (out / (name + '.csv')).open('w', newline='') as f:
            columns = ['split', 'clips', 'hours', 'normalization'] + [m + '_' + metric for m in models for metric in ['wer', 'cer']]
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(split=row['split'], clips=row['rows'], hours=row['hours'], normalization=row['normalization'],
                                     **{m + '_' + metric: row['counts'][m][metric] for m in models for metric in ['wer', 'cer']}))
    summary_lines = ['| Matched comparison | Clips | Parakeet WER | R15-0100 WER | FT-4035 WER |', '|:--|--:|--:|--:|--:|']
    for group, title in [('all', 'All 47 splits · 25 languages'), ('english', 'All 20 English splits')]:
        s = summary['sampled'][group]
        summary_lines.append('| ' + title + f" | {s['rows']:,} | " + ' | '.join(f"{s['models'][m]['wer']:.2f}%" for m in ['parakeet', 'r15', 'ft4035']) + ' |')
    summary_md = '\n'.join(summary_lines)
    complete_md = markdown_table(complete, ['parakeet', 'r15'])
    sampled_md = markdown_table(sampled, ['parakeet', 'r15', 'ft4035'])
    model_note = ('Orukeet FT-4035 is the source of the current NeMo, Q8 and F16 downloads. '
                  f"Across the 25-language comparison, **Orukeet scores {summary['sampled']['all']['models']['ft4035']['wer']:.2f}% pooled WER versus Parakeet’s {summary['sampled']['all']['models']['parakeet']['wer']:.2f}%**, "
                  f"an {improvements['all']['relative_wer_reduction_percent']:.1f}% relative reduction. "
                  f"English WER is **{summary['sampled']['english']['models']['ft4035']['wer']:.2f}% for Orukeet versus {summary['sampled']['english']['models']['parakeet']['wer']:.2f}% for Parakeet**, "
                  f"a {improvements['english']['relative_wer_reduction_percent']:.1f}% relative reduction. "
                  f"It improves WER on {improvements['all']['improved_splits']} of 47 splits, including all 20 English splits. "
                  'R15-0100 is the preceding checkpoint; both retain the same 12,288 frozen Gabor kernels.')
    artifact_url = f"https://huggingface.co/oruk/orukeet/tree/{stages['gabor']['source_release_artifact']['revision']}"
    docs = '\n\n'.join(['# Orukeet benchmark scores', model_note, summary_md, '## Evaluation method', method,
                        '## Current matched comparison: 12,006 clips', 'Every split is listed below. Cells contain WER / CER (%).', sampled_md,
                        '## Preceding R15 comparison: 327,888 clips', 'Every split is listed below. Cells contain WER / CER (%).', complete_md,
                        '## Model identities', '\n'.join(f'- {m}: `{digest}`' for m, digest in report['models'].items()),
                        f'[FT-4035 checkpoint and audit]({artifact_url}) · [Machine-readable scores](../evidence/benchmark-release-20260907/scores.json) · [Complete-partition CSV](../evidence/benchmark-release-20260907/complete.csv) · [Matched-comparison CSV](../evidence/benchmark-release-20260907/sampled.csv)',
                        'The historical 23,038-recording selection results and format-specific native measurements are retained in the [methods companion](technical-report.md).']) + '\n'
    (ROOT / 'docs/benchmark-scores.md').write_text(docs)
    card = '\n\n'.join([model_note, summary_md, method,
                         '[All score files and methods](docs/benchmark-scores.md) · [FT-4035 checkpoint](' + artifact_url + ')',
                         '<details>\n<summary>All 47 splits: matched three-model WER / CER</summary>\n\n' + sampled_md + '\n\n</details>',
                         '<details>\n<summary>Preceding R15 comparison: all 47 complete partitions</summary>\n\n' + complete_md + '\n\n</details>']) + '\n\n'
    (out / 'card-section.md').write_text(card.rstrip() + '\n')
    for name in ['README.md', 'MODEL_CARD.md', 'hub/README.md',
                 'launch/publication/github-README.md', 'launch/publication/github-MODEL_CARD.md',
                 'launch/publication/huggingface-README.md']:
        path = ROOT / name
        if not path.exists():
            continue
        content = path.read_text()
        start = content.index('Orukeet FT-4035 is the source of the current NeMo, Q8 and F16 downloads.')
        end = content.index('### Recovery comparison', start)
        path.write_text(content[:start] + card + content[end:])
    companion = ROOT / 'docs/technical-report.md'
    content = companion.read_text()
    start = content.index('Orukeet FT-4035 is the source of the current NeMo, Q8 and F16 downloads.')
    end = content.index('The PDF appendix', start)
    brief = '\n\n'.join([model_note, summary_md, method,
                           '[All score files and methods](benchmark-scores.md) · [FT-4035 checkpoint](' + artifact_url + ')'])
    companion.write_text(content[:start] + brief + '\n\n' + content[end:])
    macros = {('SampleAll' if group == 'all' else 'SampleEnglish') + 'RelativeReduction':
              f"{values['relative_wer_reduction_percent']:.2f}" for group, values in improvements.items()}
    for suite, prefix in [('complete', 'Full'), ('sampled', 'Sample')]:
        for group, group_prefix in [('english', 'English'), ('all', 'All')]:
            s = summary[suite][group]
            macros[prefix + group_prefix + 'Rows'] = f"{s['rows']:,}"
            for m, values in s['models'].items():
                short = {'parakeet': 'Base', 'r15': 'R', 'ft4035': 'FT'}[m]
                for metric in ['wer', 'cer']:
                    macros[prefix + group_prefix + short + metric.upper()] = f'{values[metric]:.2f}'
    (ROOT / 'report/benchmark-values.tex').write_text('% Generated by scripts/build_benchmark_materials.py.\n' + ''.join('\\newcommand{\\' + k + '}{' + v + '}\n' for k, v in macros.items()))
    tables = [r'\clearpage', r'\appendix', r'\section{Complete benchmark scores}',
              tex_table(complete, ['parakeet', 'r15'], 'All 47 complete partitions, totaling 327,888 clips. WER and CER are percentages. English uses standard English normalization; other languages use the multilingual normalizer. Exact membership and reference diagnostics accompany the score files.', 'tab:full-benchmark'),
              r'\clearpage', r'\section{Complete matched-comparison scores}',
              tex_table(sampled, ['parakeet', 'r15', 'ft4035'], 'All 47 splits in the 12,006-clip matched comparison. Each split has 256 fixed clips, except Lesbos (230). WER and CER are percentages. Section~\\ref{sec:benchmarks} defines the model identities, sample, and reference treatment.', 'tab:sample-benchmark')]
    (ROOT / 'report/benchmark-tables.tex').write_text('\n'.join(tables) + '\n')
    print(json.dumps(dict(complete_rows=327888, sampled_rows=12006, splits=47, summary=summary)))


if __name__ == '__main__':
    main()
