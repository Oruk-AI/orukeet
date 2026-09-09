"""Reproduce complete-test scores from edit counts and generate report tables."""
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'evidence/standard-asr-20260908'
LANGUAGES = dict(bg='Bulgarian', cs='Czech', da='Danish', de='German', el='Greek', en='English',
                 es='Spanish', et='Estonian', fi='Finnish', fr='French', hr='Croatian', hu='Hungarian',
                 it='Italian', lt='Lithuanian', lv='Latvian', mt='Maltese', nl='Dutch', pl='Polish',
                 pt='Portuguese', ro='Romanian', ru='Russian', sk='Slovak', sl='Slovenian',
                 sv='Swedish', uk='Ukrainian')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = json.loads((OUT/'comparison.json').read_text())
    prep = json.loads((OUT/'preparation.json').read_text())
    assert source['status'] == 'complete' and prep['status'] == 'prepared'
    canonical = json.loads((ROOT/'release/model-stages.json').read_text())['canonical']['source_sha256']
    assert source['models']['orukeet'] == canonical
    assert source['models']['parakeet'] == '3cbdc85877e668ca7b82d0d56770eb1fac76691f55d6b97545e8d61ca588d10d'
    assert source['manifest_sha256'] == prep['manifest_sha256']
    assert source['script_sha256'] == sha(ROOT/'evaluation/standard_asr/run.py')
    assert prep['script_sha256'] == sha(ROOT/'evaluation/standard_asr/prepare.py')
    assert source['normalizer_sha256'] == sha(ROOT/'evaluation/unseen/metrics.py')
    assert source['numeric_evidence_sha256'] == sha(OUT/'numeric-evidence.jsonl.gz')
    for name, digest in source['scoring']['code_sha256'].items():
        assert sha(ROOT/name) == digest, name
    audit = json.loads((OUT/'hypotheses-audit.json').read_text())
    assert audit['status'] == 'passed' and audit['models'] == source['models']
    assert audit['upstream_scoring_verified']['model_partition_pairs'] == 54
    assert audit['script_sha256'] == sha(ROOT/'evaluation/standard_asr/audit_predictions.py')
    for name, digest in audit['inputs_sha256'].items():
        assert sha(OUT/name) == digest, name
    expected = {f'fleurs_{lang}' for lang in LANGUAGES} | {'librispeech_test_clean','librispeech_test_other'}
    assert set(source['sets']) == set(prep['sources']) == expected
    totals, sizes, durations, seen = defaultdict(Counter), Counter(), Counter(), set()
    with gzip.open(OUT/'numeric-evidence.jsonl.gz', 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            assert row['record_sha256'] not in seen
            seen.add(row['record_sha256'])
            sizes[row['split']] += 1
            durations[row['split']] += row['duration']
            for model in ['parakeet','orukeet']:
                c = row['counts'][model]
                assert c['errors'] == c['substitutions'] + c['deletions'] + c['insertions']
                totals[(row['split'], model)].update(c)
            assert row['counts']['parakeet']['chars'] == row['counts']['orukeet']['chars']
    assert len(seen) == source['rows'] == prep['rows']
    for split, spec in source['sets'].items():
        assert sizes[split] == spec['rows'] == prep['sources'][split]['rows']
        assert math.isclose(durations[split]/3600, spec['hours'], abs_tol=1e-8)
        for model in source['models']:
            c = totals[(split, model)]
            assert all(spec['models'][model][k] == v for k,v in c.items())
            assert math.isclose(spec['models'][model]['wer'], 100*c['errors']/c['words'], abs_tol=1e-12)
            assert math.isclose(spec['models'][model]['cer'], 100*c['char_errors']/c['chars'], abs_tol=1e-12)
    summaries = {}
    for name, languages in [('fleurs_five', 'de es fr it pt'.split()), ('fleurs_all', list(LANGUAGES))]:
        members = [source['sets'][f'fleurs_{lang}'] for lang in languages]
        summaries[name] = dict(languages=languages, rows=sum(s['rows'] for s in members),
                               models={model:{metric:sum(s['models'][model][metric] for s in members)/len(members)
                                              for metric in ['wer','cer']} for model in source['models']})
    order = ['librispeech_test_clean','librispeech_test_other'] + [f'fleurs_{k}' for k in sorted(LANGUAGES,key=LANGUAGES.get)]
    names = {'librispeech_test_clean':'LibriSpeech test-clean', 'librispeech_test_other':'LibriSpeech test-other',
             **{f'fleurs_{k}':'FLEURS '+v for k,v in LANGUAGES.items()}}
    rows = [dict(split=s, label=names[s], **source['sets'][s]) for s in order]
    macros = {}
    for name, split in [('LibriClean','librispeech_test_clean'),('LibriOther','librispeech_test_other'),('FleursEnglish','fleurs_en')]:
        for model, prefix in [('parakeet','Parakeet'),('orukeet','Orukeet')]:
            macros[name+prefix+'WER'] = f"{source['sets'][split]['models'][model]['wer']:.2f}"
    for name, key in [('FleursFive','fleurs_five'),('FleursAll','fleurs_all')]:
        for model, prefix in [('parakeet','Parakeet'),('orukeet','Orukeet')]:
            macros[name+prefix+'WER'] = f"{summaries[key]['models'][model]['wer']:.2f}"
    macros['StandardTestRows'] = f"{source['rows']:,}"
    macros['FleursTestRows'] = f"{summaries['fleurs_all']['rows']:,}"
    (ROOT/'report/standard-benchmark-values.tex').write_text('% Generated from complete-test edit counts.\n'+''.join(
        '\\newcommand{\\'+key+'}{'+value+'}\n' for key,value in macros.items()))
    table = [r'\begin{table}[!ht]',r'\centering',
             r'\caption{Complete LibriSpeech and FLEURS test partitions. Both models use matched NeMo greedy decoding; WER and CER are percentages. Macro rows weight languages equally. Lower is better.}',
             r'\label{tab:standard-benchmarks}',r'\small',r'\setlength{\tabcolsep}{5pt}',
             r'\renewcommand{\arraystretch}{1.05}',r'\begin{tabular}{lrrrrr}',r'\toprule',
             r'Benchmark & Clips & \multicolumn{2}{c}{Parakeet} & \multicolumn{2}{c}{Orukeet}\\',
             r' & & WER & CER & WER & CER\\',r'\midrule']
    md = ['| Benchmark | Clips | Parakeet WER / CER | Orukeet WER / CER |','|:--|--:|--:|--:|']
    for row in rows:
        values = [f"{row['models'][model][metric]:.2f}" for model in ['parakeet','orukeet'] for metric in ['wer','cer']]
        table.append(row['label']+' & '+f"{row['rows']:,}"+' & '+' & '.join(values)+r'\\')
        md.append('| '+row['label']+' | '+f"{row['rows']:,}"+' | '+values[0]+' / '+values[1]+' | '+values[2]+' / '+values[3]+' |')
        if row['split'] == 'librispeech_test_other': table.append(r'\midrule')
    table.append(r'\midrule')
    for key, label in [('fleurs_five','FLEURS five-language macro'),('fleurs_all','FLEURS 25-language macro')]:
        row=summaries[key]
        values=[f"{row['models'][model][metric]:.2f}" for model in ['parakeet','orukeet'] for metric in ['wer','cer']]
        table.append(label+' & '+f"{row['rows']:,}"+' & '+' & '.join(values)+r'\\')
        md.append('| '+label+' | '+f"{row['rows']:,}"+' | '+values[0]+' / '+values[1]+' | '+values[2]+' / '+values[3]+' |')
    table += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (ROOT/'report/standard-benchmark-table.tex').write_text('\n'.join(table)+'\n')
    with (OUT/'scores.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['split','clips','hours','parakeet_wer','parakeet_cer','orukeet_wer','orukeet_cer'])
        for row in rows:writer.writerow([row['split'],row['rows'],row['hours']]+[row['models'][m][x] for m in ['parakeet','orukeet'] for x in ['wer','cer']])
    method = ('Both checkpoints use identical mono 16 kHz audio, NeMo greedy-batch TDT decoding, FP32 weights and BF16 CUDA autocast. '
              'English uses the pinned English text normalizer. Multilingual normalization retains diacritics and expands numbers by language. '
              'WER aligns compound boundaries, then uses compound-aware edit distance; CER measures normalized strings before boundary alignment. '
              'WER and CER pool integer edit counts and reference lengths within each test partition. Every test record is retained, including empty hypotheses. '
              'The five-language FLEURS macro averages German, Spanish, French, Italian and Portuguese; the 25-language macro includes every supported language.')
    doc = '# LibriSpeech and FLEURS benchmarks\n\n'+method+'\n\n'+'\n'.join(md)+'\n\n'+(
        '[Evaluation and reproduction](../evaluation/standard_asr/README.md) · '
        '[Full-precision scores](../evidence/standard-asr-20260908/scores.csv) · '
        '[Per-record edit counts](../evidence/standard-asr-20260908/numeric-evidence.jsonl.gz)\n')
    (ROOT/'docs/standard-asr-benchmarks.md').write_text(doc)
    receipt = dict(status='passed', publication_authorized=False, rows=len(seen), splits=len(rows),
                   model_sha256=canonical, summaries=summaries, checks=['Unique complete test membership','All paired integer counts','WER and CER recomputation','Pinned checkpoint and source hashes'],
                   inputs_sha256={name:sha(OUT/name) for name in ['comparison.json','preparation.json','numeric-evidence.jsonl.gz','hypotheses-audit.json']},
                   outputs_sha256={name:sha(ROOT/name) for name in ['report/standard-benchmark-values.tex','report/standard-benchmark-table.tex','docs/standard-asr-benchmarks.md','evidence/standard-asr-20260908/scores.csv']})
    if '--verify-pdf' in sys.argv:
        text=subprocess.check_output(['pdftotext','-layout',str(ROOT/'output/pdf/orukeet-technical-report.pdf'),'-'],text=True)
        lines={' '.join(line.split()) for line in text.splitlines()}
        for row in rows:
            values=[f"{row['models'][model][metric]:.2f}" for model in ['parakeet','orukeet'] for metric in ['wer','cer']]
            assert ' '.join([row['label'],f"{row['rows']:,}"]+values) in lines, row['split']
        normalized_lines={line.replace('\ufb01','fi') for line in lines}
        for key,label in [('fleurs_five','FLEURS five-language macro'),('fleurs_all','FLEURS 25-language macro')]:
            summary=summaries[key]
            values=[f"{summary['models'][model][metric]:.2f}" for model in ['parakeet','orukeet'] for metric in ['wer','cer']]
            assert ' '.join([label,f"{summary['rows']:,}"]+values) in normalized_lines,key
        import re
        abstract=re.split(r'\n\s*1\s+Introduction',text.split('Abstract',1)[1],maxsplit=1)[0]
        assert 'Open ASR' not in text and 'leaderboard' not in text.lower()
        for key in ['LibriClean','LibriOther','FleursEnglish','FleursFive']:
            assert macros[key+'ParakeetWER'] in abstract and macros[key+'OrukeetWER'] in abstract
        receipt['checks'].append('Rendered abstract and all 27 test table rows match measurements')
    (OUT/'materials-validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['inputs_sha256','outputs_sha256','summaries']}))


if __name__ == '__main__':
    main()
