"""Score fixed private transcripts with pinned compound-aware normalization."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from scoring import counts

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--private-records', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, default=ROOT/'evidence/standard-asr-20260908')
    a = parser.parse_args()
    source = json.loads((a.evidence/'inference-comparison.json').read_text())
    assert sha(a.private_records/'manifest.jsonl') == source['manifest_sha256']
    rows = [json.loads(line) for line in (a.private_records/'manifest.jsonl').open()]
    predictions = {}
    for model in source['models']:
        path = a.private_records/(model+'.jsonl')
        assert sha(path) == source['prediction_sha256'][model]
        records = [json.loads(line) for line in path.open()]
        predictions[model] = {r['uid']:r['prediction'] for r in records}
        assert len(records) == len(predictions[model]) == len(rows)
    totals, seen = defaultdict(Counter), set()
    with gzip.open(a.evidence/'numeric-evidence.jsonl.gz', 'wt') as stream:
        for row in rows:
            assert row['uid'] not in seen
            seen.add(row['uid'])
            values = {model:counts(row['text'], predictions[model][row['uid']], row['language'])
                      for model in source['models']}
            for model, value in values.items():
                totals[(row['split'], model)].update(value)
            stream.write(json.dumps(dict(record_sha256=hashlib.sha256(row['uid'].encode()).hexdigest(),
                                         split=row['split'], language=row['language'],
                                         duration=row['duration'], counts=values))+'\n')
    assert len(seen) == source['rows']
    for split, spec in source['sets'].items():
        spec['models'] = {model:dict(totals[(split,model)],
                                    wer=100*totals[(split,model)]['errors']/totals[(split,model)]['words'],
                                    cer=100*totals[(split,model)]['char_errors']/totals[(split,model)]['chars'])
                          for model in source['models']}
    paths = [Path(__file__), ROOT/'evaluation/standard_asr/scoring.py']
    paths += [p for p in (ROOT/'evaluation/standard_asr/vendor').iterdir() if p.is_file()]
    source['scoring'] = dict(
        protocol='Pinned English and multilingual normalization; compound-aware WER',
        dependencies={name:version(name) for name in ['kaldialign','num2words','regex','rapidfuzz']},
        code_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in paths},
        reference_word_counts='Non-English word-boundary alignment may change reference token counts separately for each model.',
        cer='Character distance before compound alignment, including normalized spaces.',
        inference_comparison_sha256=sha(a.evidence/'inference-comparison.json'))
    source['normalizers'] = dict(en='Pinned EnglishTextNormalizer with spelling/name/compound maps',
                                other='Pinned MultilingualNormalizer; diacritics retained; language-specific numbers; compound-boundary alignment')
    source['numeric_evidence_sha256'] = sha(a.evidence/'numeric-evidence.jsonl.gz')
    (a.evidence/'comparison.json').write_text(json.dumps(source,indent=2,ensure_ascii=False)+'\n')
    for split in ['librispeech_test_clean','librispeech_test_other','fleurs_en','fleurs_de','fleurs_es','fleurs_fr','fleurs_it','fleurs_pt']:
        if split in source['sets']:
            print(split,{m:round(v['wer'],3) for m,v in source['sets'][split]['models'].items()})


if __name__ == '__main__':
    main()
