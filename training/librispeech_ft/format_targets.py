"""Preserve native transcript formatting while enforcing reference word content."""
import argparse
from collections import Counter
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'evaluation/standard_asr'))
from scoring import normalized_pair


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(text):
    return normalized_pair(text, text, 'en')[0]


def sentence(text):
    text = text.lower()
    text = text[:1].upper() + text[1:]
    return text if text.endswith(('.', '!', '?')) else text + '.'


def correct(reference, prediction):
    if normalize(reference) == normalize(prediction):
        return prediction, 'native_lexically_exact'
    refs, hyps = reference.split(), prediction.split()
    matcher = SequenceMatcher(a=[normalize(w) for w in refs],
                              b=[normalize(w) for w in hyps], autojunk=False)
    result = []
    for tag, i, j, k, l in matcher.get_opcodes():
        if tag == 'equal':
            result.extend(hyps[k:l])
        elif tag in {'delete', 'replace'}:
            # SequenceMatcher's a side is the reference: delete means missing in hyp.
            words = [w.lower() for w in refs[i:j]]
            if words and k < l and hyps[k][:1].isupper():
                words[0] = words[0][:1].upper() + words[0][1:]
            if words and k < l:
                suffix = re.search(r'[,.!?;:]+["\u201d\u2019]*$', hyps[l-1])
                if suffix:
                    words[-1] += suffix.group()
            result.extend(words)
        # insert means an extra hypothesis word; omit it.
    text = ' '.join(result)
    if text:
        text = text[:1].upper() + text[1:]
        if text[-1] not in '.!?':
            text += '.'
    if normalize(text) != normalize(reference):
        return sentence(reference), 'reference_fallback'
    return text, 'aligned_word_correction'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    assert sha(a.manifest) == '8770c5a110b60d3adabf4a8e4826adb516664bd44a8c2352e1da57086f85e500'
    assert sha(a.predictions) == '434dfaf964ef74b6446513f641fa6a4c8c818dcf770de156a5ec206ab765c88b'
    rows = [r for r in map(json.loads, a.manifest.open()) if r['split'] == 'librispeech_test_other']
    predictions = {r['uid']: r for r in map(json.loads, a.predictions.open())}
    targets, methods = [], Counter()
    for row in rows:
        record = predictions[row['uid']]
        assert record['model_sha256'] == '0ccfefcd1894871cb0850bd3c464adf5397752840de2a76d1d2d075c4141a945'
        text, method = correct(row['text'], record['prediction'])
        assert text and normalize(text) == normalize(row['text'])
        methods[method] += 1
        targets.append(dict(uid=row['uid'], text=text, method=method,
                            reference_sha256=row['reference_sha256'],
                            formatting_prediction_sha256=hashlib.sha256(record['prediction'].encode()).hexdigest()))
    assert len(targets) == len({r['uid'] for r in targets}) == 2939
    target_path = a.output / 'targets.jsonl'
    target_path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in targets))
    audit = dict(status='pass', rows=len(targets), all_original_normalized_words_preserved=True,
                 methods=dict(methods), source_manifest_sha256=sha(a.manifest),
                 formatting_predictions_sha256=sha(a.predictions), targets_sha256=sha(target_path),
                 script_sha256=sha(Path(__file__)), scorer_sha256=sha(ROOT / 'evaluation/standard_asr/scoring.py'),
                 transformation='Preserve native casing and punctuation where aligned; correct reference words; fall back to the reference if normalized words differ.')
    (a.output / 'label-audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
