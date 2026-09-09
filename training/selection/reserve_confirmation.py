#!/usr/bin/env python3
"""Freeze new confirmation utterances without reading any model predictions."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_prepare import rows, text_key, identity, sentence, digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    root = args.root
    original = root / 'manifests/final'
    audited = root / 'manifests/audited_20260905'
    out = root / 'manifests/goal_v2_confirmation'
    out.mkdir(parents=True, exist_ok=False)
    texts, clips, sentences = defaultdict(set), defaultdict(set), defaultdict(set)
    cv_speakers = set()
    exposed = (list(original.glob('*_train.json')) + list(original.glob('*_test.json')) +
               list(original.glob('val_*_dev_sub.json')) +
               list(audited.glob('*_dev_selection.jsonl')) +
               list((root / 'manifests/confirmation_20260905').glob('*.jsonl')) +
               list((root / 'manifests/accent_extension_20260905').glob('*.jsonl')))
    # The recovered pilot used the complete Greek FLEURS development split.
    exposed.append(original / 'fleurs_el_dev.json')
    for path in exposed:
        if not path.is_file():
            continue
        for r in rows(path):
            lang = r.get('lang', 'en')
            texts[lang].add(text_key(r['text']))
            clips[lang].add(identity(r))
            if sentence(r):
                sentences[lang].add((r.get('src'), sentence(r)))
            if r.get('client_id'):
                cv_speakers.add(r['client_id'])
    report = {}
    english_pool = []
    selected_english_speakers = set()
    for path in sorted(audited.glob('*_dev.jsonl')):
        src, lang, _ = path.stem.split('_')
        count = Counter()
        pool = []
        for r in rows(path):
            count['input_rows'] += 1
            if src == 'cv' and (not r.get('speaker_id') or r['speaker_id'] in cv_speakers):
                count['exposed_or_missing_speaker'] += 1
                continue
            if (text_key(r['text']) in texts[lang] or identity(r) in clips[lang] or
                    (r.get('src'), sentence(r)) in sentences[lang]):
                count['exposed_utterance_or_sentence'] += 1
                continue
            rank = hashlib.sha256(('goal-v2:' + r['audio_filepath']).encode()).hexdigest()
            pool.append((rank, r))
        selected = [r for _, r in sorted(pool)[:400 if src == 'cv' else 250]]
        groups = {r.get('speaker_id') or ('sentence:' + str(r.get('fleurs_id'))) for r in selected}
        name = f'{src}_{lang}_confirmation'
        dest = out / (name + '.jsonl')
        with dest.open('w') as f:
            for r in selected:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        report[name] = dict(count, eligible_rows=len(pool), rows=len(selected), groups=len(groups),
                            primary_eligible=len(groups) >= 20, sha256=digest(dest),
                            grouping='speaker' if src == 'cv' else 'parallel_sentence_id_not_speaker')
        if src == 'cv' and lang == 'en':
            english_pool = [r for _, r in sorted(pool)]
            selected_english_speakers = {r['speaker_id'] for r in selected}
        print(name, json.dumps(report[name]), flush=True)
    accents = defaultdict(list)
    for r in english_pool:
        if r['speaker_id'] not in selected_english_speakers:
            accents[r.get('accent') or 'unknown'].append(r)
    accent_report = {}
    for label, pool in sorted(accents.items()):
        if label == 'unknown' or len({r['speaker_id'] for r in pool}) < 20:
            continue
        selected = pool[:200]
        if sum(len(r['text'].split()) for r in selected) < 500:
            continue
        label_hash = hashlib.sha256(label.encode()).hexdigest()[:12]
        dest = out / f'accent_en_{label_hash}_confirmation.jsonl'
        with dest.open('w') as f:
            for r in selected:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        accent_report[dest.stem] = {'label': label, 'rows': len(selected),
                                   'groups': len({r['speaker_id'] for r in selected}),
                                   'sha256': digest(dest)}
    registry = {'status': 'sealed_before_model_selection', 'source_language_sets': report,
                'english_accent_sets': accent_report, 'exposure_manifest_paths': [str(p) for p in exposed],
                'limitations': ['FLEURS speaker IDs are unavailable; sentence-level exclusion and clustering only',
                               'Cross-language parallel FLEURS texts may have been seen in another language',
                               'No claim about upstream stock-model pretraining',
                               'CV speaker IDs are inherited truncated hashes',
                               'No complete acoustic near-duplicate certification']}
    (out / 'gate_registry.json').write_text(json.dumps(registry, indent=2, ensure_ascii=False) + '\n')
    print('GOAL_V2_GATE_SEALED', len(accent_report), 'annotated English accent slices', flush=True)


if __name__ == '__main__':
    main()
