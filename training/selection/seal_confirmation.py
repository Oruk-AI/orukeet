#!/usr/bin/env python3
"""Freeze all metric membership using manifests alone, before inference."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


MULTILINGUAL = ['cv','fleurs','voxpopuli','ftspeech','rixvox']


def cluster(row):
    source = row['src']
    if source == 'fleurs':
        if not row.get('fleurs_id'):
            raise ValueError('Missing FLEURS sentence identity')
        return 'fleurs:parallel_sentence:' + str(row['fleurs_id'])
    speaker = row.get('speaker_id') or row.get('client_id')
    if not speaker or str(speaker).lower() in ['none','unknown']:
        raise ValueError('Missing speaker identity')
    return source + ':speaker:' + str(speaker)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    args = p.parse_args()
    root = args.root
    folder = root/'manifests/goal_v2_confirmation'
    protocol_path = root/'goal_v2/protocol.json'
    protocol = json.loads(protocol_path.read_text())
    rows_by_set, sets = {}, {}
    all_audio = set()
    for path in sorted(folder.glob('*.jsonl')):
        data = [json.loads(line) for line in path.open() if line.strip()]
        if not data:
            continue
        srcs,langs = {r['src'] for r in data},{r['lang'] for r in data}
        assert len(srcs) == len(langs) == 1
        for row in data:
            audio = row['audio_filepath']
            if audio in all_audio:
                raise ValueError('Duplicate recording in confirmation: '+audio)
            all_audio.add(audio)
            if not Path(audio).is_file():
                raise FileNotFoundError(audio)
        source,lang = next(iter(srcs)),next(iter(langs))
        groups = {cluster(r) for r in data}
        rows_by_set[path.stem] = data
        sets[path.stem] = {'path':str(path),'source':source,'language':lang,'rows':len(data),
            'hours':sum(r['duration'] for r in data)/3600,'groups':len(groups),
            'manifest_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'multilingual_stratum':source in MULTILINGUAL and path.stem.startswith(source+'_')}
    primary_languages = {}
    families = defaultdict(list)
    contributing = defaultdict(list)
    min_source = protocol['confirmation']['minimum_groups_for_contributing_source_language']
    for name, entry in sets.items():
        if entry['multilingual_stratum'] and entry['groups'] >= min_source:
            contributing[entry['language']].append(name)
            families[entry['source']].append(name)
    for lang,names in sorted(contributing.items()):
        groups = {cluster(r) for name in names for r in rows_by_set[name]}
        if len(groups) >= protocol['confirmation']['minimum_groups_for_primary_language']:
            primary_languages[lang] = {'sets':names,'groups':len(groups)}
    english = {
        'common_voice':[name for name,s in sets.items() if s['source']=='cv' and s['language']=='en'],
        'fleurs':['fleurs_en_confirmation'],
        'voxpopuli':['voxpopuli_en_confirmation'],
        'voxpopuli_accented':['parliamentaccent_en_voxpopuli_confirmation'],
        'core':['core_en_libri_test_clean'],
        'learner':['learner_en_speechocean_official_test'],
        'dialect':['dialect_en_english_dialects_source_holdout']}
    for corpus,names in english.items():
        if not names or any(name not in sets for name in names):
            raise ValueError('Missing English corpus '+corpus)
    if len(primary_languages) < 20:
        raise ValueError('Insufficient primary languages: '+str(primary_languages))
    if set(families) != set(MULTILINGUAL):
        raise ValueError('Incomplete source families')
    registry = {'status':'sealed_before_confirmation_inference','sets':sets,
        'primary_languages':primary_languages,'families':dict(families),'english_corpora':english,
        'total_rows':sum(s['rows'] for s in sets.values()),'total_hours':sum(s['hours'] for s in sets.values()),
        'protocol_sha256':hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        'confirmation_predictions_observed':False}
    dest = folder/'sealed_metric_registry.json'
    with dest.open('x') as f:
        json.dump(registry,f,indent=2)
        f.write('\n')
    print('CONFIRMATION_SEALED',json.dumps({'rows':registry['total_rows'],'hours':registry['total_hours'],
        'primary_languages':list(primary_languages),'registry_sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}),flush=True)


if __name__ == '__main__':
    main()
