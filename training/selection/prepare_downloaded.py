#!/usr/bin/env python3
"""Prepare source-held-out English data and a general-English development set."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_prepare import rows, text_key, digest
import soundfile as sf


def extract(archive, destination):
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            path = (destination / member.name).resolve()
            if not path.is_relative_to(destination.resolve()) or not (member.isfile() or member.isdir()):
                raise ValueError('Unsafe archive member: ' + member.name)
        tar.extractall(destination, filter='data')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    root = args.root
    source = root / 'data/goal_v2_sources'
    dev = root / 'manifests/goal_v2_dev'
    confirm = root / 'manifests/goal_v2_confirmation'
    dev.mkdir(exist_ok=True)
    if not (confirm / 'gate_registry.json').is_file():
        raise RuntimeError('Freeze the main gate first')
    registry = {}
    for split, target, name in [('dev-other', dev, 'libri_en_dev_other'),
                                 ('test-clean', confirm, 'core_en_libri_test_clean')]:
        path = target / (name + '.jsonl')
        if path.exists():
            raise FileExistsError(path)
        extract(source / (split + '.tar.gz'), source / 'libri')
        data = []
        for transcript in sorted((source / 'libri/LibriSpeech' / split).rglob('*.trans.txt')):
            for line in transcript.read_text().splitlines():
                uid, text = line.split(' ', 1)
                audio = transcript.parent / (uid + '.flac')
                info = sf.info(audio)
                if info.samplerate != 16000 or info.channels != 1:
                    raise ValueError('Unexpected LibriSpeech audio format')
                data.append({'audio_filepath': str(audio), 'text': text, 'duration': info.duration,
                             'lang': 'en', 'src': 'librispeech', 'source_split': split,
                             'speaker_id': 'librispeech:' + uid.split('-')[0], 'utterance_id': uid,
                             'sample_rate': 16000, 'channels': 1, 'accent': 'unknown',
                             'transcript_provenance': 'official_human_transcript'})
        with path.open('x') as f:
            for r in data:
                f.write(json.dumps(r) + '\n')
        registry[name] = {'rows': len(data), 'groups': len({r['speaker_id'] for r in data}),
                          'hours': sum(r['duration'] for r in data) / 3600, 'sha256': digest(path)}
        print('MANIFEST_READY', name, registry[name], flush=True)
    extract(source / 'speechocean762.tar.gz', source)
    speech = source / 'speechocean762'
    seen_speakers = set()
    seen_texts = set()
    for path in (root / 'data/accent_extension_20260905').glob('*.jsonl'):
        for r in rows(path):
            if r['source_id'] == 'speechocean762':
                seen_speakers.add(r['split_group'].rsplit(':', 1)[-1])
                seen_texts.add(text_key(r['text']))
    wavs = dict(line.split(maxsplit=1) for line in (speech / 'test/wav.scp').read_text().splitlines())
    data = []
    excluded = 0
    for line in (speech / 'test/text').read_text().splitlines():
        uid, text = line.split(maxsplit=1)
        audio = speech / wavs[uid]
        speaker = audio.parent.name.removeprefix('SPEAKER')
        if speaker in seen_speakers:
            excluded += 1
            continue
        info = sf.info(audio)
        if info.samplerate != 16000 or info.channels != 1:
            raise ValueError('Unexpected SpeechOcean audio format')
        data.append({'audio_filepath': str(audio), 'text': text, 'duration': info.duration,
                     'lang': 'en', 'src': 'speechocean762', 'source_split': 'official_test',
                     'speaker_id': 'speechocean762:' + speaker, 'utterance_id': uid,
                     'sample_rate': 16000, 'channels': 1, 'accent': 'mandarin_l1_english',
                     'transcript_seen_in_prior_source_partitions': text_key(text) in seen_texts,
                     'transcript_provenance': 'official_pronunciation_assessment_transcript'})
    path = confirm / 'learner_en_speechocean_official_test.jsonl'
    with path.open('x') as f:
        for r in data:
            f.write(json.dumps(r) + '\n')
    registry[path.stem] = {'rows': len(data), 'groups': len({r['speaker_id'] for r in data}),
                           'excluded_previously_seen_speaker_rows': excluded,
                           'transcript_overlap_rows': sum(r['transcript_seen_in_prior_source_partitions'] for r in data),
                           'hours': sum(r['duration'] for r in data) / 3600, 'sha256': digest(path)}
    (confirm / 'english_external_registry.json').write_text(json.dumps(registry, indent=2) + '\n')
    print('ENGLISH_SOURCES_PREPARED', registry, flush=True)


if __name__ == '__main__':
    main()
