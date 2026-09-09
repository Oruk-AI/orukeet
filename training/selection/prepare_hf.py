#!/usr/bin/env python3
"""Prepare source-held-out recordings without loading model predictions."""
import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import soxr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_prepare import rows, text_key, digest


def rank(value):
    return hashlib.sha256(('goal-v2:' + str(value)).encode()).hexdigest()


def balanced(pool, cap):
    groups = defaultdict(list)
    for key, row in pool.items():
        groups[row['speaker_id']].append(key)
    order = sorted(groups, key=rank)
    for group in order:
        groups[group].sort(key=rank)
    result = []
    for offset in range(max(map(len, groups.values()), default=0)):
        for group in order:
            if offset < len(groups[group]):
                result.append(groups[group][offset])
                if len(result) == cap:
                    return set(result)
    return set(result)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--source', choices=['english', 'voxpopuli'], required=True)
    args = p.parse_args()
    root = args.root
    source = root / 'data/goal_v2_sources' / args.source
    assert (source / '_READY').is_file()
    receipt = json.loads((source / 'source_receipt.json').read_text())
    out = root / 'manifests/goal_v2_confirmation'
    audio = source / 'normalized'
    audio.mkdir(exist_ok=True)
    exposed_speakers, exposed_pcm = set(), set()
    exposed_text = defaultdict(set)
    for path in sorted((root/'manifests/final').glob('*_train.json')):
        for row in rows(path):
            exposed_text[row.get('lang','en')].add(text_key(row['text']))
    for path in sorted((root/'data/accent_extension_20260905').glob('*.jsonl')):
        for row in rows(path):
            if row['source_id'] == 'english_dialects':
                exposed_speakers.add(row['speaker_id'])
            exposed_text['en'].add(text_key(row['text']))
    for path in sorted((root/'manifests/accent_extension_20260905').glob('*.jsonl')):
        for row in rows(path):
            exposed_pcm.add(row['pcm_sha256'])
    pool, counts = defaultdict(dict), defaultdict(Counter)
    files = sorted(p for p in source.glob('*/*.parquet') if 'normalized' not in p.parts)
    seen_ids = set()
    for path in files:
        config = path.parent.name
        is_english = args.source == 'english'
        lang = 'en' if is_english or config == 'en_accented' else config
        name = 'dialect_en_english_dialects_source_holdout' if is_english else (
            'parliamentaccent_en_voxpopuli_confirmation' if config == 'en_accented' else f'voxpopuli_{lang}_confirmation')
        pf = pq.ParquetFile(path)
        columns = [c for c in pf.schema_arrow.names if c != 'audio']
        index = 0
        for batch in pf.iter_batches(columns=columns):
            for row in batch.to_pylist():
                key = (str(path), index)
                index += 1
                counts[name]['input_rows'] += 1
                speaker = str(row.get('speaker_id', ''))
                if speaker.lower() in ['', 'none', 'unknown', 'null', 'na']:
                    counts[name]['excluded_missing_speaker'] += 1
                    continue
                speaker = f'english_dialects:{config}:speaker:{speaker}' if is_english else 'voxpopuli:' + speaker
                if speaker in exposed_speakers:
                    counts[name]['excluded_exposed_speaker'] += 1
                    continue
                text = row['text'] if is_english else row['normalized_text']
                overlap = text_key(text) in exposed_text[lang]
                if not is_english and (not row['is_gold_transcript'] or overlap):
                    counts[name]['excluded_nongold_or_exposed_text'] += 1
                    continue
                uid = f'{config}:{row["speaker_id"]}:{row["line_id"]}:{index}' if is_english else row['audio_id']
                if uid in seen_ids:
                    counts[name]['duplicate_source_id'] += 1
                    continue
                seen_ids.add(uid)
                pool[name][key] = {'text':text, 'lang':lang, 'src':'english_dialects' if is_english else 'voxpopuli',
                    'speaker_id':speaker, 'source_row_id':uid, 'source_config':config,
                    'accent':config.rsplit('_',1)[0] if is_english else row.get('accent','unknown'),
                    'transcript_seen_in_prior_campaign':overlap,
                    'source_repository':receipt['repository'], 'source_revision':receipt['revision'],
                    'transcript_provenance':'published_human_source_transcript',
                    'source_split':'train_source_speaker_holdout' if is_english else 'test',
                    'rights_lane':'conditional_contract' if is_english else 'evaluation_only_source_license'}
    chosen = {}
    for name, candidates in pool.items():
        # Dialect holdout uses every eligible new source speaker. Parliamentary
        # selection cycles through speakers, independently of all predictions.
        keys = set(candidates) if args.source == 'english' else balanced(candidates, 1200 if name.startswith('parliamentaccent') else 600)
        chosen.update({key:(name,candidates[key]) for key in keys})
    records = defaultdict(list)
    seen_pcm = set(exposed_pcm)
    for path in files:
        if not any(key[0] == str(path) for key in chosen):
            continue
        index = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=16):
            for raw in batch.to_pylist():
                key = (str(path), index)
                index += 1
                if key not in chosen:
                    continue
                name, row = chosen[key]
                data, sr = sf.read(io.BytesIO(raw['audio']['bytes']), dtype='float32', always_2d=True)
                if not np.isfinite(data).all() or not np.any(data):
                    counts[name]['excluded_invalid_audio'] += 1
                    continue
                data = data.mean(axis=1)
                if sr != 16000:
                    data = soxr.resample(data, sr, 16000)
                duration = len(data)/16000
                if not 0.4 <= duration <= 60:
                    counts[name]['excluded_duration_outside_0_4_to_60_seconds'] += 1
                    continue
                pcm = (np.clip(data,-1,1)*32767).astype('<i2')
                sha = hashlib.sha256(pcm.tobytes()).hexdigest()
                if sha in seen_pcm:
                    counts[name]['excluded_duplicate_pcm'] += 1
                    continue
                seen_pcm.add(sha)
                dest = audio / (sha + '.flac')
                sf.write(dest, pcm, 16000, subtype='PCM_16')
                records[name].append(dict(row,audio_filepath=str(dest),duration=duration,
                    sample_rate=16000,channels=1,pcm_sha256=sha))
    report = {'source':receipt, 'sets':{}, 'selection_seed':'goal-v2',
        'limitations':['English Dialects numeric speaker IDs are scoped to source configurations, based on observed cross-configuration collisions',
                      'Cross-corpus real-world speaker identity is not certified',
                      'PCM deduplication covers the recovered accent pool and this source, not upstream pretraining',
                      'Maximum duration 60 seconds; no model-score-based filtering']}
    for name, data in sorted(records.items()):
        path = out / (name + '.jsonl')
        with path.open('x') as f:
            for row in sorted(data,key=lambda r:rank(r['source_row_id'])):
                f.write(json.dumps(row,ensure_ascii=False)+'\n')
        report['sets'][name] = dict(counts[name],rows=len(data),groups=len({r['speaker_id'] for r in data}),
            hours=sum(r['duration'] for r in data)/3600,transcript_overlap_rows=sum(r['transcript_seen_in_prior_campaign'] for r in data),
            sha256=digest(path))
        print('HF_MANIFEST_READY',name,json.dumps(report['sets'][name]),flush=True)
    (out / (args.source + '_source_registry.json')).write_text(json.dumps(report,indent=2)+'\n')
    print('HF_PREPARATION_COMPLETE',args.source,flush=True)


if __name__ == '__main__':
    main()
