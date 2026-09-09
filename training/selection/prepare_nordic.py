#!/usr/bin/env python3
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import soxr

from prepare_hf import balanced, rank
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit_prepare import rows, text_key, digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    args = p.parse_args()
    root = args.root
    folders = {name:root/'data/goal_v2_sources'/name for name in ['ftspeech','rixvox','rixvox_dev']}
    assert all((p/'_READY').is_file() for p in folders.values())
    out = root/'manifests/goal_v2_confirmation'
    exposed = {lang:set() for lang in ['da','sv']}
    for lang in exposed:
        for path in (root/'manifests/final').glob('*_'+lang+'_train.json'):
            exposed[lang].update(text_key(r['text']) for r in rows(path))
    report = {'sources':{name:json.loads((folder/'source_receipt.json').read_text()) for name,folder in folders.items()},'sets':{},
        'limitations':['RixVox references are human parliamentary protocols, automatically aligned, and may depart from verbatim speech.',
                      'RixVox official development and test are both previously unused for this campaign and are combined prospectively for speaker coverage.',
                      'No upstream pretraining or cross-corpus real-world speaker decontamination claim.']}
    for source, lang in [('ftspeech','da'),('rixvox','sv')]:
        pool = {}
        counts = Counter()
        if source == 'ftspeech':
            files = sorted(folders[source].glob('data/*.parquet'))
            for path in files:
                pf = pq.ParquetFile(path)
                cols = [c for c in pf.schema_arrow.names if c != 'audio']
                for index,row in enumerate(pf.read(columns=cols).to_pylist()):
                    counts['input_rows'] += 1
                    if not row['speaker_id'] or text_key(row['sentence']) in exposed[lang]:
                        counts['excluded_missing_speaker_or_training_text'] += 1
                        continue
                    pool[(str(path),index)] = {'speaker_id':'ftspeech:'+row['speaker_id'],'text':row['sentence'],
                        'source_row_id':row['utterance_id'],'source_split':path.name.split('-')[0],
                        'gender':row['speaker_gender'],'transcript_provenance':'human_parliamentary_transcript'}
        else:
            for part in ['rixvox','rixvox_dev']:
                for path in folders[part].glob('data/*_metadata.parquet'):
                    for row in pq.read_table(path).to_pylist():
                        counts['input_rows'] += 1
                        speaker = row['intressent_id']
                        if not speaker or not row['speaker_from_id'] or text_key(row['text']) in exposed[lang]:
                            counts['excluded_missing_speaker_or_training_text'] += 1
                            continue
                        pool[row['filename']] = {'speaker_id':'rixvox:'+speaker,'text':row['text'],
                            'source_row_id':row['filename'],'source_split':path.name.split('_')[0],
                            'gender':row['gender'],'transcript_provenance':'human_parliamentary_protocol_automatic_alignment_not_always_verbatim'}
        chosen = balanced(pool, 600)
        records = []
        audio_dir = folders[source]/'normalized'
        audio_dir.mkdir(exist_ok=True)
        seen = set()
        def convert(key, blob):
            row = pool[key]
            data,sr = sf.read(io.BytesIO(blob),dtype='float32',always_2d=True)
            if not np.isfinite(data).all() or not np.any(data):
                counts['excluded_invalid_audio'] += 1
                return
            data = data.mean(axis=1)
            if sr != 16000:
                data = soxr.resample(data,sr,16000)
            duration = len(data)/16000
            if not 0.4 <= duration <= 60:
                counts['excluded_duration'] += 1
                return
            pcm = (np.clip(data,-1,1)*32767).astype('<i2')
            sha = hashlib.sha256(pcm.tobytes()).hexdigest()
            if sha in seen:
                counts['excluded_duplicate_pcm'] += 1
                return
            seen.add(sha)
            dest = audio_dir/(sha+'.flac')
            sf.write(dest,pcm,16000,subtype='PCM_16')
            records.append(dict(row,lang=lang,src=source,accent='unknown',audio_filepath=str(dest),duration=duration,
                pcm_sha256=sha,sample_rate=16000,channels=1,source_repository=report['sources'][source]['repository'],
                source_revision=report['sources'][source]['revision'],rights_lane='evaluation_only_source_license'))
        if source == 'ftspeech':
            for path in files:
                index = 0
                for batch in pq.ParquetFile(path).iter_batches(batch_size=16):
                    for row in batch.to_pylist():
                        key = (str(path),index)
                        index += 1
                        if key in chosen:
                            convert(key,row['audio']['bytes'])
        else:
            found = set()
            for part in ['rixvox','rixvox_dev']:
                for path in sorted(folders[part].rglob('*.tar.gz')):
                    with tarfile.open(path,mode='r|gz') as tar:
                        for member in tar:
                            if member.isfile() and member.name in chosen:
                                convert(member.name,tar.extractfile(member).read())
                                found.add(member.name)
            if found != chosen:
                raise ValueError('Missing selected RixVox audio: '+str(len(chosen-found)))
        name = f'{source}_{lang}_confirmation'
        path = out/(name+'.jsonl')
        with path.open('x') as f:
            for row in sorted(records,key=lambda r:rank(r['source_row_id'])):
                f.write(json.dumps(row,ensure_ascii=False)+'\n')
        report['sets'][name] = dict(counts,rows=len(records),groups=len({r['speaker_id'] for r in records}),
            hours=sum(r['duration'] for r in records)/3600,sha256=digest(path))
        print('NORDIC_MANIFEST_READY',name,json.dumps(report['sets'][name]),flush=True)
    (out/'nordic_source_registry.json').write_text(json.dumps(report,indent=2)+'\n')
    print('NORDIC_PREPARATION_COMPLETE',flush=True)


if __name__ == '__main__':
    main()
