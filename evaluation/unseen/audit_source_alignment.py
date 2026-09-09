#!/usr/bin/env python3
"""Diagnose source/reference alignment without changing benchmark membership.

This post-inference diagnostic is not an alternative scoring protocol. It
checks two source shards with unexpectedly large errors against their pinned
Parquet bytes, including deterministic audio fingerprint spot checks.
"""
import argparse, hashlib, io, json, shutil
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import soxr
from run import atomic_json, fetch, read_rows, sha
from metrics import counts, normalize
from rapidfuzz.distance import Levenshtein


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--scratch',type=Path,required=True)
    a=p.parse_args()
    source=json.loads((a.experiment/'metadata/sources.json').read_text())['eurospeech']
    result={'purpose':'Post-inference source-integrity diagnostic; no membership, reference, model, or metric changes.',
            'source_revision':source['revision'],'script_sha256':sha(__file__),'shards':{}}
    for bucket,filename in [('eurospeech_el','greece/test-00005-of-00007.parquet'),('eurospeech_it','italy/test-00004-of-00009.parquet')]:
        allrows=read_rows(a.experiment/'metadata/manifests'/(bucket+'.jsonl'))
        rows={r['source_index']:r for r in allrows if r['source_file']==filename}
        references={r['uid']:normalize(r['text']).split() for r in allrows}
        byuid={r['uid']:r for r in allrows}
        predictions={m:{r['uid']:r for r in read_rows(a.experiment/'results'/m/(bucket+'.jsonl'))} for m in ['parakeet','orukeet']}
        receipt={r['uid']:r for r in read_rows(a.experiment/'audio-receipts'/(bucket+'.jsonl'))}
        file=next(f for f in source['files'] if f['path']==filename)
        path=fetch(source,file,a.scratch/bucket)
        selected=set(np.linspace(0,len(rows)-1,8,dtype=int).tolist())
        checked=[];metadata_checks=0;index=0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=16,columns=['audio','key','human_transcript','asr_transcript','wer','cer']):
            for raw in batch.to_pylist():
                row=rows[index]
                assert raw['key']==row['id'] and raw['human_transcript']==row['text']
                metadata_checks+=1
                if index in selected:
                    audio,sr=sf.read(io.BytesIO(raw['audio']['bytes']),dtype='float32',always_2d=True)
                    audio=audio.mean(axis=1)
                    if sr!=16000:audio=soxr.resample(audio,sr,16000,quality='HQ')
                    audio=np.ascontiguousarray(audio,dtype='<f4')
                    digest=hashlib.sha256(audio.tobytes()).hexdigest()
                    assert digest==receipt[row['uid']]['float32_pcm_sha256']
                    item={'source_index':index,'record_sha256':hashlib.sha256(row['uid'].encode()).hexdigest(),
                          'pcm_matches_inference':True,'provider_wer_field':raw['wer'],'provider_cer_field':raw['cer'],
                          'provider_asr_vs_human':counts(raw['human_transcript'],raw['asr_transcript'])}
                    for m in predictions:
                        hyp=predictions[m][row['uid']]['pred_text']
                        item[m+'_vs_human']=counts(raw['human_transcript'],hyp)
                        item[m+'_vs_provider_asr']=counts(raw['asr_transcript'],hyp)
                    # Locate text similarity elsewhere in the same recording.
                    # This diagnoses offsets; it never replaces the scored ref.
                    words=normalize(predictions['parakeet'][row['uid']]['pred_text']).split()
                    candidates=[(Levenshtein.distance(words,ref)/max(len(words),len(ref),1),uid)
                                for uid,ref in references.items() if byuid[uid]['cluster']==row['cluster']]
                    distance,nearest=min(candidates)
                    item['nearest_same_session_reference']={
                        'record_sha256':hashlib.sha256(nearest.encode()).hexdigest(),
                        'start_offset_seconds':byuid[nearest]['start_seconds']-row['start_seconds'],
                        'normalized_word_distance':distance,
                        'assigned_reference_normalized_word_distance':Levenshtein.distance(words,references[row['uid']])/max(len(words),len(references[row['uid']]),1)}
                    checked.append(item)
                    # Private log only, never included in the numeric receipt.
                    print(bucket,index,'HUMAN',raw['human_transcript'],'PROVIDER_ASR',raw['asr_transcript'],
                          'BASE',predictions['parakeet'][row['uid']]['pred_text'],flush=True)
                index+=1
        assert metadata_checks==len(rows)
        result['shards'][filename]={'source_sha256':file['sha256'],'metadata_rows_verified':metadata_checks,'audio_samples_verified':checked}
        shutil.rmtree(a.scratch/bucket)
    atomic_json(a.experiment/'source-alignment-audit.json',result)


if __name__=='__main__':main()
