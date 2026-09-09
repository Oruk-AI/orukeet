#!/usr/bin/env python3
"""Exercise historical-overlap detection with recorded positives and synthetic noise.

The controls input contains private historical manifest rows. Only numeric
verification results and content hashes are written to the output receipt.
"""
import argparse, gzip, hashlib, json, sqlite3, tempfile
from pathlib import Path
import numpy as np
import soundfile as sf
from audit_history import fingerprint, sha
from compare import historical_pcm_overlap
from run import atomic_json, write_audio


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--history',type=Path,required=True)
    p.add_argument('--controls',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    index=a.history/'history.sqlite';before=sha(index)
    audit=json.loads((a.history/'decoded-fingerprints-audit.json').read_text())
    decoded_path=a.history/'decoded-fingerprints.json.gz'
    assert before==audit['original_history_index_sha256']
    assert sha(decoded_path)==audit['fingerprint_file_sha256']
    with gzip.open(decoded_path,'rt') as f:decoded=json.load(f)
    floats=set(decoded['float32_pcm_sha256']);legacy=set(decoded['legacy_pcm_sha256'])
    db=sqlite3.connect('file:'+str(index)+'?mode=ro',uri=True)
    results=[];unique=set()
    with tempfile.TemporaryDirectory() as folder:
        for i,r in enumerate(json.loads(a.controls.read_text())):
            digest=sha(r['audio_filepath'])
            if digest in unique:continue
            unique.add(digest)
            stored=bool(db.execute('SELECT 1 FROM exposure WHERE kind=? AND fingerprint=?',('pcm_sha256',fingerprint(r['pcm_sha256']))).fetchone())
            assert stored
            data,sr=sf.read(r['audio_filepath'],dtype='float32',always_2d=True)
            row=write_audio({'uid':'positive-'+str(i),'duration':len(data)/sr},data,sr,Path(folder),db)
            detected=historical_pcm_overlap(row,floats,legacy)
            assert detected
            results.append({'source':r.get('src',r.get('source_id')),'source_audio_sha256':digest,
                            'stored_pcm_in_original_index':stored,'original_detector_found_overlap':row['known_pcm_overlap'],
                            'decoded_history_detector_found_overlap':detected})
        assert len(results)>=2
        noise=np.random.default_rng(20260907).normal(0,.1,16000).astype('float32')
        row=write_audio({'uid':'synthetic-negative','duration':1.0},noise,16000,Path(folder),db)
        assert not historical_pcm_overlap(row,floats,legacy)
    db.close();assert sha(index)==before
    atomic_json(a.output,{'status':'passed','method':'Known historical recordings passed through the current audio-preparation and final-scoring overlap paths; no ASR inference.',
                         'positive_controls':results,'unique_positive_controls':len(results),
                         'synthetic_negative_detected_as_overlap':False,'negative_seed':20260907,
                         'original_index_unchanged':True,'original_index_sha256':before,
                         'decoded_fingerprints_sha256':sha(decoded_path),'script_sha256':sha(__file__)})
    print('HISTORY_CONTROLS_PASSED',len(results),'positive, 1 negative')


if __name__=='__main__':main()
