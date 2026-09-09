#!/usr/bin/env python3
"""Supplement sealed historical hashes with decoded-input fingerprints.

Stored PCM16 hashes and a second float-to-PCM16 conversion are different
representations. Positive controls exposed that distinction. This audit reads
the historical recordings, preserves the original index, and fingerprints the
decoded input through the same path used by the new evaluation.
"""
import argparse, concurrent.futures, gzip, hashlib, json, sqlite3
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import soundfile as sf
from audit_history import fingerprint, sha
from run import atomic_json


def decoded(job):
    expected,path=job
    try:
        pcm,sr=sf.read(path,dtype='int16',always_2d=True)
        if sr!=16000 or pcm.shape[1]!=1:return {'status':'unsupported_format'}
        if hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()!=expected:return {'status':'stored_pcm_mismatch'}
        data,_=sf.read(path,dtype='float32',always_2d=True);data=np.ascontiguousarray(data.mean(axis=1),dtype='<f4')
        requantized=(np.clip(data,-1,1)*32767).astype('<i2')
        return {'status':'verified','float32_pcm_sha256':hashlib.sha256(data.tobytes()).hexdigest(),
                'legacy_pcm_sha256':hashlib.sha256(requantized.tobytes()).hexdigest(),'seconds':len(data)/16000}
    except Exception as exc:return {'status':'decode_error','error_type':type(exc).__name__}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--history',type=Path,required=True);a=p.parse_args()
    original=json.loads((a.history/'history-audit.json').read_text());index=a.history/'history.sqlite';indexhash=sha(index)
    assert indexhash==original['index_sha256']
    db=sqlite3.connect('file:'+str(index)+'?mode=ro',uri=True);wanted={v[0] for v in db.execute("SELECT fingerprint FROM exposure WHERE kind='pcm_sha256'")};db.close()
    seen=set();available={};sources=[]
    files=sorted((f for f in original['files'] if f.get('indexed_records')),key=lambda f:(not f['path'].startswith('manifests/'),f['path']))
    for item in files:
        path=a.root/item['path'];data=path.read_bytes();assert hashlib.sha256(data).hexdigest()==item['sha256']
        if b'"pcm_sha256"' not in data:continue
        added=0
        for line in data.splitlines():
            if b'"pcm_sha256"' not in line:continue
            try:row=json.loads(line)
            except ValueError:continue
            if not isinstance(row,dict) or not row.get('pcm_sha256'):continue
            value=row['pcm_sha256']
            if fingerprint(value) not in wanted:continue
            seen.add(value)
            if value in available:continue
            path=Path(row.get('audio_filepath') or row.get('audio_path') or '/nonexistent')
            if path.is_file():available[value]=str(path);added+=1
        if added:sources.append({'path_sha256':fingerprint(item['path']),'content_sha256':item['sha256'],'additional_pcm_keys':added})
        if len(available)==len(wanted):break
    assert len(seen)==len(wanted),(len(seen),len(wanted))
    print('HISTORICAL_AUDIO',len(available),'/',len(wanted),flush=True)
    output=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for i,result in enumerate(pool.map(decoded,sorted(available.items()))):
            output.append(result)
            if i%2000==0:print('DECODED',i+1,flush=True)
    from collections import Counter
    statuses=dict(Counter(r['status'] for r in output));valid=[r for r in output if r['status']=='verified']
    payload={'original_history_index_sha256':indexhash,
             'float32_pcm_sha256':sorted({r['float32_pcm_sha256'] for r in valid}),
             'legacy_pcm_sha256':sorted({r['legacy_pcm_sha256'] for r in valid})}
    dest=a.history/'decoded-fingerprints.json.gz';temp=Path(str(dest)+'.tmp')
    with gzip.open(temp,'wt') as f:json.dump(payload,f,sort_keys=True)
    temp.replace(dest)
    assert sha(index)==indexhash
    receipt={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'complete',
             'reason':'Known historical positive controls failed the stored-PCM versus re-encoded-PCM comparison. Add fingerprints of decoded historical inputs without altering the sealed original index.',
             'original_history_index_sha256':indexhash,'original_index_unchanged':True,
             'historical_pcm_keys':len(wanted),'available_files':len(available),'missing_files':len(wanted)-len(available),
             'verification':statuses,'verified_hours':sum(r['seconds'] for r in valid)/3600,
             'fingerprint_file_sha256':sha(dest),'script_sha256':sha(__file__),'inventory_files_used':sources,
             'scoring_effect':'Complete-split recognition scores stay fixed. Strict-history-disjoint sensitivity also excludes matches to these decoded-input fingerprints.',
             'limits':'Exact stored and decoded representations only; not an acoustic near-duplicate audit or a full fingerprinting of every historical training recording.'}
    atomic_json(a.history/'decoded-fingerprints-audit.json',receipt)
    print('DECODED_HISTORY_COMPLETE',json.dumps(statuses),flush=True)


if __name__=='__main__':main()
