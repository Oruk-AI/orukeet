#!/usr/bin/env python3
"""Run two frozen NeMo models on sealed, complete new benchmark splits."""
import argparse, gc, hashlib, importlib.metadata, io, json, os, shutil, sqlite3, sys, tarfile, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import soxr
from huggingface_hub import hf_hub_download
from audit_history import fingerprint,sha


def read_rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]


def atomic_json(p,value):
    temp=Path(str(p)+'.tmp');temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');temp.replace(p)


def fetch(source,file,folder):
    p=Path(hf_hub_download(source['repository'],file['path'],repo_type='dataset',revision=source['revision'],local_dir=folder,token=False))
    assert p.stat().st_size==file['bytes'],str(p)
    if file['sha256']:assert sha(p)==file['sha256'],str(p)
    return p


def write_audio(row,data,sr,out,db):
    data=np.asarray(data,dtype=np.float32)
    if data.ndim==2:data=data.mean(axis=1)
    assert data.ndim==1 and len(data)>0 and np.isfinite(data).all(),row['uid']
    if sr!=16000:data=soxr.resample(data,sr,16000,quality='HQ')
    data=np.ascontiguousarray(data,dtype='<f4')
    expected=row['duration'];duration=len(data)/16000
    duration_mismatch=abs(expected-duration)>max(.1,.01*expected)
    # Published EuroSpeech audio can be trimmed without updating duration metadata.
    # Preserve the official waveform/reference and record the discrepancy; the
    # sealed protocol prohibits duration-based selection of test records.
    path=out/(fingerprint(row['uid'])+'.wav')
    # Float WAV preserves the resampled input, avoiding extra PCM quantization.
    sf.write(path,data,16000,subtype='FLOAT')
    legacy_pcm_hash=hashlib.sha256((np.clip(data,-1,1)*32767).astype('<i2').tobytes()).hexdigest()
    overlap=bool(db.execute('SELECT 1 FROM exposure WHERE kind=? AND fingerprint=?',('pcm_sha256',fingerprint(legacy_pcm_hash))).fetchone())
    return dict(row,audio_filepath=str(path),actual_duration=duration,declared_duration_mismatch=duration_mismatch,duration_difference_seconds=duration-expected,float32_pcm_sha256=hashlib.sha256(data.tobytes()).hexdigest(),legacy_pcm_sha256=legacy_pcm_hash,known_pcm_overlap=overlap,audio_file_sha256=sha(path))


def materialize(bucket,rows,sources,scratch,out,history):
    receipt_path=out/'audio-receipts'/(bucket+'.jsonl');receipt_path.parent.mkdir(exist_ok=True)
    audio_dir=scratch/'audio'/bucket;audio_dir.mkdir(parents=True,exist_ok=True)
    if receipt_path.exists():
        previous=read_rows(receipt_path)
        if all(Path(r['audio_filepath']).is_file() and sha(r['audio_filepath'])==r['audio_file_sha256'] for r in previous):return previous
    download_dir=scratch/'downloads'/bucket;download_dir.mkdir(parents=True,exist_ok=True)
    source=sources[rows[0]['source']];db=sqlite3.connect('file:'+str(history)+'?mode=ro',uri=True)
    result={};source_name=rows[0]['source'];wanted={r['uid'] for r in rows}
    if source_name in ['monsoon','eurospeech']:
        byfile=defaultdict(dict)
        for r in rows:byfile[r['source_file']][r['source_index']]=r
        for filename,indices in sorted(byfile.items()):
            file=next(f for f in source['files'] if f['path']==filename);path=fetch(source,file,download_dir)
            index=0
            for batch in pq.ParquetFile(path).iter_batches(batch_size=16,columns=['audio']):
                for raw in batch.to_pylist():
                    if index in indices:
                        row=indices[index];data,sr=sf.read(io.BytesIO(raw['audio']['bytes']),dtype='float32',always_2d=True)
                        result[row['uid']]=write_audio(row,data,sr,audio_dir,db)
                    index+=1
            print('AUDIO_SHARD',bucket,filename,len(result),flush=True)
            path.unlink() # Only this run's reproducible download; metadata/pins retained.
    else:
        archive=next(f for f in source['files'] if f['path']==rows[0]['source_file'].replace('metadata.json','audio.tar.gz'))
        path=fetch(source,archive,download_dir);byaudio=defaultdict(list)
        for r in rows:byaudio[r['audio_id']].append(r)
        with tarfile.open(path,'r:gz') as t:
            for member in t:
                aid=Path(member.name).stem
                if not member.isfile() or aid not in byaudio:continue
                data,sr=sf.read(io.BytesIO(t.extractfile(member).read()),dtype='float32',always_2d=True)
                for row in byaudio.pop(aid):
                    clip=data[int(row['start_seconds']*sr):int(row['end_seconds']*sr)]
                    result[row['uid']]=write_audio(row,clip,sr,audio_dir,db)
        if byaudio:raise ValueError('Missing source recordings '+str(list(byaudio)[:5]))
        path.unlink()
    db.close()
    assert set(result)==wanted,(bucket,len(result),len(wanted))
    result=[result[r['uid']] for r in rows]
    if any(r['known_pcm_overlap'] for r in result):print('KNOWN_PCM_OVERLAP',bucket,sum(r['known_pcm_overlap'] for r in result),flush=True)
    temp=Path(str(receipt_path)+'.tmp');temp.write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in result));temp.replace(receipt_path)
    shutil.rmtree(download_dir)
    print('AUDIO_READY',bucket,len(result),flush=True)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--experiment',type=Path,required=True);p.add_argument('--scratch',type=Path,required=True);p.add_argument('--sets',nargs='*');a=p.parse_args()
    seal_path=a.experiment/'metadata/seal.json';seal=json.loads(seal_path.read_text());seal_hash=sha(seal_path)
    assert seal['status']=='sealed_before_inference'
    assert seal['parakeet_sha256']=='3cbdc85877e668ca7b82d0d56770eb1fac76691f55d6b97545e8d61ca588d10d'
    assert seal['orukeet_sha256']=='4295a6d820a40b99786331d1c7a6b6c328916c8329b23d39415b0649a5d42811'
    sources=json.loads((a.experiment/'metadata/sources.json').read_text());assert sha(a.experiment/'metadata/sources.json')==seal['sources_sha256']
    for info in seal['sets'].values():assert sha(info['path'])==info['sha256']
    out=a.experiment/'results';out.mkdir(exist_ok=True)
    import torch
    from nemo.collections.asr.models import ASRModel
    from omegaconf import OmegaConf
    from nemo.utils import logging as nemo_logging
    nemo_logging.set_verbosity(nemo_logging.ERROR)
    torch.manual_seed(seal['seed']);np.random.seed(seal['seed'])
    models={};decodes={};model_hashes={}
    for label,relative in seal['models'].items():
        path=a.root/relative;model_hashes[label]=sha(path);assert model_hashes[label]==seal[label+'_sha256']
        model=ASRModel.restore_from(str(path),map_location='cuda');model.eval();model.freeze()
        decode=OmegaConf.create(OmegaConf.to_container(model.cfg.decoding,resolve=True))
        assert decode.strategy==seal['decoding']['strategy'] and decode.greedy.max_symbols==10
        model.change_decoding_strategy(decode);models[label]=model;decodes[label]=OmegaConf.to_container(decode,resolve=True)
    assert decodes['orukeet']==decodes['parakeet']
    versions={name:importlib.metadata.version(name) for name in ['torch','nemo_toolkit','pyarrow','huggingface_hub','soundfile','soxr','rapidfuzz','whisper_normalizer']}
    runtime={'started_utc':datetime.now(timezone.utc).isoformat(),'seal_sha256':seal_hash,'model_hashes':model_hashes,'decoding':decodes['parakeet'],'versions':versions,'gpu':torch.cuda.get_device_name(),'precision':seal['precision'],'script_sha256':sha(__file__),'sets':{}}
    prior_path=out/'runtime.json'
    if prior_path.exists():
        previous=json.loads(prior_path.read_text());assert previous['seal_sha256']==seal_hash and previous['model_hashes']==model_hashes;runtime['sets']=previous['sets'];runtime['resumed_from']=previous['started_utc']
    runtime['cuda_version']=torch.version.cuda
    runtime['cudnn_version']=torch.backends.cudnn.version()
    runtime['float32_matmul_precision']=torch.get_float32_matmul_precision()
    runtime['tf32_matmul_allowed']=torch.backends.cuda.matmul.allow_tf32
    runtime['tf32_cudnn_allowed']=torch.backends.cudnn.allow_tf32
    runtime['compute_capability']=torch.cuda.get_device_capability()
    atomic_json(prior_path,runtime)
    buckets=a.sets or (['monsoon_en_in']+sorted(s for s in seal['sets'] if s!='monsoon_en_in'))
    preparation=ThreadPoolExecutor(max_workers=1)
    futures={}
    def prefetch(bucket):
        manifest=read_rows(seal['sets'][bucket]['path'])
        return materialize(bucket,manifest,sources,a.scratch,a.experiment,a.experiment/'history/history.sqlite')
    for bucket in buckets:
        if not (out/(bucket+'.complete.json')).exists():
            futures[bucket]=preparation.submit(prefetch,bucket);break
    for bi,bucket in enumerate(buckets):
        info=seal['sets'][bucket];rows=read_rows(info['path']);completion=out/(bucket+'.complete.json')
        if completion.exists():
            c=json.loads(completion.read_text());assert c['seal_sha256']==seal_hash
            for label in models:assert sha(out/label/(bucket+'.jsonl'))==c['prediction_sha256'][label]
            print('ALREADY_COMPLETE',bucket,flush=True);continue
        rows=futures[bucket].result() if bucket in futures else prefetch(bucket)
        for next_bucket in buckets[bi+1:]:
            if not (out/(next_bucket+'.complete.json')).exists():
                if next_bucket not in futures:futures[next_bucket]=preparation.submit(prefetch,next_bucket)
                break
        # Fixed duration sorting reduces padding; both models receive identical
        # files and batches. Alternate model order by split to reduce order bias.
        rows=sorted(rows,key=lambda r:(r['actual_duration'],r['uid']))
        labels=list(models) if bi%2==0 else list(reversed(models));timings={};failures={}
        for label in labels:
            model=models[label];folder=out/label;folder.mkdir(exist_ok=True);path=folder/(bucket+'.jsonl')
            prior=read_rows(path) if path.exists() else [];done={r['uid'] for r in prior}
            assert len(done)==len(prior) and done<={r['uid'] for r in rows}
            pending=[r for r in rows if r['uid'] not in done];start=time.monotonic();failure_count=0
            def infer(batch):
                nonlocal failure_count
                try:
                    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                        hyps=model.transcribe([r['audio_filepath'] for r in batch],batch_size=min(seal['batch_size'],len(batch)),num_workers=4,verbose=False)
                    assert len(hyps)==len(batch)
                    return [(h.text if hasattr(h,'text') else str(h),None) for h in hyps]
                except Exception as exc:
                    torch.cuda.empty_cache();gc.collect()
                    if len(batch)>1:
                        mid=len(batch)//2;return infer(batch[:mid])+infer(batch[mid:])
                    failure_count+=1;return [('',type(exc).__name__+': '+str(exc)[:400])]
            with path.open('a') as f:
                for start_index in range(0,len(pending),256):
                    batch=pending[start_index:start_index+256];hyps=infer(batch)
                    for row,(hyp,error) in zip(batch,hyps):
                        prediction={'uid':row['uid'],'reference_sha256':fingerprint(row['text']),'audio_sha256':row['audio_file_sha256'],'pred_text':hyp,'failure':error,'model_sha256':model_hashes[label],'seal_sha256':seal_hash}
                        f.write(json.dumps(prediction,ensure_ascii=False)+'\n')
                    f.flush()
                    if start_index%256==0:print('PROGRESS',bucket,label,len(done)+start_index+len(batch),'/',len(rows),flush=True)
            torch.cuda.synchronize();timings[label]={'seconds':time.monotonic()-start,'rows_this_process':len(pending),'resumed_rows':len(done)};failures[label]=failure_count
            check=read_rows(path);assert len(check)==len(rows) and {r['uid'] for r in check}=={r['uid'] for r in rows}
            print('MODEL_COMPLETE',bucket,label,failures[label],timings[label],flush=True)
        receipt={'seal_sha256':seal_hash,'rows':len(rows),'audio_receipt_sha256':sha(a.experiment/'audio-receipts'/(bucket+'.jsonl')),'prediction_sha256':{label:sha(out/label/(bucket+'.jsonl')) for label in models},'timing':timings,'failures_this_process':failures,'completed_utc':datetime.now(timezone.utc).isoformat()}
        atomic_json(completion,receipt);runtime['sets'][bucket]=receipt;atomic_json(prior_path,runtime)
        shutil.rmtree(a.scratch/'audio'/bucket)
        print('SPLIT_COMPLETE',bucket,len(rows),flush=True)
    preparation.shutdown(wait=True)
    runtime['final_model_hashes']={label:sha(a.root/relative) for label,relative in seal['models'].items()}
    assert runtime['final_model_hashes']==model_hashes
    runtime['completed_utc']=datetime.now(timezone.utc).isoformat();runtime['complete']=set(runtime['sets'])==set(seal['sets']);atomic_json(prior_path,runtime)
    print('BENCHMARK_COMPLETE',runtime['complete'],len(runtime['sets']),flush=True)

if __name__=='__main__':main()
