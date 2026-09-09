#!/usr/bin/env python3
"""Pin new benchmark inputs and seal reference membership before inference."""
import argparse, concurrent.futures, hashlib, json, sqlite3, time
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
from huggingface_hub import HfApi,HfFileSystem,hf_hub_download
import pyarrow.parquet as pq
from audit_history import fingerprint,text_key,sha

SOURCES={
 'monsoon':('VoiceArena/MonsoonASR-Open-ASR-leaderboard-en-IN','bc1da7b42ef6e2853123c97bf6d22067e4802d11'),
 'gigaspeechbench':('speechcolab/GigaSpeechBench','680d3057641b7507a1ef14974407c7b0a7964e64'),
 'eurospeech':('disco-eth/EuroSpeech','7a55bbcee3933f5d9817bb2d4c37399d596f543d'),
}
EURO={'bulgaria':'bg','croatia':'hr','estonia':'et','finland':'fi','france':'fr','germany':'de','greece':'el','italy':'it','latvia':'lv','lithuania':'lt','malta':'mt','portugal':'pt','slovakia':'sk','slovenia':'sl','uk':'en','ukraine':'uk'}


def get_parquet_metadata(job):
    source,repo,rev,file,cache=job
    if cache.is_file():return source,file,json.loads(cache.read_text())
    fs=HfFileSystem(token=False)
    for attempt in range(4):
        try:
            with fs.open(f'datasets/{repo}@{rev}/{file}','rb',block_size=64*1024) as f:
                pf=pq.ParquetFile(f,pre_buffer=False)
                columns=[c for c in pf.schema_arrow.names if c!='audio']
                # Only scoring/group metadata: do not retain personal profiles.
                if source=='monsoon':columns=['id','text','speaker_id','gender','native_state','audio_length_s']
                if source=='eurospeech':columns=['key','country','language','video_id','transcript_id','start_seconds','end_seconds','duration_seconds','human_transcript']
                return source,file,pf.read(columns=columns,use_threads=False).to_pylist()
        except Exception:
            if attempt==3:raise
            time.sleep(2**attempt)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--history',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);(a.out/'raw_metadata').mkdir(exist_ok=True);(a.out/'manifests').mkdir(exist_ok=True)
    if (a.out/'seal.json').exists():raise SystemExit('Membership already sealed')
    api=HfApi(token=False);source_receipts={};jobs=[];sets={}
    for name,(repo,rev) in SOURCES.items():
        info=api.dataset_info(repo,revision=rev,files_metadata=True)
        assert info.sha==rev
        files=[]
        for s in info.siblings:
            n=s.rfilename
            include=(name=='monsoon' and n.startswith('data/test-') and n.endswith('.parquet')) or (name=='eurospeech' and n.split('/')[0] in EURO and '/test-' in n and n.endswith('.parquet')) or (name=='gigaspeechbench' and '/data/' in n and '-EN/' in n and n.endswith(('metadata.json','audio.tar.gz')))
            if include:
                files.append({'path':n,'bytes':s.size,'sha256':s.lfs.sha256 if s.lfs else None})
                if n.endswith('.parquet'):jobs.append((name,repo,rev,n,a.out/'raw_metadata'/(name+'__'+n.replace('/','__')+'.json')))
        source_receipts[name]={'repository':repo,'revision':rev,'files':files}
        print('PINNED',name,len(files),sum(s['bytes'] for s in files),flush=True)
    (a.out/'sources.json').write_text(json.dumps(source_receipts,indent=2)+'\n')
    # Read all metadata before either model produces outputs. No sample caps,
    # confidence filtering, predicted-WER filtering, or duration filtering.
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        for source,file,data in ex.map(get_parquet_metadata,jobs):
            cache=a.out/'raw_metadata'/(source+'__'+file.replace('/','__')+'.json')
            cache.write_text(json.dumps(data,ensure_ascii=False)+'\n')
            bucket=source+'_'+(EURO[file.split('/')[0]] if source=='eurospeech' else 'en_in')
            records=sets.setdefault(bucket,[])
            for i,r in enumerate(data):
                if source=='monsoon':
                    row={'id':str(r['id']),'text':r['text'],'lang':'en','cluster':'monsoon:'+str(r['speaker_id']),'duration':r['audio_length_s'],'accent':'Indian English','gender':r['gender'],'region':r['native_state']}
                else:
                    row={'id':r['key'],'text':r['human_transcript'],'lang':EURO[file.split('/')[0]],'cluster':'eurospeech:'+file.split('/')[0]+':'+str(r['video_id']),'duration':r['duration_seconds'],'country':r['country'],'source_session_id':r['video_id'],'source_transcript_id':r['transcript_id'],'start_seconds':r['start_seconds'],'end_seconds':r['end_seconds']}
                records.append(dict(row,source=source,source_file=file,source_index=i))
            print('METADATA',bucket,file,len(data),flush=True)
    repo,rev=SOURCES['gigaspeechbench']
    for f in source_receipts['gigaspeechbench']['files']:
        if not f['path'].endswith('metadata.json'):continue
        path=hf_hub_download(repo,f['path'],revision=rev,repo_type='dataset',cache_dir=str(a.out/'hf_metadata'),token=False)
        data=json.loads(Path(path).read_text());code=f['path'].split('/')[2];bucket='gigaspeechbench_'+code.lower().replace('-','_');records=sets.setdefault(bucket,[])
        for audio in data['audios']:
            for segment in audio['segments']:
                records.append({'id':segment['sid'],'text':segment['text'],'lang':'en','cluster':'gigaspeechbench:'+audio['aid'],'source':'gigaspeechbench','source_file':f['path'],'audio_id':audio['aid'],'start_seconds':float(segment['begin_time']),'end_seconds':float(segment['end_time']),'duration':float(segment['end_time'])-float(segment['begin_time']),'category':code,'module':f['path'].split('/')[0]})
        print('METADATA',bucket,len(records),flush=True)
    while not (a.history/'history-audit.json').is_file():time.sleep(5)
    history=json.loads((a.history/'history-audit.json').read_text());db=sqlite3.connect('file:'+str(a.history/'history.sqlite')+'?mode=ro',uri=True)
    seen=set();registry={}
    for bucket,records in sorted(sets.items()):
        overlap=Counter();retained=[]
        for row in records:
            row['uid']=row['source']+':'+row['id']
            if row['uid'] in seen:raise ValueError('Duplicate official ID '+row['uid'])
            seen.add(row['uid'])
            if not isinstance(row['text'],str):raise ValueError('Missing human reference '+row['uid'])
            row['reference_normalizes_to_empty']=not bool(text_key(row['text']))
            reasons=[]
            if db.execute('SELECT 1 FROM exposure WHERE kind=? AND fingerprint=?',('text',fingerprint(text_key(row['text'])))).fetchone():reasons.append('exact_historical_normalized_text')
            if db.execute('SELECT 1 FROM exposure WHERE kind=? AND fingerprint=?',('source_row_id',fingerprint(row['id']))).fetchone():reasons.append('historical_source_row_id')
            row['history_overlap_reasons']=reasons;row['strict_text_unseen']=not reasons;overlap.update(reasons);retained.append(row)
        dest=a.out/'manifests'/(bucket+'.jsonl')
        dest.write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in sorted(retained,key=lambda r:r['uid'])))
        registry[bucket]={'path':str(dest),'sha256':sha(dest),'rows':len(retained),'hours':sum(r['duration'] for r in retained)/3600,'clusters':len({r['cluster'] for r in retained}),'lang':retained[0]['lang'],'strict_text_unseen_rows':sum(r['strict_text_unseen'] for r in retained),'overlap_counts':dict(overlap)}
    protocol={
      'created_utc':datetime.now(timezone.utc).isoformat(),'status':'sealed_before_inference','seed':20260907,'bootstrap_replicates':10000,
      'models':{'parakeet':'models/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo','orukeet':'gabor_half_20260906/orukeet_gabor_half_r15_20260906/step-0100.nemo'},
      'orukeet_expected_sha256':'4295a6d820a40b99786331d1c7a6b6c328916c8329b23d39415b0649a5d42811',
      'decoding':{'strategy':'greedy_batch','model_type':'tdt','durations':[0,1,2,3,4],'greedy':{'max_symbols':10}},
      'precision':'FP32 model parameters, CUDA BF16 autocast','batch_size':32,
      'primary_metrics':['Per-split corpus WER and CER using frozen legacy-compatible NFC scoring','EuroSpeech fixed-language macro WER','GigaSpeechBench six-accent macro WER','GigaSpeechBench twelve-domain macro WER','Monsoon full test corpus WER'],
      'secondary_metrics':['All primary endpoints restricted to exact-history-text-disjoint records','English Whisper-normalized WER','Micro WER, utterance error rate, substitutions/deletions/insertions, long-clip and demographic strata'],
      'empty_reference_policy':'Official empty or punctuation-only references remain in complete-split scoring; any predicted words count as corpus-level insertions. These records are counted separately.',
      'inference_scope':'Every segment in all pinned selected official splits, without sample caps or model-score-dependent filtering.',
      'inference_failure_policy':'Retry failing batches in smaller batches with identical settings; permanent item failure recorded and scored as an empty hypothesis. Never silently drop a reference.',
      'uncertainty':'Paired cluster bootstrap by source recording/session or speaker. Omit interval when fewer than 2 clusters; flag fewer than 20 independent clusters. No claim of unseen speakers across corpora.',
      'selection_rule':'Both checkpoints remain frozen; no tuning, replacement, or selection on these outcomes.',
      'exclusions':['EuroSpeech Denmark and Sweden excluded because prior FTSpeech/RixVox evaluation may share parliament recordings.','Unsupported-language EuroSpeech configurations excluded.','Sierra mu-bench requires an additional gated data agreement; no access request or submission made.','MUSCAT deferred: manual segments repeat conversations across devices; not part of this sealed primary suite.'],
      'coverage_statement':'New evaluation sources absent from recorded Orukeet adaptation/evaluation history. Some original EuroSpeech metadata had been inspected in an abandoned preparation draft; no model trained or evaluated on it. Exact reference matches are flagged and separately excluded in a strict sensitivity analysis. Complete NVIDIA pretraining membership and cross-corpus near-duplicates cannot be established from released records.',
      'sources_sha256':sha(a.out/'sources.json'),'history_audit_sha256':sha(a.history/'history-audit.json'),'history_index_sha256':history['index_sha256'],'sets':registry,
    }
    for label,relative in protocol['models'].items():protocol[label+'_sha256']=sha(Path('/home/nathanroll/parakeet-ft')/relative)
    assert protocol['orukeet_sha256']==protocol['orukeet_expected_sha256']
    (a.out/'seal.json').write_text(json.dumps(protocol,indent=2,ensure_ascii=False)+'\n')
    print('SEALED',len(registry),sum(x['rows'] for x in registry.values()),sum(x['hours'] for x in registry.values()),sha(a.out/'seal.json'),flush=True)

if __name__=='__main__':main()
