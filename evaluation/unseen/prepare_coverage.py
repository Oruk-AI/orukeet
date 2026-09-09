#!/usr/bin/env python3
"""Seal a mechanical coverage extension for the nine missing model languages."""
import argparse,concurrent.futures,json,sqlite3,time
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
from huggingface_hub import HfApi,HfFileSystem
import pyarrow.parquet as pq
from audit_history import fingerprint,text_key,sha

SOURCES={
 'voxpopuli':{'repository':'facebook/voxpopuli','revision':'42f01879c780b4a2e90ec0b4f616c2ece526e4f1','languages':['cs','es','hu','nl','pl','ro'],'split':'validation'},
 'nst_da':{'repository':'alexandrainst/nst-da','revision':'0f14ad2005e0aab8f56cf3213b7689da1faf23c2','languages':['da'],'split':'test'},
 'nst_sv':{'repository':'jzju/nst','revision':'ade45482b1fa163b34177963c1e6f4d29621e24f','languages':['sv'],'split':'test'},
 'golos_crowd':{'repository':'bond005/sberdevices_golos_10h_crowd','revision':'e634b6b810e4d30c81b4c6d8262379fe8b9f708c','languages':['ru'],'split':'test'},
 'golos_farfield':{'repository':'bond005/sberdevices_golos_100h_farfield','revision':'c93949f7140beef4adc404e7b54841e957f81c54','languages':['ru'],'split':'test'},
}


def metadata(job):
    source,file,cache=job
    if cache.is_file():return file,json.loads(cache.read_text())
    fs=HfFileSystem(token=False)
    for attempt in range(4):
        try:
            with fs.open(f"datasets/{source['repository']}@{source['revision']}/{file}",'rb',block_size=65536) as f:
                pf=pq.ParquetFile(f,pre_buffer=False);cols=[c for c in pf.schema_arrow.names if c!='audio']
                # Skip personal attributes that do not define scoring or groups.
                cols=[c for c in cols if c not in ['age','recording_datetime','raw_text']]
                data=pf.read(columns=cols,use_threads=False).to_pylist()
            cache.write_text(json.dumps(data,ensure_ascii=False)+'\n');return file,data
        except Exception:
            if attempt==3:raise
            time.sleep(2**attempt)


def main():
    p=argparse.ArgumentParser();p.add_argument('--primary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--alignment-followup',action='store_true');p.add_argument('--coverage',type=Path);a=p.parse_args()
    out=a.output/'metadata';out.mkdir(parents=True,exist_ok=True);(out/'raw_metadata').mkdir(exist_ok=True);(out/'manifests').mkdir(exist_ok=True)
    if (out/'seal.json').exists():raise SystemExit('Coverage already sealed')
    primary=json.loads((a.primary/'metadata/seal.json').read_text());wanted={'bg','hr','cs','da','nl','en','et','fi','fr','de','el','hu','it','lv','lt','mt','pl','pt','ro','ru','sk','sl','es','sv','uk'}-{r['lang'] for r in primary['sets'].values()}
    assert wanted=={'cs','da','es','hu','nl','pl','ro','ru','sv'}
    chosen_sources=SOURCES
    if a.alignment_followup:
        assert a.coverage is not None
        wanted={'el','it'}
        chosen_sources={
            'voxpopuli':{**SOURCES['voxpopuli'],'languages':['it']},
            'lesbos':{'repository':'ilsp/lesbian-speech-corpus','revision':'6bb115fa67491dd74ba81f59e8a0fbb94384ea72','languages':['el'],'split':'test'},
        }
    db=sqlite3.connect('file:'+str(a.primary/'history/history.sqlite')+'?mode=ro',uri=True)
    db.execute('PRAGMA cache_size=-262144')
    primary_text=set()
    registries=[primary]
    if a.alignment_followup:registries.append(json.loads((a.coverage/'metadata/seal.json').read_text()))
    for info in [v for registry in registries for v in registry['sets'].values()]:
        for line in Path(info['path']).open():primary_text.add(fingerprint(text_key(json.loads(line)['text'])))
    api=HfApi(token=False);sources={};jobs=[]
    for name,source in chosen_sources.items():
        info=api.dataset_info(source['repository'],revision=source['revision'],files_metadata=True);files=[]
        for f in info.siblings:
            n=f.rfilename
            keep=n.endswith('.parquet') and ((name=='voxpopuli' and n.split('/')[0] in source['languages'] and '/validation-' in n) or (name!='voxpopuli' and n.startswith('data/test-')))
            if keep:
                files.append({'path':n,'bytes':f.size,'sha256':f.lfs.sha256 if f.lfs else None})
                jobs.append((name,source,n,out/'raw_metadata'/(name+'__'+n.replace('/','__')+'.json')))
        sources[name]={**source,'format':'parquet','files':files}
        print('PINNED',name,len(files),sum(f['bytes'] for f in files),flush=True)
    sets={};excluded=Counter();input_counts=Counter();seen=set()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        futures=[ex.submit(metadata,(source,file,cache)) for _,source,file,cache in jobs]
        for (name,source,filename,_),future in zip(jobs,futures):
            _,data=future.result();lang=filename.split('/')[0] if name=='voxpopuli' else source['languages'][0];bucket=name+'_'+lang;rows=sets.setdefault(bucket,[])
            for i,r in enumerate(data):
                input_counts[bucket]+=1
                if name=='voxpopuli' and r.get('is_gold_transcript') is not True:excluded[bucket+':nonhuman_transcript']+=1;continue
                original_id=r.get('audio_id');record_id=original_id or filename+':'+str(i)
                if original_id and db.execute('SELECT 1 FROM exposure WHERE kind=? AND fingerprint=?',('source_row_id',fingerprint(original_id))).fetchone():excluded[bucket+':historically_exposed_id']+=1;continue
                uid=name+':'+lang+':'+record_id
                assert uid not in seen,uid;seen.add(uid)
                text=r.get('normalized_text',r.get('text',r.get('transcription')))
                if not isinstance(text,str):excluded[bucket+':missing_transcript']+=1;continue
                text_fp=fingerprint(text_key(text));reasons=[]
                if db.execute('SELECT 1 FROM exposure WHERE kind=? AND fingerprint=?',('text',text_fp)).fetchone():reasons.append('exact_historical_normalized_text')
                if text_fp in primary_text:reasons.append('exact_primary_suite_normalized_text')
                speaker=r.get('speaker_id');available=speaker is not None and str(speaker).lower() not in ['','none','unknown','null']
                # Do not create an utterance-level CI when speaker/session IDs
                # have been stripped by a source mirror.
                cluster=name+':'+str(speaker) if available else 'unavailable:'+name
                known_speaker=False
                if available:
                    known_speaker=any(db.execute('SELECT 1 FROM exposure WHERE kind=? AND fingerprint=?',('speaker',fingerprint(v))).fetchone() for v in [name+':'+str(speaker),name+':'+name+':'+str(speaker)])
                rows.append({'uid':uid,'id':record_id,'source_row_id':original_id,'source':name,'lang':lang,'text':text,'duration':None,'cluster':cluster,'cluster_metadata_available':available,'historically_evaluated_speaker':known_speaker,'source_file':filename,'source_index':i,'source_split':source['split'],'source_original_id_available':original_id is not None,'history_overlap_reasons':reasons,'strict_text_unseen':not reasons,'reference_normalizes_to_empty':not bool(text_key(text)),'accent':r.get('accent') or r.get('dialect') or 'unknown','gender':r.get('gender') or r.get('sex') or 'unknown'})
            print('METADATA',bucket,filename,len(data),flush=True)
    registry={}
    for bucket,rows in sorted(sets.items()):
        dest=out/'manifests'/(bucket+'.jsonl');dest.write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in sorted(rows,key=lambda r:r['uid'])))
        has_groups=all(r['cluster_metadata_available'] for r in rows)
        registry[bucket]={'path':str(dest),'sha256':sha(dest),'rows':len(rows),'hours':None,'lang':rows[0]['lang'],'source':rows[0]['source'],'clusters':len({r['cluster'] for r in rows}) if has_groups else None,'cluster_metadata_available':has_groups,'strict_text_unseen_rows':sum(r['strict_text_unseen'] for r in rows),'historically_evaluated_speaker_rows':sum(r['historically_evaluated_speaker'] for r in rows),'overlap_counts':dict(Counter(reason for r in rows for reason in r['history_overlap_reasons']))}
    assert {r['lang'] for r in registry.values()}==wanted
    if not a.alignment_followup:assert input_counts['golos_crowd_ru']==9994 and input_counts['golos_farfield_ru']==1916
    (out/'sources.json').write_text(json.dumps(sources,indent=2)+'\n')
    seal={k:v for k,v in primary.items() if k not in ['sets','sources_sha256','created_utc','exclusions','primary_metrics','secondary_metrics','coverage_statement']}
    seal.update(created_utc=datetime.now(timezone.utc).isoformat(),suite='language_coverage_extension',status='sealed_before_inference',seed=20260908,sets=registry,sources_sha256=sha(out/'sources.json'),primary_protocol_sha256=sha(a.primary/'metadata/seal.json'),
      selection_rule='Mechanically cover the nine supported languages absent from the first suite, using every human-transcribed record of the pinned published test/validation partitions except exact previously exposed source IDs. No model tuning or checkpoint selection.',
      primary_metrics=['Per-split and per-language corpus WER/CER. This supplementary pass does not replace the original primary endpoints.'],secondary_metrics=['Exact-reference-history-disjoint sensitivity analysis. Source speaker/recording-cluster uncertainty where grouping IDs are available.'],excluded_rows=dict(excluded),input_rows_by_split=dict(input_counts),
      coverage_statement='The supplemental partition choices follow missing-language coverage after the first suite was sealed. Some first-suite results were available during this preparation; no outputs from these supplemental partitions were examined before this seal. Checkpoints are unchanged.',
      limitations=['VoxPopuli validation clips can have speakers previously heard in other VoxPopuli partitions; speaker familiarity is reported separately from record-level exclusion.',
        'The Swedish NST mirror and Golos mirrors omit speaker/session identifiers. Their confidence intervals are not estimated.',
        'The NST Swedish partition is the published mirror test partition; it is not asserted to be the original NST speaker-held-out test.',
        'NVIDIA pretraining membership and cross-corpus acoustic near-duplicates remain unverified. Sources are not claimed to postdate pretraining.',
        'The Golos parquet mirrors contain null references; unannotated rows are excluded before inference and counted explicitly. No reference is invented.',
        'Durations are obtained from decoded audio, since these parquet metadata omit reliable durations.'])
    if a.alignment_followup:
        seal.update(suite='reference_quality_followup',seed=20260909,
          selection_rule='Two independent-language checks added after source alignment defects were observed in Greek and Italian EuroSpeech. Use the full Lesbos Greek-dialect test and Italian VoxPopuli validation with human references; no output from these partitions has been seen before sealing.',
          coverage_statement='Supplementary post-diagnostic comparison, not a replacement for the sealed primary scores. The Greek corpus measures the Lesbos dialect, not standard Greek in general. Speaker identifiers are provided by the datasets; pretraining overlap and cross-corpus voice identity remain unaudited.',
          preceding_coverage_protocol_sha256=sha(a.coverage/'metadata/seal.json'))
    (out/'seal.json').write_text(json.dumps(seal,indent=2,ensure_ascii=False)+'\n')
    link=a.output/'history'
    if not link.exists():link.symlink_to(a.primary/'history',target_is_directory=True)
    print('COVERAGE_SEALED',len(registry),sum(r['rows'] for r in registry.values()),sha(out/'seal.json'),flush=True)

if __name__=='__main__':main()
