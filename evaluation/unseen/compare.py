#!/usr/bin/env python3
"""Strict one-to-one comparison with paired recording/speaker bootstraps."""
import argparse,gzip,json
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from audit_history import fingerprint,sha
from metrics import counts,ENGLISH
from run import read_rows,atomic_json


def historical_pcm_overlap(audio,decoded_float,decoded_legacy):
    return audio['known_pcm_overlap'] or audio['float32_pcm_sha256'] in decoded_float or audio['legacy_pcm_sha256'] in decoded_legacy


def aggregate(items):
    n=len(items);labels=['parakeet','orukeet'];result={'rows':n}
    if not n:return result
    result['hours']=sum(r['duration'] for r in items)/3600;result['cluster_metadata_available']=all(r.get('cluster_metadata_available',True) for r in items);result['clusters']=len({r['cluster'] for r in items}) if result['cluster_metadata_available'] else None
    for label in labels:
        totals=Counter()
        for row in items:totals.update(row[label])
        result[label]=dict(totals)
        for numerator,denominator,name in [('errors','words','wer'),('char_errors','chars','cer'),('utterance_error','rows','utterance_error_rate')]:
            den=n if denominator=='rows' else totals[denominator]
            result[label][name]=100*totals[numerator]/den if den else None
    b=result['parakeet']['wer'];c=result['orukeet']['wer'];result['wer_delta_pp']=c-b if b is not None and c is not None else None;result['relative_wer_reduction_pct']=100*(b-c)/b if b else None
    return result


def paired_bootstrap(items,replicates=10000,seed=20260907):
    if any(r.get('cluster_metadata_available',True) is False for r in items):
        return None,None,{'clusters':None,'reason':'Complete speaker/session grouping is unavailable for this partition; no independence-based interval is estimated.'}
    grouped=defaultdict(lambda:np.zeros(3,dtype=np.float64))
    for r in items:grouped[r['cluster']]+=np.array([r['parakeet']['errors'],r['orukeet']['errors'],r['parakeet']['words']])
    if len(grouped)<2:return None,None,{'clusters':len(grouped),'reason':'Fewer than two independent recorded groups'}
    data=np.array([grouped[g] for g in sorted(grouped)]);rng=np.random.default_rng(seed);draws=[]
    for start in range(0,replicates,128):
        ix=rng.integers(0,len(data),size=(min(128,replicates-start),len(data)));total=data[ix].sum(axis=1);valid=total[:,2]>0
        draws.append(np.column_stack([100*total[valid,0]/total[valid,2],100*total[valid,1]/total[valid,2]]))
    draws=np.concatenate(draws);delta=draws[:,1]-draws[:,0];relative=100*(draws[:,0]-draws[:,1])/np.where(draws[:,0]>0,draws[:,0],np.nan)
    return draws,delta,{'clusters':len(data),'replicates':len(draws),'wer_delta_pp_95ci':np.quantile(delta,[.025,.975]).tolist(),'relative_wer_reduction_pct_95ci':np.nanquantile(relative,[.025,.975]).tolist(),'small_cluster_count':len(data)<20,'bootstrap_method':'paired ordinary cluster resampling with replacement'}


def macro_endpoint(names,points,draws):
    kept=[n for n in names if n in points and 'parakeet' in points[n]]
    if not kept:return {'sets':[]}
    b=float(np.mean([points[n]['parakeet']['wer'] for n in kept]));c=float(np.mean([points[n]['orukeet']['wer'] for n in kept]));result={'sets':kept,'parakeet_wer':b,'orukeet_wer':c,'wer_delta_pp':c-b,'relative_wer_reduction_pct':100*(b-c)/b if b else None}
    if all(n in draws and draws[n] is not None for n in kept):
        length=min(len(draws[n]) for n in kept);matrix=np.mean(np.array([draws[n][:length] for n in kept]),axis=0);d=matrix[:,1]-matrix[:,0]
        result['wer_delta_pp_95ci']=np.quantile(d,[.025,.975]).tolist();result['relative_wer_reduction_pct_95ci']=np.quantile(100*(matrix[:,0]-matrix[:,1])/matrix[:,0],[.025,.975]).tolist();result['bootstrap_method']='Paired cluster bootstrap independently stratified by fixed split; equal split weights.'
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--allow-partial',action='store_true');a=p.parse_args()
    sealpath=a.experiment/'metadata/seal.json';seal=json.loads(sealpath.read_text());sealhash=sha(sealpath);allrows=[];sets={};english_sets={};strict_sets={};strict_english_sets={};draws={};en_draws={};strict_draws={};strict_en_draws={};integrity={};numeric=[];missing=[]
    decoded_audit_path=a.experiment/'history/decoded-fingerprints-audit.json'
    decoded_path=a.experiment/'history/decoded-fingerprints.json.gz'
    decoded_audit=json.loads(decoded_audit_path.read_text())
    assert decoded_audit['status']=='complete' and decoded_audit['original_history_index_sha256']==seal['history_index_sha256']
    assert sha(decoded_path)==decoded_audit['fingerprint_file_sha256']
    with gzip.open(decoded_path,'rt') as f:decoded=json.load(f)
    decoded_float=set(decoded['float32_pcm_sha256']);decoded_legacy=set(decoded['legacy_pcm_sha256'])
    for si,(bucket,info) in enumerate(sorted(seal['sets'].items())):
        complete=a.experiment/'results'/(bucket+'.complete.json')
        if not complete.exists():missing.append(bucket);continue
        receipt=json.loads(complete.read_text());assert receipt['seal_sha256']==sealhash
        manifestpath=Path(info['path']);assert sha(manifestpath)==info['sha256'];raw=read_rows(manifestpath);index={r['uid']:r for r in raw};assert len(index)==len(raw)==info['rows']
        audio_path=a.experiment/'audio-receipts'/(bucket+'.jsonl');assert sha(audio_path)==receipt['audio_receipt_sha256'];audio={r['uid']:r for r in read_rows(audio_path)};assert set(audio)==set(index)
        effective_overlap={uid:historical_pcm_overlap(v,decoded_float,decoded_legacy) for uid,v in audio.items()}
        pred={}
        for label in ['parakeet','orukeet']:
            path=a.experiment/'results'/label/(bucket+'.jsonl');assert sha(path)==receipt['prediction_sha256'][label]
            records=read_rows(path);pred[label]={r['uid']:r for r in records};assert len(pred[label])==len(records)==len(index) and set(pred[label])==set(index)
        items=[];enitems=[];failures=Counter();audiohashes=Counter()
        cluster_indices={name:i for i,name in enumerate(sorted({r['cluster'] for r in raw}))}
        for uid,r in sorted(index.items()):
            item={'uid':uid,'bucket':bucket,'lang':r['lang'],'source':r['source'],'cluster':r['cluster'],'duration':audio[uid]['actual_duration'],'strict_text_unseen':r['strict_text_unseen'] and not effective_overlap[uid],'cluster_metadata_available':r.get('cluster_metadata_available',True),'known_historical_pcm_overlap':effective_overlap[uid]}
            for name in ['region','gender','accent','category']:item[name]=r.get(name) or 'unknown'
            enitem=dict(item)
            for label in ['parakeet','orukeet']:
                h=pred[label][uid]
                assert h['reference_sha256']==fingerprint(r['text']) and h['audio_sha256']==audio[uid]['audio_file_sha256'] and h['model_sha256']==seal[label+'_sha256'] and h['seal_sha256']==sealhash
                item[label]=counts(r['text'],h['pred_text'])
                if r['lang']=='en':enitem[label]=counts(r['text'],h['pred_text'],ENGLISH)
                if h['failure']:failures[label]+=1
            assert item['orukeet']['words']==item['parakeet']['words'] and item['orukeet']['chars']==item['parakeet']['chars']
            items.append(item)
            if r['lang']=='en':enitems.append(enitem)
            audiohashes[audio[uid]['float32_pcm_sha256']]+=1
            # Recomputable numeric evidence without source audio/text or direct identifiers.
            number={k:v for k,v in item.items() if k not in ['uid','cluster','region','gender','accent']};number.update(record_sha256=fingerprint(uid),cluster_sha256=fingerprint(item['cluster']),cluster_index=cluster_indices[item['cluster']],reference_sha256=fingerprint(r['text']),audio_sha256=audio[uid]['audio_file_sha256'],pcm_sha256=audio[uid]['float32_pcm_sha256'])
            if r['lang']=='en':number['english_standard']={label:enitem[label] for label in ['parakeet','orukeet']}
            numeric.append(number)
        allrows.extend(items);sets[bucket]=aggregate(items);sets[bucket]['failures']=dict(failures);draws[bucket],_,interval=paired_bootstrap(items,seal['bootstrap_replicates'],seal['seed']+si);sets[bucket]['uncertainty']=interval
        strict=[r for r in items if r['strict_text_unseen']];strict_sets[bucket]=aggregate(strict);strict_draws[bucket],_,ci=paired_bootstrap(strict,seal['bootstrap_replicates'],seal['seed']+si);strict_sets[bucket]['uncertainty']=ci
        if enitems:
            english_sets[bucket]=aggregate(enitems);en_draws[bucket],_,ci=paired_bootstrap(enitems,seal['bootstrap_replicates'],seal['seed']+si);english_sets[bucket]['uncertainty']=ci
            strict_en=[r for r in enitems if r['strict_text_unseen']];strict_english_sets[bucket]=aggregate(strict_en);strict_en_draws[bucket],_,ci=paired_bootstrap(strict_en,seal['bootstrap_replicates'],seal['seed']+si);strict_english_sets[bucket]['uncertainty']=ci
        integrity[bucket]={'rows':len(items),'manifest_sha256':info['sha256'],'completion_sha256':sha(complete),'duplicate_audio_content_rows':sum(n-1 for n in audiohashes.values()),'known_historical_pcm_overlap_rows':sum(effective_overlap.values()),'original_pcm_overlap_rows':sum(r['known_pcm_overlap'] for r in audio.values()),'decoded_history_additional_overlap_rows':sum(effective_overlap[uid] and not r['known_pcm_overlap'] for uid,r in audio.items()),'declared_duration_mismatch_rows':sum(r.get('declared_duration_mismatch',False) for r in audio.values()),'declared_hours':None if any(r['duration'] is None for r in raw) else sum(r['duration'] for r in raw)/3600,'decoded_hours':sum(r['actual_duration'] for r in audio.values())/3600,'zero_legacy_reference_word_rows':sum(r['parakeet']['words']==0 for r in items),'zero_english_reference_word_rows':sum(r['parakeet']['words']==0 for r in enitems),'orukeet_fewer_word_errors_rows':sum(r['orukeet']['errors']<r['parakeet']['errors'] for r in items),'orukeet_more_word_errors_rows':sum(r['orukeet']['errors']>r['parakeet']['errors'] for r in items),'equal_word_errors_rows':sum(r['orukeet']['errors']==r['parakeet']['errors'] for r in items)}
        print('SCORED',bucket,len(items),round(sets[bucket]['parakeet']['wer'],3),round(sets[bucket]['orukeet']['wer'],3),flush=True)
    if missing and not a.allow_partial:raise SystemExit('Incomplete splits: '+','.join(missing))
    endpoints={'eurospeech_language_macro':[s for s in seal['sets'] if s.startswith('eurospeech_')],
      'gigaspeechbench_accent_macro':['gigaspeechbench_'+c+'_en' for c in ['chn','ind','jpn','phl','sct','sgp']],
      'gigaspeechbench_domain_macro':['gigaspeechbench_'+c+'_en' for c in ['agr','ait','art','bio','ecm','eng','ent','fin','hum','law','med','mil']],
      'monsoon_english':['monsoon_en_in']}
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'complete' if not missing else 'partial','missing_splits':missing,'seal_sha256':sealhash,'models':{label:seal[label+'_sha256'] for label in ['parakeet','orukeet']},'scoring_script_sha256':sha(__file__),'normalizer_script_sha256':sha(Path(__file__).parent/'metrics.py'),'decoded_history_audit_sha256':sha(decoded_audit_path),'decoded_history_fingerprints_sha256':sha(decoded_path),'totals':aggregate(allrows),'sets':sets,'strict_history_disjoint_sets':strict_sets,'english_standard_sets':english_sets,'strict_history_disjoint_english_standard_sets':strict_english_sets,'endpoints':{},'integrity':integrity}
    for name,names in endpoints.items():
        if not names or any(n not in sets for n in names):continue
        result['endpoints'][name]={'legacy':macro_endpoint(names,sets,draws),'strict_history_disjoint':macro_endpoint(names,strict_sets,strict_draws)}
        if all(n in english_sets for n in names):
            result['endpoints'][name]['english_standard']=macro_endpoint(names,english_sets,en_draws)
            result['endpoints'][name]['strict_history_disjoint_english_standard']=macro_endpoint(names,strict_english_sets,strict_en_draws)
    result['slices']={}
    for dimension in ['lang','source','gender','region','accent']:
        groups=defaultdict(list)
        for r in allrows:
            key=r[dimension]
            if key!='unknown':groups[key].append(r)
        result['slices'][dimension]={key:aggregate(values) for key,values in sorted(groups.items())}
    result['slices']['duration_seconds']={label:aggregate([r for r in allrows if lo<=r['duration']<hi]) for label,lo,hi in [('under_5',0,5),('5_to_15',5,15),('15_to_30',15,30),('30_and_over',30,float('inf'))]}
    a.output.mkdir(parents=True,exist_ok=True);atomic_json(a.output/'comparison.json',result)
    with gzip.open(a.output/'numeric-evidence.jsonl.gz','wt') as f:
        for r in numeric:f.write(json.dumps(r,sort_keys=True)+'\n')
    atomic_json(a.output/'comparison-receipt.json',{'status':'complete','rows':len(numeric),'seal_sha256':sealhash,
        'files':{name:{'sha256':sha(a.output/name),'bytes':(a.output/name).stat().st_size} for name in ['comparison.json','numeric-evidence.jsonl.gz']},
        'written_utc':datetime.now(timezone.utc).isoformat()})
    print('COMPARISON_COMPLETE',result['status'],len(allrows),flush=True)

if __name__=='__main__':main()
