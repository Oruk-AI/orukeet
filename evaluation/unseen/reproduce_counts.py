#!/usr/bin/env python3
"""Independently reproduce WER/CER and paired intervals with only NumPy.

No source audio, transcripts, ASR runtime, or model downloads are required.
"""
import argparse, gzip, hashlib, json
from collections import defaultdict
from pathlib import Path
import numpy as np


def close(got,want):
    assert np.allclose(got,want,rtol=0,atol=1e-9),(got,want)


def summarize(rows,expected,seed,replicates,english=False):
    assert len(rows)==expected['rows']
    def count(row,model):return row['english_standard'][model] if english else row[model]
    for model in ['parakeet','orukeet']:
        totals={key:sum(count(r,model)[key] for r in rows) for key in count(rows[0],model)}
        for key,value in totals.items():assert value==expected[model][key],key
        close(100*totals['errors']/totals['words'],expected[model]['wer'])
        close(100*totals['char_errors']/totals['chars'],expected[model]['cer'])
    if not all(r['cluster_metadata_available'] for r in rows):
        assert expected['clusters'] is None and 'wer_delta_pp_95ci' not in expected['uncertainty'];return None
    groups=defaultdict(lambda:np.zeros(3))
    for row in rows:
        b=count(row,'parakeet');c=count(row,'orukeet')
        groups[row['cluster_index']]+=np.array([b['errors'],c['errors'],b['words']])
    assert len(groups)==expected['clusters']
    if len(groups)<2:
        assert 'wer_delta_pp_95ci' not in expected['uncertainty'];return None
    data=np.array([groups[k] for k in sorted(groups)]);rng=np.random.default_rng(seed);draws=[]
    for start in range(0,replicates,128):
        indices=rng.integers(len(data),size=(min(128,replicates-start),len(data)))
        totals=data[indices].sum(axis=1);totals=totals[totals[:,2]>0]
        draws.append(100*totals[:,:2]/totals[:,2:])
    draws=np.concatenate(draws);ci=np.quantile(draws[:,1]-draws[:,0],[.025,.975])
    close(ci,expected['uncertainty']['wer_delta_pp_95ci'])
    relative=100*(draws[:,0]-draws[:,1])/np.where(draws[:,0]>0,draws[:,0],np.nan)
    close(np.nanquantile(relative,[.025,.975]),expected['uncertainty']['relative_wer_reduction_pct_95ci'])
    return draws


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--phases',nargs='+',choices=['primary','coverage','followup'],default=['primary','coverage','followup'])
    a=p.parse_args();phases={};inputs={}
    for name,folder in [('primary',a.evidence),('coverage',a.evidence/'coverage'),('followup',a.evidence/'alignment-followup')]:
        if name not in a.phases:continue
        protocol=json.loads((folder/'protocol.json').read_text());comparison=json.loads((folder/'comparison.json').read_text())
        assert comparison['status']=='complete';records=defaultdict(list)
        with gzip.open(folder/'numeric-evidence.jsonl.gz','rt') as f:
            for line in f:
                row=json.loads(line);records[row['bucket']].append(row)
        assert set(records)==set(protocol['sets'])==set(comparison['sets']);draws=defaultdict(dict);checked=0
        variants=[('legacy','sets',False,False),('strict_history_disjoint','strict_history_disjoint_sets',True,False),
                  ('english_standard','english_standard_sets',False,True),('strict_history_disjoint_english_standard','strict_history_disjoint_english_standard_sets',True,True)]
        for i,key in enumerate(sorted(protocol['sets'])):
            rows=records[key];assert len(rows)==protocol['sets'][key]['rows']
            assert len({r['record_sha256'] for r in rows})==len(rows)
            for variant,field,strict,english in variants:
                if key not in comparison[field]:continue
                subset=[r for r in rows if not strict or r['strict_text_unseen']]
                draws[variant][key]=summarize(subset,comparison[field][key],protocol['seed']+i,protocol['bootstrap_replicates'],english)
                checked+=1
        for endpoint,variants in comparison['endpoints'].items():
            for variant,value in variants.items():
                field=next(f for v,f,_,_ in [('legacy','sets',False,False),('strict_history_disjoint','strict_history_disjoint_sets',True,False),('english_standard','english_standard_sets',False,True),('strict_history_disjoint_english_standard','strict_history_disjoint_english_standard_sets',True,True)] if v==variant)
                keys=value['sets']
                for model in ['parakeet','orukeet']:close(np.mean([comparison[field][k][model]['wer'] for k in keys]),value[model+'_wer'])
                if all(draws[variant][k] is not None for k in keys):
                    matrix=np.mean([draws[variant][k] for k in keys],axis=0)
                    close(np.quantile(matrix[:,1]-matrix[:,0],[.025,.975]),value['wer_delta_pp_95ci'])
                    close(np.quantile(100*(matrix[:,0]-matrix[:,1])/matrix[:,0],[.025,.975]),value['relative_wer_reduction_pct_95ci'])
                else:assert 'wer_delta_pp_95ci' not in value
        phases[name]={'rows':sum(map(len,records.values())),'splits':len(records),'scoring_variants_checked':checked,'endpoint_groups':len(comparison['endpoints'])}
        inputs[name]={filename:hashlib.sha256((folder/filename).read_bytes()).hexdigest() for filename in ['protocol.json','comparison.json','numeric-evidence.jsonl.gz','comparison-receipt.json']}
        print('REPRODUCED',name,phases[name],flush=True)
    receipt={'status':'passed','method':'Independent NumPy reduction and seeded paired cluster bootstrap from transcript-free numeric counts.',
             'phases':phases,'inputs':inputs,'numpy':np.__version__,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    filename='count-reproduction.json' if set(phases)=={'primary','coverage','followup'} else 'count-reproduction-'+'-'.join(phases)+'.json'
    (a.evidence/filename).write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
